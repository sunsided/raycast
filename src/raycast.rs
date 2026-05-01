//! # Raycast Module - Digital Differential Analyzer (DDA) Raycasting
//!
//! Implements the classic DDA (Digital Differential Analyzer) algorithm for
//! finding where a ray from the player's position first intersects a wall in
//! the 2D grid map. This is the heart of any raycasting engine.
//!
//! ## What is DDA?
//!
//! The DDA algorithm is an efficient grid traversal technique that steps through
//! a 2D grid cell by cell along a ray's path. Instead of testing every point
//! along the ray (which would be wasteful), it computes exactly which cell
//! boundary the ray crosses next and jumps directly to it.
//!
//! The algorithm was popularized for raycasting by the article "A Fast Voxel
//! Traversal Algorithm for Ray Tracing" by Amanatides & Woo (1987).
//!
//! ## How DDA Works
//!
//! Given a ray starting at the player's position and traveling in a direction:
//!
//! 1. **Initialize**: Determine which map cell the player is currently in, and
//!    compute the distance to the next cell boundary on each axis.
//!
//! 2. **Step**: Compare the distances to the next X and Y boundaries. Whichever
//!    is smaller is the boundary the ray reaches first - step to that cell.
//!
//! 3. **Repeat**: Update the distance to the *next* boundary on the axis we
//!    stepped along, then go back to step 2.
//!
//! 4. **Terminate**: When we step into a cell that contains a wall, we've found
//!    the hit point.
//!
//! ```text
//!     Player at (2.5, 2.0), ray heading northeast:
//!
//!     ┌───┬───┬───┬───┐
//!     │   │   │   │   │
//!     ├───┼───┼───┼───┤
//!     │   │ P ╱   │   │  ← P = player, ╱ = ray path
//!     ├───┼───┼═══╪═══┤  ← ═ = wall hit
//!     │   │   ╱   │ # │
//!     └───┴───┴───┴───┘
//!
//!     The ray passes through empty cells until it hits the wall cell (#).
//!     DDA finds this in exactly 2 steps: first X, then Y.
//! ```
//!
//! ## Perpendicular Distance vs. Euclidean Distance
//!
//! A common mistake in raycasting is using the raw Euclidean distance from the
//! ray hit to calculate the wall height. This creates a **fisheye lens effect**
//! where walls appear curved because rays at the edge of the field of view
//! travel a longer diagonal distance.
//!
//! The correct approach is to use the **perpendicular distance** - the distance
//! from the player to the wall, measured perpendicular to the wall surface. This
//! is computed from the raw hit distance by accounting for the ray's angle
//! relative to the wall normal.
//!
//! The formula used here avoids trigonometry by using the `step` direction:
//! ```text
//! perp_dist = (map_x - player_x + (1 - step_x) / 2) / dir_x   (for vertical walls)
//! perp_dist = (map_y - player_y + (1 - step_y) / 2) / dir_y   (for horizontal walls)
//! ```
//!
//! This is mathematically equivalent to `raw_dist * cos(ray_angle - player_angle)`
//! but faster to compute.

use crate::map::is_wall;
use crate::player::Player;

/// Identifies which face of a wall cell was hit by a ray.
///
/// In a 2D grid, walls have two orientations:
/// - **North/South** - Vertical walls (parallel to the Y axis). These are the
///   left and right faces of a wall cell. Hit when the ray crosses a vertical
///   grid line.
/// - **East/West** - Horizontal walls (parallel to the X axis). These are the
///   top and bottom faces of a wall cell. Hit when the ray crosses a horizontal
///   grid line.
///
/// This distinction is important for **shading**. By making one wall side darker
/// than the other, the renderer creates the illusion of directional lighting
/// and makes corners and room geometry more visually distinct - a technique
/// pioneered in Wolfenstein 3D.
///
/// ```text
///        North
///         ↑
///    ┌────┼────┐
///    │    │    │
/// W  │    │    │  E   ← East/West wall (horizontal face)
/// e  │    │    │  a   ← Hit when ray crosses horizontal grid line
/// s  │    │    │  s
/// t  │    │    │  t
///    │    │    │
///    ├────┼────┤
///         ↓
///       South
///
///     ↑
///   North/South wall (vertical face)
///   Hit when ray crosses vertical grid line
/// ```
#[derive(PartialEq)]
pub enum WallSide {
    /// A vertical wall face (parallel to the Y axis).
    ///
    /// Hit when the ray crosses a vertical grid boundary (X changes by ±1).
    /// Typically rendered brighter to simulate light coming from the side.
    NorthSouth,
    /// A horizontal wall face (parallel to the X axis).
    ///
    /// Hit when the ray crosses a horizontal grid boundary (Y changes by ±1).
    /// Typically rendered darker for shading contrast.
    EastWest,
}

/// The result of a successful ray cast.
///
/// Contains the perpendicular distance to the wall (corrected to avoid
/// fisheye distortion) and which side of the wall was hit (for shading).
pub struct RayHit {
    /// Perpendicular distance from the player to the wall.
    ///
    /// This is **not** the raw Euclidean distance along the ray. It is the
    /// distance measured perpendicular to the wall surface, which is what
    /// you need for correct wall-height projection without fisheye distortion.
    ///
    /// For a wall directly in front of the player, this equals the ray distance.
    /// For a wall at an angle, this is shorter (the adjacent side of the right
    /// triangle formed by the ray and the perpendicular).
    ///
    /// Small values mean the wall is close (appears tall). Large values mean
    /// the wall is far (appears short). A value near zero indicates the player
    /// is very close to a wall.
    pub distance: f32,

    /// Which face of the wall was hit.
    ///
    /// Used by the renderer to apply different shading - walls hit on their
    /// East/West face are typically drawn darker to simulate directional
    /// lighting and improve depth perception.
    pub side: WallSide,
}

/// Casts a single ray from the player's position using the DDA algorithm.
///
/// This is the core algorithm of the raycasting engine. For a given ray angle
/// (relative to world coordinates, not the player's facing direction), it finds
/// the first wall the ray intersects and returns the perpendicular distance
/// and which wall face was hit.
///
/// # Algorithm Details
///
/// ## Initialization
///
/// 1. **Direction vector**: `(cos(angle), sin(angle))` - the unit direction
///    the ray travels in.
///
/// 2. **Starting cell**: The map cell the player is currently in, found by
///    truncating the player's world coordinates to integers.
///
/// 3. **Delta distances**: How far the ray must travel to cross one full cell
///    on each axis. Computed as `1 / |direction|`. If the ray is nearly
///    parallel to an axis, this value is very large (it takes a long time
///    to cross cells on that axis).
///
/// 4. **Step direction and initial side distance**: For each axis:
///    - `step` is +1 or -1 depending on whether the ray travels in the
///      positive or negative direction on that axis.
///    - `side_dist` is the distance from the ray origin to the *first*
///      cell boundary on that axis. This is the "partial cell" distance
///      before the regular stepping pattern begins.
///
/// ## Main Loop
///
/// The loop compares `side_dist_x` and `side_dist_y`:
/// - The smaller one is the next boundary the ray reaches.
/// - We step to the adjacent cell on that axis.
/// - We update the side distance by adding the delta (distance to cross
///   one full cell).
/// - We check if the new cell contains a wall. If yes, we're done.
///
/// ## Distance Calculation
///
/// Once a wall is found, we compute the perpendicular distance to avoid
/// fisheye distortion. The formula uses the hit cell coordinate, the player
/// position, the step direction, and the ray direction:
///
/// ```text
/// For a vertical wall (North/South):
///   perp_dist = (map_x - player_x + (1 - step_x) / 2) / dir_x
///
/// For a horizontal wall (East/West):
///   perp_dist = (map_y - player_y + (1 - step_y) / 2) / dir_y
/// ```
///
/// The `(1 - step) / 2` term adjusts for which face was hit:
/// - If `step = 1` (ray travels in positive direction), the adjustment is 0.
/// - If `step = -1` (ray travels in negative direction), the adjustment is 1.
///
/// # Arguments
///
/// * `player` - The player whose position is the ray origin.
/// * `ray_angle` - The angle of the ray in world coordinates (radians).
///   This is typically computed as `player.angle - fov/2 + (x/WIDTH) * fov`
///   to spread rays across the player's field of view.
///
/// # Returns
///
/// A [`RayHit`] containing the perpendicular distance to the first wall hit
/// and which wall face was struck.
///
/// # Performance
///
/// The DDA algorithm runs in O(N) where N is the number of cells the ray
/// traverses before hitting a wall. In a typical 8×8 map, this is usually
/// 1-5 iterations. The algorithm is very branch-predictor-friendly because
/// the comparison `side_dist_x < side_dist_y` has consistent patterns for
/// similar ray angles.
///
/// # Edge Cases
///
/// - **Ray parallel to an axis**: One of `dir_x` or `dir_y` is zero, making
///   the corresponding delta distance infinite. The algorithm naturally handles
///   this - it will only step on the other axis since the infinite delta
///   distance will never be the minimum.
/// - **Player very close to a wall**: The initial side distance may be very
///   small or zero. The algorithm still works correctly.
/// - **Player inside a wall**: The loop immediately detects the wall and
///   returns a distance of zero (or near zero). The renderer handles this
///   by skipping the frame for that column.
pub fn raycast_dda(player: &Player, ray_angle: f32) -> RayHit {
    // Compute the ray's direction vector.
    // This is a unit vector: sqrt(dir_x^2 + dir_y^2) = 1.0
    let dir_x = ray_angle.cos();
    let dir_y = ray_angle.sin();

    // Determine which map cell the player is currently in.
    // Truncation (not floor) is used here because the player position
    // is always positive in our coordinate system.
    let mut map_x = player.x as i32;
    let mut map_y = player.y as i32;

    // delta_dist_x: distance the ray must travel to cross one full cell on the X axis.
    // If dir_x = 0.5, delta_dist_x = 2.0 - it takes 2 units of ray travel to
    // cross from one vertical grid line to the next.
    // If dir_x ≈ 0 (ray nearly horizontal), delta_dist_x → ∞ - the ray barely
    // moves in X, so it takes a very long time to cross cells horizontally.
    let delta_dist_x = 1.0 / dir_x.abs();
    let delta_dist_y = 1.0 / dir_y.abs();

    // For each axis, compute:
    // 1. step: +1 or -1, the direction to the next cell boundary
    // 2. side_dist: distance from ray origin to the FIRST cell boundary
    //
    // For the positive direction (dir >= 0):
    //   step = +1
    //   side_dist = (map_x + 1 - player_x) * delta_dist_x
    //   This is the distance from the player to the RIGHT edge of their current cell.
    //
    // For the negative direction (dir < 0):
    //   step = -1
    //   side_dist = (player_x - map_x) * delta_dist_x
    //   This is the distance from the player to the LEFT edge of their current cell.
    //
    // Visual example for positive X:
    //   ┌─────────┬─────────┐
    //   │  player │         │
    //   │    •───→│ next    │
    //   │  map_x  │ map_x+1 │
    //   └─────────┴─────────┘
    //   side_dist = (map_x + 1 - player_x) * delta_dist_x
    let (step_x, mut side_dist_x) = if dir_x < 0.0 {
        (-1, (player.x - map_x as f32) * delta_dist_x)
    } else {
        (1, (map_x as f32 + 1.0 - player.x) * delta_dist_x)
    };
    let (step_y, mut side_dist_y) = if dir_y < 0.0 {
        (-1, (player.y - map_y as f32) * delta_dist_y)
    } else {
        (1, (map_y as f32 + 1.0 - player.y) * delta_dist_y)
    };

    // Main DDA loop: step through cells until we hit a wall.
    // Each iteration advances to the next cell boundary (X or Y, whichever
    // is closer) and checks if that cell contains a wall.
    //
    // The #[allow(unused_assignments)] is needed because `side` is assigned
    // in both branches of the if/else, but the compiler doesn't know that
    // the loop always executes at least once (since the player starts in
    // an empty cell and the map is surrounded by walls).
    #[allow(unused_assignments)]
    let mut side = WallSide::NorthSouth;
    loop {
        // Step in the direction of the nearest cell boundary.
        if side_dist_x < side_dist_y {
            // X boundary is closer: step horizontally.
            side_dist_x += delta_dist_x; // Distance to the NEXT X boundary
            map_x += step_x;
            side = WallSide::NorthSouth; // Hit a vertical wall face
        } else {
            // Y boundary is closer: step vertically.
            side_dist_y += delta_dist_y; // Distance to the NEXT Y boundary
            map_y += step_y;
            side = WallSide::EastWest; // Hit a horizontal wall face
        }

        // Check if the new cell contains a wall.
        // The map is guaranteed to have walls around its border, so this
        // loop always terminates.
        if is_wall(map_x as f32, map_y as f32) {
            break;
        }
    }

    // Compute the perpendicular distance to the wall.
    //
    // The raw distance along the ray would produce a fisheye effect because
    // rays at the edge of the FOV travel farther than rays in the center,
    // even if they hit walls at the same "depth."
    //
    // The perpendicular distance is the length of the line from the player
    // to the wall, measured perpendicular to the wall surface. This is the
    // "correct" distance for rendering walls at the right apparent height.
    //
    // The formula:
    //   (map_x - player_x + (1 - step_x) / 2) / dir_x
    //
    // breaks down as:
    // - `map_x - player_x`: signed distance from player to the wall cell
    // - `(1 - step_x) / 2`: adjusts for which face of the cell was hit
    //   - If step_x = 1: adjustment = 0 (hit the far face of previous cell)
    //   - If step_x = -1: adjustment = 1 (hit the near face of the wall cell)
    // - Division by dir_x: projects onto the ray direction
    //
    // This is mathematically equivalent to raw_dist * cos(ray_angle - player_angle)
    // but avoids the trigonometric call and is more numerically stable.
    let perp_dist = if side == WallSide::NorthSouth {
        (map_x as f32 - player.x + (1.0 - step_x as f32) / 2.0) / dir_x
    } else {
        (map_y as f32 - player.y + (1.0 - step_y as f32) / 2.0) / dir_y
    };

    RayHit {
        distance: perp_dist,
        side,
    }
}
