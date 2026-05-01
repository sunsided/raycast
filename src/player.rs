//! # Player Module - Camera State and Movement
//!
//! Manages the player's position and orientation in the world, along with
//! movement functions that handle collision detection with walls.
//!
//! ## Player Representation
//!
//! The player is a **point camera** in a 2D world defined by:
//! - Position `(x, y)` in continuous world coordinates (each map cell = 1 unit)
//! - Viewing angle in radians, where 0.0 points east (positive X axis)
//!
//! This is the minimal representation needed for a raycasting engine. There is
//! no concept of player height, head bob, or collision radius - the player is
//! an infinitesimal point.
//!
//! ## Movement Model
//!
//! Movement uses **trigonometric decomposition**: the direction angle is
//! converted to a direction vector using `cos()` and `sin()`, then scaled
//! by the movement amount. This is more elegant than storing a direction
//! vector directly because turning is a simple angle addition.
//!
//! ## Sliding Collision
//!
//! [`move_forward()`] implements **axis-separated collision detection**:
//! it tries to move on X first, then on Y independently. If the X movement
//! would hit a wall, it's cancelled, but the Y movement still happens. This
//! produces the classic "wall sliding" behavior seen in Doom and Wolfenstein 3D
//! - you can walk along a wall by holding forward + a diagonal direction.

use crate::map::is_wall;

/// The player's state in the world.
///
/// Represents a point camera with a viewing direction. All coordinates are
/// in **world space** where each map cell is 1×1 unit. A player at `(2.5, 3.0)`
/// is in the middle of column 2, at the top edge of row 3.
///
/// # Coordinate System
///
/// - **X axis**: Increases to the right (east). Column 0 is the leftmost column.
/// - **Y axis**: Increases downward (south). Row 0 is the topmost row.
/// - **Angle**: In radians, measured counter-clockwise from the positive X axis.
///   - `0.0` → facing east (right)
///   - `π/2` → facing north (up)
///   - `π` → facing west (left)
///   - `3π/2` → facing south (down)
///
/// # Why `f32`?
///
/// `f32` is used instead of `f64` because:
/// 1. Precision beyond ~7 significant digits is unnecessary for a game world
///    that's only 8×8 units.
/// 2. `f32` operations are faster on most hardware, especially GPUs.
/// 3. The `pixels` crate and most game engines use `f32` throughout.
pub struct Player {
    /// X coordinate in world space.
    ///
    /// Must be within `[0, MAP_WIDTH]` to be on the map. Values outside this
    /// range mean the player is out of bounds (which shouldn't happen during
    /// normal play due to collision detection in [`move_forward`]).
    pub x: f32,

    /// Y coordinate in world space.
    ///
    /// Must be within `[0, MAP_HEIGHT]` to be on the map. Same constraints
    /// and guarantees as [`x`](Player::x).
    pub y: f32,

    /// Viewing direction in radians.
    ///
    /// Measured counter-clockwise from the positive X axis (east).
    /// The angle is **not** normalized to `[0, 2π)` - it can grow
    /// indefinitely through repeated turning. This is intentional and safe
    /// because `sin()` and `cos()` handle any input value correctly, and
    /// normalizing would add unnecessary computation per frame.
    ///
    /// Note that Rust's trigonometric functions use radians, not degrees.
    /// To convert: `degrees * π / 180 = radians`.
    pub angle: f32,
}

/// Moves the player forward in their current viewing direction.
///
/// This is the primary movement function. It decomposes the player's angle
/// into a direction vector, scales it by the movement amount, then applies
/// movement on each axis independently with collision detection.
///
/// # How It Works
///
/// 1. **Direction vector**: Computes `(cos(angle), sin(angle))` to get the
///    unit direction the player is facing.
///
/// 2. **Scale by amount**: Multiplies the direction by the movement amount
///    (e.g., `0.05` units per frame) to get the displacement `(dx, dy)`.
///
/// 3. **X-axis collision**: Checks if `player.x + dx` would place the player
///    inside a wall. If not, applies the X displacement. If it would hit a
///    wall, the X movement is cancelled but Y is still attempted.
///
/// 4. **Y-axis collision**: Same as X but for the Y axis. This independence
///    is what creates the "wall sliding" effect - you can move along a wall
///    even when pressing diagonally toward it.
///
/// # Sliding Collision Explained
///
/// Consider the player at `(2.0, 2.0)` facing northeast toward a wall at
/// column 3. With the old "all-or-nothing" collision, the player wouldn't
/// move at all. With sliding collision:
/// - X movement is blocked (wall ahead)
/// - Y movement succeeds (no wall in Y direction)
/// - Result: player slides along the wall in the Y direction
///
/// This feels much more natural to players and is the standard approach
/// in first-person games.
///
/// # Arguments
///
/// * `player` - Mutable reference to the player to move.
/// * `amount` - Distance to move in world units. Positive values move forward,
///   negative values move backward (though [`move_backward`] is preferred for
///   clarity).
///
/// # Example
///
/// ```
/// // Player at (2.0, 2.0) facing east (angle = 0.0)
/// let mut player = Player { x: 2.0, y: 2.0, angle: 0.0 };
/// move_forward(&mut player, 0.5);
/// // Player is now at approximately (2.5, 2.0) - moved along X only
/// ```
pub fn move_forward(player: &mut Player, amount: f32) {
    // Compute the displacement vector from the player's facing angle.
    // cos(angle) gives the X component, sin(angle) gives the Y component.
    // This produces a unit vector (length = 1.0) that we scale by `amount`.
    let dx = player.angle.cos() * amount;
    let dy = player.angle.sin() * amount;

    // Try moving on the X axis. If it would hit a wall, skip it.
    // This independent axis check enables wall-sliding behavior.
    if !is_wall(player.x + dx, player.y) {
        player.x += dx;
    }

    // Try moving on the Y axis. Same logic as X - independent check.
    // Even if X movement was blocked, Y movement may still succeed.
    if !is_wall(player.x, player.y + dy) {
        player.y += dy;
    }
}

/// Moves the player backward, opposite to their current viewing direction.
///
/// Implemented as [`move_forward`] with a negated amount. This is purely
/// for API clarity - callers can express intent ("move backward") without
/// having to remember to pass a negative value.
///
/// # Arguments
///
/// * `player` - Mutable reference to the player to move.
/// * `amount` - Distance to move backward (always positive; it's negated internally).
pub fn move_backward(player: &mut Player, amount: f32) {
    move_forward(player, -amount);
}

/// Rotates the player counter-clockwise (to the left).
///
/// Decreases the player's angle, which rotates the viewing direction
/// counter-clockwise on the 2D plane. In a standard mathematical
/// coordinate system (Y up), this would be the positive rotation
/// direction. However, since our screen has Y increasing downward,
/// the visual effect is a counter-clockwise turn.
///
/// # Arguments
///
/// * `player` - Mutable reference to the player to rotate.
/// * `amount` - Rotation amount in radians. For example, `0.03` radians
///   is approximately `1.7°`.
pub fn turn_left(player: &mut Player, amount: f32) {
    player.angle -= amount;
}

/// Rotates the player clockwise (to the right).
///
/// Increases the player's angle, rotating the viewing direction clockwise.
/// This is the inverse of [`turn_left`].
///
/// # Arguments
///
/// * `player` - Mutable reference to the player to rotate.
/// * `amount` - Rotation amount in radians. Should typically match the
///   value used for [`turn_left`] for consistent turning speed.
pub fn turn_right(player: &mut Player, amount: f32) {
    player.angle += amount;
}
