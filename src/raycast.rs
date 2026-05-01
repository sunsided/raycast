//! # Raycast Module - Digital Differential Analyzer (DDA) Raycasting
//!
//! Implements the classic DDA algorithm for finding where a ray from the
//! player's position first intersects a wall in the 2D grid map.
//!
//! DDA is an efficient grid traversal technique that steps cell by cell
//! along a ray's path, computing exactly which cell boundary the ray
//! crosses next and jumping directly to it. Popularized by Amanatides & Woo
//! (1987) in "A Fast Voxel Traversal Algorithm for Ray Tracing."
//!
//! ## Perpendicular Distance vs. Euclidean Distance
//!
//! Using raw Euclidean distance creates a fisheye lens effect because rays
//! at the edge of the field of view travel a longer diagonal distance. The
//! correct approach is perpendicular distance — the distance from the player
//! to the wall measured perpendicular to the wall surface. The formula avoids
//! trigonometry by using the step direction:
//! ```text
//! perp_dist = (map_x - player_x + (1 - step_x) / 2) / dir_x   (vertical walls)
//! perp_dist = (map_y - player_y + (1 - step_y) / 2) / dir_y   (horizontal walls)
//! ```
//! This is equivalent to `raw_dist * cos(ray_angle - player_angle)` but faster.

use crate::map::is_wall;
use crate::player::Player;

/// Identifies which face of a wall cell was hit by a ray.
///
/// - **North/South** — vertical wall face (parallel to Y axis). Hit when the
///   ray crosses a vertical grid line.
/// - **East/West** — horizontal wall face (parallel to X axis). Hit when the
///   ray crosses a horizontal grid line.
///
/// This distinction matters for shading: one face is drawn darker to create
/// the illusion of directional lighting and make corners visually distinct.
#[derive(PartialEq, Debug)]
pub enum WallSide {
    /// Hit when the ray crosses a vertical grid boundary.
    /// Typically rendered brighter to simulate side lighting.
    NorthSouth,
    /// Hit when the ray crosses a horizontal grid boundary.
    /// Typically rendered darker for shading contrast.
    EastWest,
}

/// Result of a successful ray cast.
///
/// Contains the perpendicular distance to the wall (corrected to avoid
/// fisheye distortion) and which side of the wall was hit (for shading).
pub struct RayHit {
    /// Perpendicular distance from the player to the wall.
    ///
    /// This is **not** the raw Euclidean distance along the ray. It is the
    /// distance measured perpendicular to the wall surface, needed for correct
    /// wall-height projection without fisheye distortion.
    ///
    /// For a wall directly in front of the player, this equals the ray distance.
    /// For a wall at an angle, this is shorter. Small values mean the wall is
    /// close (appears tall); large values mean the wall is far (appears short).
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
/// For a given ray angle (in world coordinates), finds the first wall the ray
/// intersects and returns the perpendicular distance and which wall face was hit.
///
/// The algorithm initializes direction vector, starting cell, delta distances,
/// and step direction/side distance for each axis. The main loop then compares
/// side distances, steps to the nearest cell boundary, and checks for walls.
/// Once a wall is found, perpendicular distance is computed using the formula:
/// ```text
/// perp_dist = (map_x - player_x + (1 - step_x) / 2) / dir_x   (vertical walls)
/// perp_dist = (map_y - player_y + (1 - step_y) / 2) / dir_y   (horizontal walls)
/// ```
pub fn raycast_dda(player: &Player, ray_angle: f32) -> RayHit {
    let dir_x = ray_angle.cos();
    let dir_y = ray_angle.sin();

    let mut map_x = player.x as i32;
    let mut map_y = player.y as i32;

    // Distance to cross one full cell on each axis.
    // If dir ≈ 0, delta_dist → ∞ (ray barely moves on that axis).
    let delta_dist_x = 1.0 / dir_x.abs();
    let delta_dist_y = 1.0 / dir_y.abs();

    // step: direction to the next cell boundary (+1 or -1)
    // side_dist: distance from ray origin to the first cell boundary on that axis
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

    // Main DDA loop: advance to the nearest cell boundary and check for walls.
    // The map is surrounded by walls, so this always terminates.
    // The #[allow] is needed because `side` is assigned in both branches,
    // but the compiler can't prove the loop always runs at least once.
    #[allow(unused_assignments)]
    let mut side = WallSide::NorthSouth;
    loop {
        if side_dist_x < side_dist_y {
            side_dist_x += delta_dist_x;
            map_x += step_x;
            side = WallSide::NorthSouth;
        } else {
            side_dist_y += delta_dist_y;
            map_y += step_y;
            side = WallSide::EastWest;
        }

        if is_wall(map_x as f32, map_y as f32) {
            break;
        }
    }

    // Compute perpendicular distance to avoid fisheye distortion.
    // The (1 - step) / 2 term adjusts for which face of the cell was hit:
    // step = 1 → adjustment = 0; step = -1 → adjustment = 1.
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    const EPS: f32 = 1e-3;

    fn make_player(x: f32, y: f32, angle: f32) -> Player {
        Player { x, y, angle }
    }

    #[test]
    fn facing_east_hits_northsouth_wall() {
        let p = make_player(1.5, 1.5, 0.0);
        let hit = raycast_dda(&p, 0.0);
        assert!((hit.distance - 5.5).abs() < EPS);
        assert_eq!(hit.side, WallSide::NorthSouth);
    }

    #[test]
    fn facing_west_hits_northsouth_wall() {
        let p = make_player(1.5, 1.5, PI);
        let hit = raycast_dda(&p, PI);
        assert!((hit.distance - 0.5).abs() < EPS);
        assert_eq!(hit.side, WallSide::NorthSouth);
    }

    #[test]
    fn facing_south_hits_eastwest_wall() {
        let p = make_player(1.5, 1.5, PI / 2.0);
        let hit = raycast_dda(&p, PI / 2.0);
        assert!((hit.distance - 5.5).abs() < EPS);
        assert_eq!(hit.side, WallSide::EastWest);
    }

    #[test]
    fn facing_north_hits_eastwest_wall() {
        let p = make_player(1.5, 1.5, -PI / 2.0);
        let hit = raycast_dda(&p, -PI / 2.0);
        assert!((hit.distance - 0.5).abs() < EPS);
        assert_eq!(hit.side, WallSide::EastWest);
    }

    #[test]
    fn diagonal_into_pillar() {
        let p = make_player(1.5, 1.5, PI / 4.0);
        let hit = raycast_dda(&p, PI / 4.0);
        assert!(hit.distance > 0.0);
        assert!(hit.distance.is_finite());
    }

    #[test]
    fn no_fisheye_property() {
        let p = make_player(4.0, 4.0, PI / 2.0);
        let offset = 0.1;
        let hit_left = raycast_dda(&p, PI / 2.0 + offset);
        let hit_right = raycast_dda(&p, PI / 2.0 - offset);
        assert!(
            (hit_left.distance - hit_right.distance).abs() < EPS,
            "left={}, right={}",
            hit_left.distance,
            hit_right.distance
        );
    }

    #[test]
    fn rayhit_distance_is_positive() {
        let p = make_player(4.0, 4.0, 0.0);
        for i in 0..200 {
            let angle = (i as f32) * 0.0314;
            let hit = raycast_dda(&p, angle);
            assert!(hit.distance > 0.0, "distance={}", hit.distance);
            assert!(hit.distance.is_finite(), "distance=NaN or inf");
        }
    }
}
