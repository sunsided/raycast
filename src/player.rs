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
/// # Why `f32`?
///
/// `f32` is used instead of `f64` because:
/// 1. Precision beyond ~7 significant digits is unnecessary for a game world
///    that's only 8×8 units.
/// 2. `f32` operations are faster on most hardware, especially GPUs.
/// 3. The `pixels` crate and most game engines use `f32` throughout.
pub struct Player {
    /// X coordinate in world space. Must be within `[0, MAP_WIDTH]`.
    pub x: f32,

    /// Y coordinate in world space. Must be within `[0, MAP_HEIGHT]`.
    pub y: f32,

    /// Viewing direction in radians.
    ///
    /// Measured counter-clockwise from the positive X axis (east):
    /// - `0.0` → facing east (right)
    /// - `π/2` → facing north (up)
    /// - `π` → facing west (left)
    /// - `3π/2` → facing south (down)
    ///
    /// The angle is **not** normalized to `[0, 2π)` — it can grow
    /// indefinitely through repeated turning. This is intentional and safe
    /// because `sin()` and `cos()` handle any input value correctly, and
    /// normalizing would add unnecessary computation per frame.
    pub angle: f32,
}

/// Moves the player forward in their current viewing direction.
///
/// Converts the player's angle to a direction vector using `cos()`/`sin()`,
/// then applies movement on each axis independently with collision detection.
/// The independent-axis check is what enables wall-sliding behavior: if one
/// axis is blocked by a wall, the other axis still moves.
pub fn move_forward(player: &mut Player, amount: f32) {
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
/// Delegates to [`move_forward`] with a negated amount.
pub fn move_backward(player: &mut Player, amount: f32) {
    move_forward(player, -amount);
}

/// Rotates the player counter-clockwise (to the left).
///
/// Decreases the player's angle, rotating counter-clockwise on the 2D plane.
/// Note: because the screen coordinate system has Y increasing downward, the
/// visual effect appears counter-clockwise to the player.
pub fn turn_left(player: &mut Player, amount: f32) {
    player.angle -= amount;
}

/// Rotates the player clockwise (to the right). Inverse of [`turn_left`].
pub fn turn_right(player: &mut Player, amount: f32) {
    player.angle += amount;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::{MAP_HEIGHT, MAP_WIDTH};
    use std::f32::consts::PI;

    const EPS: f32 = 1e-3;

    #[test]
    fn turn_left_decreases_angle() {
        let mut p = Player {
            x: 2.0,
            y: 2.0,
            angle: 0.0,
        };
        turn_left(&mut p, 0.1);
        assert!((p.angle - (-0.1)).abs() < EPS);
    }

    #[test]
    fn turn_right_increases_angle() {
        let mut p = Player {
            x: 2.0,
            y: 2.0,
            angle: 0.0,
        };
        turn_right(&mut p, 0.1);
        assert!((p.angle - 0.1).abs() < EPS);
    }

    #[test]
    fn turn_left_right_inverse() {
        let mut p = Player {
            x: 2.0,
            y: 2.0,
            angle: 0.5,
        };
        turn_left(&mut p, 0.3);
        turn_right(&mut p, 0.3);
        assert!((p.angle - 0.5).abs() < EPS);
    }

    #[test]
    fn turning_does_not_move() {
        let mut p = Player {
            x: 2.5,
            y: 3.7,
            angle: 0.0,
        };
        turn_left(&mut p, 1.0);
        assert!((p.x - 2.5).abs() < EPS);
        assert!((p.y - 3.7).abs() < EPS);
        turn_right(&mut p, 2.0);
        assert!((p.x - 2.5).abs() < EPS);
        assert!((p.y - 3.7).abs() < EPS);
    }

    #[test]
    fn move_forward_along_positive_x() {
        let mut p = Player {
            x: 1.5,
            y: 1.5,
            angle: 0.0,
        };
        move_forward(&mut p, 0.05);
        assert!((p.x - 1.55).abs() < EPS);
        assert!((p.y - 1.5).abs() < EPS);
    }

    #[test]
    fn move_forward_blocked_by_wall() {
        let mut p = Player {
            x: 1.5,
            y: 1.5,
            angle: PI,
        };
        let old_x = p.x;
        move_forward(&mut p, 10.0);
        assert!((p.x - old_x).abs() < EPS);
    }

    #[test]
    fn move_forward_sliding_collision() {
        let mut p = Player {
            x: 2.5,
            y: 2.5,
            angle: PI / 4.0,
        };
        move_forward(&mut p, 1.0);
        assert!((p.x - 2.5).abs() < EPS);
        let expected_y = 2.5 + (PI / 4.0).sin();
        assert!((p.y - expected_y).abs() < EPS);
    }

    #[test]
    fn move_backward_is_negative_forward() {
        let mut p1 = Player {
            x: 3.0,
            y: 3.0,
            angle: 0.5,
        };
        let mut p2 = Player {
            x: 3.0,
            y: 3.0,
            angle: 0.5,
        };
        move_forward(&mut p1, -0.05);
        move_backward(&mut p2, 0.05);
        assert!((p1.x - p2.x).abs() < EPS);
        assert!((p1.y - p2.y).abs() < EPS);
        assert!((p1.angle - p2.angle).abs() < EPS);
    }

    #[test]
    fn move_keeps_player_in_map_property() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut p = Player {
            x: 4.0,
            y: 4.0,
            angle: 0.0,
        };
        let step = 0.05;
        for i in 0..100u64 {
            let mut h = DefaultHasher::new();
            i.hash(&mut h);
            let seed = h.finish() as f32;
            let angle = seed * 0.1;
            p.angle = angle;
            move_forward(&mut p, step);
            assert!(p.x >= 0.0 && p.x < MAP_WIDTH as f32, "x={}", p.x);
            assert!(p.y >= 0.0 && p.y < MAP_HEIGHT as f32, "y={}", p.y);
        }
    }
}
