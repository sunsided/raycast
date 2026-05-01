//! # Map Module - 2D World Grid
//!
//! Defines the game world as a simple 2D grid of characters where each cell
//! is either a wall (`#`) or empty space (`.`). This is the classic approach
//! used in early raycasting engines like Wolfenstein 3D.
//!
//! ## Coordinate System
//!
//! The map uses a **row-major** layout where:
//! - `MAP[y][x]` accesses the cell at column `x`, row `y`
//! - The origin `(0, 0)` is the **top-left** corner
//! - X increases to the right, Y increases downward
//!
//! ```text
//!     Y=0  ########
//!     Y=1  #......#
//!     Y=2  #..##..#
//!            ↑
//!          X=2
//! ```
//!
//! ## World vs. Map Coordinates
//!
//! The player exists in **world coordinates** (continuous `f32` values like
//! `x = 2.5, y = 3.7`), not discrete map indices. To check if a world position
//! is inside a wall, we use `floor()` to convert to map coordinates:
//!
//! ```text
//!     world (2.5, 3.7) → map cell (2, 3) → MAP[3][2]
//! ```
//!
//! This means the integer part of a world coordinate is the map cell index,
//! and the fractional part is the position within that cell.

/// The game map as an array of strings. Each character determines what occupies
/// that cell: `#` for a solid wall, `.` for empty space.
///
/// The map is 8×8 cells, surrounded by walls on all four edges to prevent
/// the player from escaping into an undefined area. The interior contains
/// a small maze-like structure with a central wall segment.
///
/// ```text
///     01234567  ← X (columns)
///   0 ########
///   1 #......#
///   2 #..##..#
///   3 #......#
///   4 #......#
///   5 #..##..#
///   6 #......#
///   7 ########
///   ↑
///   Y (rows)
/// ```
pub const MAP: [&str; 8] = [
    "########", "#......#", "#..##..#", "#......#", "#......#", "#..##..#", "#......#", "########",
];

/// Number of columns (X dimension) in the [`MAP`].
pub const MAP_WIDTH: usize = 8;

/// Number of rows (Y dimension) in the [`MAP`].
pub const MAP_HEIGHT: usize = 8;

/// Checks whether a point in world coordinates is inside a wall.
///
/// Converts world coordinates to discrete map cell indices using `floor()`,
/// then looks up the cell in [`MAP`]. Out-of-bounds positions are treated
/// as walls to prevent the player from escaping and ensure rays always hit.
pub fn is_wall(x: f32, y: f32) -> bool {
    // Convert continuous world coordinates to discrete map cell indices.
    // floor() ensures that positions 2.0 through 2.999... all map to cell 2.
    let mx = x.floor() as i32;
    let my = y.floor() as i32;

    // Out-of-bounds positions are treated as walls. This serves two purposes:
    // 1. Prevents the player from walking off the edge of the map into undefined space.
    // 2. Guarantees that rays always eventually hit something, preventing infinite loops.
    if mx < 0 || mx >= MAP_WIDTH as i32 || my < 0 || my >= MAP_HEIGHT as i32 {
        return true;
    }

    MAP[my as usize].as_bytes()[mx as usize] == b'#'
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_walls() {
        assert!(is_wall(0.5, 0.5));
        assert!(is_wall(7.5, 0.5));
        assert!(is_wall(3.5, 2.5));
        assert!(is_wall(4.5, 2.5));
        assert!(is_wall(3.5, 5.5));
        assert!(is_wall(4.5, 5.5));
    }

    #[test]
    fn known_open_cells() {
        assert!(!is_wall(1.5, 1.5));
        assert!(!is_wall(3.5, 3.5));
        assert!(!is_wall(6.5, 6.5));
        assert!(!is_wall(1.5, 3.5));
        assert!(!is_wall(5.5, 4.5));
    }

    #[test]
    fn out_of_bounds_is_wall() {
        assert!(is_wall(-0.01, 1.5));
        assert!(is_wall(1.5, -0.01));
        assert!(is_wall(8.0, 1.5));
        assert!(is_wall(1.5, 8.0));
        assert!(is_wall(-1.0, -1.0));
        assert!(is_wall(99.0, 99.0));
    }

    #[test]
    fn floor_semantics() {
        assert!(!is_wall(2.0, 2.0));
        assert!(!is_wall(2.999, 2.0));
        assert!(is_wall(3.0, 2.0));
        assert!(!is_wall(1.999, 1.999));
        assert!(!is_wall(2.999, 1.999));
        assert!(!is_wall(2.999, 3.999));
        assert!(is_wall(3.0, 5.0));
    }

    #[test]
    fn map_dimensions_sanity() {
        assert_eq!(MAP.len(), MAP_HEIGHT);
        assert!(MAP.iter().all(|r| r.len() == MAP_WIDTH));
        assert_eq!(MAP_WIDTH, 8);
        assert_eq!(MAP_HEIGHT, 8);
    }
}
