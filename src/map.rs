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

/// The game map as an array of strings.
///
/// Each string is one row of the map. The character at `MAP[y][x]` determines
/// what occupies that cell:
/// - `#` - Solid wall (blocks movement and rays)
/// - `.` - Empty space (player can walk through, rays pass through)
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
///
/// To visualize this as a top-down view, each `#` is a wall block and each
/// `.` is floor space the player can walk on. The inner walls at columns
/// 3-4 in rows 2 and 5 form two obstacles in the middle of the room.
pub const MAP: [&str; 8] = [
    "########",
    "#......#",
    "#..##..#",
    "#......#",
    "#......#",
    "#..##..#",
    "#......#",
    "########",
];

/// Number of columns (X dimension) in the [`MAP`].
///
/// This equals the length of each row string. Used for boundary checks
/// in [`is_wall()`] to determine if a coordinate falls outside the map.
pub const MAP_WIDTH: usize = 8;

/// Number of rows (Y dimension) in the [`MAP`].
///
/// This equals the number of strings in the [`MAP`] array. Used for
/// boundary checks in [`is_wall()`] to determine if a coordinate falls
/// outside the map.
pub const MAP_HEIGHT: usize = 8;

/// Checks whether a point in world coordinates is inside a wall.
///
/// This is the core collision detection function used by both the player
/// movement code (to prevent walking through walls) and the raycasting
/// algorithm (to find where rays hit).
///
/// # How It Works
///
/// 1. **Convert to map coordinates**: Takes the floor of each world coordinate
///    to find which map cell the point falls in. For example, world position
///    `(2.7, 3.2)` maps to cell `(2, 3)`.
///
/// 2. **Boundary check**: If the resulting map coordinates are outside the
///    valid range `[0, MAP_WIDTH)` × `[0, MAP_HEIGHT)`, the function returns
///    `true` - treating out-of-bounds areas as walls. This prevents the player
///    from escaping the map and rays from indexing into invalid memory.
///
/// 3. **Cell lookup**: Accesses `MAP[my][mx]` and checks if the byte at that
///    position equals `b'#'`. Using `.as_bytes()` avoids allocating a String
///    and accesses the underlying UTF-8 bytes directly.
///
/// # Arguments
///
/// * `x` - World X coordinate (continuous, not cell index).
/// * `y` - World Y coordinate (continuous, not cell index).
///
/// # Returns
///
/// `true` if the point is inside a wall cell or outside the map bounds,
/// `false` if it's in an empty space cell.
///
/// # Examples
///
/// ```
/// // Standing in the middle of cell (2, 1) - that's a '.' cell
/// assert!(!is_wall(2.5, 1.5));
///
/// // Standing at the edge of cell (0, 0) - that's a '#' cell
/// assert!(is_wall(0.5, 0.5));
///
/// // Outside the map - treated as wall
/// assert!(is_wall(-1.0, 2.0));
/// assert!(is_wall(10.0, 2.0));
/// ```
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

    // Look up the character in the map. We use as_bytes() to get the raw UTF-8
    // bytes without allocating a String, then index directly. Since the map only
    // contains ASCII characters (# and .), each character is exactly one byte.
    MAP[my as usize].as_bytes()[mx as usize] == b'#'
}
