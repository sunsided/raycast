//! # Render Module - Frame Buffer Rendering
//!
//! Handles drawing a single frame to the pixel buffer. This is the final stage
//! of the raycasting pipeline: it takes the player state, casts one ray per
//! screen column, and draws vertical wall slices with distance-based shading.
//!
//! ## Rendering Pipeline
//!
//! Each frame goes through two passes:
//!
//! 1. **Background pass**: Fill the entire frame buffer with ceiling and floor
//!    colors. The top half is the ceiling (dark blue-gray), the bottom half
//!    is the floor (dark yellow-gray). This is done as a single linear scan
//!    over the flat RGBA byte array.
//!
//! 2. **Wall pass**: For each screen column (0 to `WIDTH-1`):
//!    a. Compute the ray angle for this column based on the player's facing
//!       direction and the field of view (FOV).
//!    b. Cast the ray using DDA to find the first wall hit.
//!    c. Compute the wall slice height from the perpendicular distance.
//!    d. Draw a vertical line with brightness inversely proportional to distance.
//!
//! ## Frame Buffer Layout
//!
//! The [`pixels`](https://crates.io/crates/pixels) crate provides a flat
//! `&mut [u8]` buffer where each pixel is 4 consecutive bytes in RGBA order:
//!
//! ```text
//!     [R, G, B, A,  R, G, B, A,  R, G, B, A,  ...]
//!      ^ pixel 0    ^ pixel 1    ^ pixel 2
//!
//!     Pixel (x, y) starts at byte index: (y * WIDTH + x) * 4
//! ```
//!
//! The buffer contains `WIDTH * HEIGHT * 4` bytes total.
//!
//! ## Field of View
//!
//! The FOV is `π/3` radians (60°), matching the human eye's natural
//! perspective and the classic Doom/Wolfenstein feel. Each of the `WIDTH`
//! rays covers an equal angular slice:
//!
//! ```text
//!     player.angle - 30°         player.angle         player.angle + 30°
//!           ↓                        ↓                        ↓
//!     ──────┼────────────────────────┼────────────────────────┼──────
//!           │←─── column 0 ────────→│←── column WIDTH-1 ────→│
//! ```
//!
//! ## Distance Shading
//!
//! Walls are shaded using a simple inverse-distance model:
//! ```
//! shade = clamp(255 / distance, 0, 255)
//! ```
//!
//! This means:
//! - A wall at distance 1.0 is fully bright (255)
//! - A wall at distance 2.0 is half as bright (127)
//! - A wall at distance 4.0 is quarter as bright (63)
//! - Walls at distance ≥ 255 are fully black
//!
//! Additionally, walls hit on their East/West face are rendered at half
//! brightness (`shade >> 1`) to simulate directional lighting and make
//! room corners visually distinct.

use crate::player::Player;
use crate::raycast::{WallSide, raycast_dda};

/// Internal render width in pixels.
///
/// The frame buffer is `WIDTH × HEIGHT` pixels, but the window is typically
/// displayed at 3× this size for a pixel-art aesthetic. The GPU's nearest-neighbor
/// upscaling preserves the crisp, blocky look.
///
/// At 320 pixels wide with a 60° FOV, each column represents 60/320 ≈ 0.1875°
/// of the player's field of view - more than enough resolution to perceive
/// wall geometry smoothly.
pub const WIDTH: u32 = 320;

/// Internal render height in pixels.
///
/// The screen is split evenly: the top `HEIGHT/2` rows are ceiling, the
/// bottom `HEIGHT/2` rows are floor. Wall slices are vertically centered,
/// extending equally above and below the midpoint.
pub const HEIGHT: u32 = 200;

/// Renders one complete frame into the pixel buffer.
///
/// This is the main rendering function, called once per frame. It completely
/// overwrites the frame buffer - there is no incremental or dirty-rectangle
/// rendering.
///
/// # Arguments
///
/// * `frame` - Mutable reference to the RGBA pixel buffer provided by the
///   [`pixels`] crate. Must be at least `WIDTH * HEIGHT * 4` bytes long.
///   Each group of 4 bytes represents one pixel in RGBA order.
/// * `player` - The current player state, used to determine ray origins
///   and directions.
///
/// # Performance
///
/// This function casts one ray per screen column (`WIDTH = 320` rays total).
/// Each ray traverses the map using DDA, which typically takes 1-5 cell
/// steps in an 8×8 map. The total work per frame is roughly:
/// - 320 ray casts × ~3 average steps = ~960 cell checks
/// - 320 vertical lines drawn × ~50 average height = ~16,000 pixel writes
///
/// At 60 FPS, this is well within the capabilities of modern hardware,
/// even without optimization.
pub fn render(frame: &mut [u8], player: &Player) {
    // Ceiling and floor colors in RGBA format.
    // The ceiling is a cool dark blue-gray (#202030), the floor is a
    // warm dark yellow-gray (#303020). This color contrast helps players
    // distinguish up from down, which is surprisingly important for
    // spatial orientation in a 3D-like view.
    //
    // The alpha channel is 0xFF (fully opaque) since we don't use
    // transparency.
    let ceil = [0x20u8, 0x20, 0x30, 0xFF];
    let floor = [0x30u8, 0x30, 0x20, 0xFF];

    // Pass 1: Fill the entire frame buffer with ceiling and floor colors.
    // The frame buffer is a flat [u8] array with 4 bytes per pixel (RGBA).
    // We iterate over each pixel by grouping the buffer into 4-byte chunks.
    //
    // chunks_exact_mut(4) yields non-overlapping mutable slices of exactly
    // 4 bytes each. If the buffer size isn't a multiple of 4, the remainder
    // is silently dropped (which shouldn't happen with a valid frame buffer).
    //
    // The pixel index `i` is the pixel number in row-major order:
    //   i = 0          → pixel at (0, 0)  - top-left
    //   i = WIDTH      → pixel at (0, 1)  - left edge, second row
    //   i = WIDTH * y  → pixel at (0, y)  - left edge, row y
    for (i, pixel) in frame.chunks_exact_mut(4).enumerate() {
        let y = i / WIDTH as usize;
        if y < (HEIGHT / 2) as usize {
            // Top half of the screen: ceiling
            pixel.copy_from_slice(&ceil);
        } else {
            // Bottom half of the screen: floor
            pixel.copy_from_slice(&floor);
        }
    }

    // Field of view: 60 degrees (π/3 radians).
    // This is a classic choice that matches human perspective well and
    // was used in many classic games. A wider FOV would make the scene
    // feel more "fish-eye" and increase motion sickness; a narrower FOV
    // would feel like looking through binoculars.
    let fov = std::f32::consts::PI / 3.0;

    // Pass 2: Cast one ray per column and draw wall slices.
    // For each screen column, we:
    // 1. Compute the ray angle that this column represents
    // 2. Cast the ray to find the nearest wall
    // 3. Calculate how tall the wall appears at that distance
    // 4. Draw a vertical line with appropriate shading
    for x in 0..WIDTH {
        // Compute the ray angle for this column.
        //
        // The rays span from (player.angle - fov/2) to (player.angle + fov/2),
        // evenly distributed across the screen width.
        //
        // For column x:
        //   ray_angle = player.angle - fov/2 + (x / WIDTH) * fov
        //
        // When x = 0:         ray_angle = player.angle - fov/2  (leftmost ray)
        // When x = WIDTH/2:   ray_angle = player.angle          (center ray)
        // When x = WIDTH - 1: ray_angle ≈ player.angle + fov/2  (rightmost ray)
        //
        // The division (x as f32 / WIDTH as f32) produces a value in [0, 1)
        // that is then scaled to the FOV range.
        let ray_angle = player.angle - fov / 2.0 + (x as f32 / WIDTH as f32) * fov;
        let hit = raycast_dda(player, ray_angle);

        // The perpendicular distance to the wall. We use perpendicular distance
        // (not Euclidean) to avoid the fisheye lens effect.
        let perp_dist = hit.distance;

        // Guard against division by zero or near-zero distances.
        // If the player is standing right against a wall, perp_dist can be
        // extremely small, causing the wall slice height to overflow to
        // infinity. Skip this column - the wall is too close to render
        // meaningfully (it would fill the entire column anyway).
        if perp_dist < 0.001 {
            continue;
        }

        // Calculate the height of the wall slice to draw in this column.
        //
        // The formula is: line_height = HEIGHT / perp_dist
        //
        // This comes from the principle of similar triangles. A wall of
        // height 1 (one map cell) at distance d appears on screen with
        // height proportional to 1/d. We scale this so that a wall at
        // distance 1.0 fills exactly the screen height.
        //
        // Examples:
        //   perp_dist = 1.0 → line_height = 200 (fills entire screen)
        //   perp_dist = 2.0 → line_height = 100 (half screen height)
        //   perp_dist = 4.0 → line_height = 50  (quarter screen height)
        let line_height = (HEIGHT as f32 / perp_dist) as i32;

        // Calculate the vertical range to draw the wall slice.
        // Walls are vertically centered on the screen:
        //   draw_start = (HEIGHT - line_height) / 2  - top of wall
        //   draw_end   = (HEIGHT + line_height) / 2  - bottom of wall
        //
        // The .max(0) and .min(HEIGHT - 1) clamping ensures we don't
        // write outside the frame buffer when the wall is very close
        // (line_height > HEIGHT).
        let draw_start = ((HEIGHT as i32 - line_height) / 2).max(0) as u32;
        let draw_end = ((HEIGHT as i32 + line_height) / 2).min(HEIGHT as i32 - 1) as u32;

        // Compute the brightness of the wall based on distance.
        //
        // The formula shade = 255 / perp_dist creates an inverse-distance
        // falloff that is more aggressive than linear but simpler than
        // the physically correct inverse-square law.
        //
        // The .clamp(0.0, 255.0) ensures the result fits in a u8:
        // - Distances < 1.0 would produce values > 255, clamped to 255
        // - Distances > 255 would produce values < 1, clamped to 0
        //
        // We then darken East/West walls by shifting right by 1 (dividing by 2).
        // This creates the classic "one side of the wall is darker" effect
        // that helps players perceive room geometry and corners.
        let shade = (255.0 / perp_dist).clamp(0.0, 255.0) as u8;
        let shade = match hit.side {
            WallSide::NorthSouth => shade,     // Brighter (lit side)
            WallSide::EastWest => shade >> 1,  // Darker (shadowed side)
        };

        // Draw the vertical wall line for this column.
        // We use the same shade value for R, G, and B to produce a
        // grayscale wall. The color is purely determined by distance
        // and wall side - no textures in this basic engine.
        draw_vertical_line(frame, x, draw_start, draw_end, shade, shade, shade);
    }
}

/// Draws a vertical line in the frame buffer.
///
/// Writes a solid-colored vertical line from `y_start` to `y_end` (inclusive)
/// in column `x`. Each pixel is set to the specified RGBA color.
///
/// This is the lowest-level drawing primitive in the engine - there are no
/// higher-level abstractions like sprites, textures, or shapes. Everything
/// visible on screen ultimately goes through this function.
///
/// # Arguments
///
/// * `frame` - Mutable reference to the RGBA pixel buffer.
/// * `x` - The column to draw in (0 to `WIDTH - 1`).
/// * `y_start` - The top row of the line (0 to `HEIGHT - 1`).
/// * `y_end` - The bottom row of the line (inclusive, 0 to `HEIGHT - 1`).
/// * `r` - Red component (0 = none, 255 = full).
/// * `g` - Green component (0 = none, 255 = full).
/// * `b` - Blue component (0 = none, 255 = full).
///
/// # Buffer Indexing
///
/// The byte offset for pixel `(x, y)` in the flat RGBA buffer is:
/// ```text
///     index = (y * WIDTH + x) * 4
/// ```
///
/// Breaking this down:
/// 1. `y * WIDTH` - number of pixels in all rows above row `y`
/// 2. `+ x` - add the column offset within row `y`
/// 3. `* 4` - convert pixel index to byte index (4 bytes per pixel: RGBA)
///
/// The alpha channel is always set to `0xFF` (fully opaque).
///
/// # Safety
///
/// This function assumes `x`, `y_start`, and `y_end` are within the valid
/// range. The caller (`render`) is responsible for ensuring the bounds are
/// correct via the `.max(0)` and `.min(HEIGHT - 1)` clamping. Out-of-bounds
/// access would cause a panic.
fn draw_vertical_line(frame: &mut [u8], x: u32, y_start: u32, y_end: u32, r: u8, g: u8, b: u8) {
    for y in y_start..=y_end {
        // Calculate the byte offset for this pixel in the flat RGBA buffer.
        // See the module-level documentation for the indexing formula.
        let idx = ((y * WIDTH + x) * 4) as usize;
        frame[idx] = r;
        frame[idx + 1] = g;
        frame[idx + 2] = b;
        frame[idx + 3] = 0xFF; // Alpha: fully opaque
    }
}
