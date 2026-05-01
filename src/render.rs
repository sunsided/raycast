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
//!    direction and the field of view (FOV).
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
/// The window is typically displayed at 3× this size for a pixel-art aesthetic.
/// GPU nearest-neighbor upscaling preserves the crisp, blocky look.
pub const WIDTH: u32 = 320;

/// Internal render height in pixels.
///
/// The screen is split evenly: top half is ceiling, bottom half is floor.
/// Wall slices are vertically centered.
pub const HEIGHT: u32 = 200;

/// Renders one complete frame into the pixel buffer.
///
/// Two passes: fill ceiling/floor background, then cast one ray per column
/// to draw wall slices with distance-based shading.
pub fn render(frame: &mut [u8], player: &Player) {
    let ceil = [0x20u8, 0x20, 0x30, 0xFF];
    let floor = [0x30u8, 0x30, 0x20, 0xFF];

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

    // FOV: 60 degrees (π/3 radians), matching human perspective and classic games.
    let fov = std::f32::consts::PI / 3.0;

    for x in 0..WIDTH {
        // Spread rays evenly across FOV:
        // x=0 → player.angle - fov/2 (leftmost)
        // x=WIDTH/2 → player.angle (center)
        // x=WIDTH-1 → ≈ player.angle + fov/2 (rightmost)
        let ray_angle = player.angle - fov / 2.0 + (x as f32 / WIDTH as f32) * fov;
        let hit = raycast_dda(player, ray_angle);

        let perp_dist = hit.distance;

        // Skip if too close — prevents overflow in line_height.
        if perp_dist < 0.001 {
            continue;
        }

        // Wall slice height from similar triangles:
        // distance 1.0 → fills screen (200px); distance 2.0 → half screen.
        let line_height = (HEIGHT as f32 / perp_dist) as i32;

        // Vertically centered; clamped to avoid out-of-bounds writes.
        let draw_start = ((HEIGHT as i32 - line_height) / 2).max(0) as u32;
        let draw_end = ((HEIGHT as i32 + line_height) / 2).min(HEIGHT as i32 - 1) as u32;

        // Inverse-distance shading: shade = 255 / perp_dist, clamped to [0, 255].
        // East/West walls are halved for the "one side darker" corner effect.
        let shade = (255.0 / perp_dist).clamp(0.0, 255.0) as u8;
        let shade = match hit.side {
            WallSide::NorthSouth => shade,
            WallSide::EastWest => shade >> 1,
        };

        draw_vertical_line(frame, x, draw_start, draw_end, shade, shade, shade);
    }
}

/// Draws a vertical line in the frame buffer.
///
/// Writes a solid-colored vertical line from `y_start` to `y_end` (inclusive)
/// in column `x`. The byte offset for pixel `(x, y)` in the flat RGBA buffer is
/// `(y * WIDTH + x) * 4`. Alpha is always `0xFF` (fully opaque).
fn draw_vertical_line(frame: &mut [u8], x: u32, y_start: u32, y_end: u32, r: u8, g: u8, b: u8) {
    for y in y_start..=y_end {
        let idx = ((y * WIDTH + x) * 4) as usize;
        frame[idx] = r;
        frame[idx + 1] = g;
        frame[idx + 2] = b;
        frame[idx + 3] = 0xFF; // Alpha: fully opaque
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::player::Player;

    const CEIL: [u8; 4] = [0x20, 0x20, 0x30, 0xFF];
    const FLOOR: [u8; 4] = [0x30, 0x30, 0x20, 0xFF];

    fn pixel(buf: &[u8], x: u32, y: u32) -> [u8; 4] {
        let idx = ((y * WIDTH + x) * 4) as usize;
        [buf[idx], buf[idx + 1], buf[idx + 2], buf[idx + 3]]
    }

    #[test]
    fn width_and_height_constants() {
        assert_eq!(WIDTH, 320);
        assert_eq!(HEIGHT, 200);
    }

    #[test]
    fn render_ceiling_fill() {
        let mut buf = vec![0u8; (WIDTH * HEIGHT * 4) as usize];
        let p = Player {
            x: 1.5,
            y: 1.5,
            angle: 0.0,
        };
        render(&mut buf, &p);
        let col = WIDTH / 2;
        assert_eq!(pixel(&buf, col, 0), CEIL);
    }

    #[test]
    fn render_floor_fill() {
        let mut buf = vec![0u8; (WIDTH * HEIGHT * 4) as usize];
        let p = Player {
            x: 1.5,
            y: 1.5,
            angle: 0.0,
        };
        render(&mut buf, &p);
        let col = WIDTH / 2;
        assert_eq!(pixel(&buf, col, HEIGHT - 1), FLOOR);
    }

    #[test]
    fn render_alpha_always_ff() {
        let mut buf = vec![0u8; (WIDTH * HEIGHT * 4) as usize];
        let p = Player {
            x: 1.5,
            y: 1.5,
            angle: 0.0,
        };
        render(&mut buf, &p);
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let idx = ((y * WIDTH + x) * 4 + 3) as usize;
                assert_eq!(buf[idx], 0xFF, "alpha at ({}, {}) != 0xFF", x, y);
            }
        }
    }

    #[test]
    fn render_buffer_length_invariant() {
        let mut buf = vec![0u8; (WIDTH * HEIGHT * 4) as usize];
        let p = Player {
            x: 1.5,
            y: 1.5,
            angle: 0.0,
        };
        render(&mut buf, &p);
        assert_eq!(buf.len(), (WIDTH * HEIGHT * 4) as usize);
    }

    #[test]
    fn render_draws_wall_slices() {
        let mut buf = vec![0u8; (WIDTH * HEIGHT * 4) as usize];
        let p = Player {
            x: 1.5,
            y: 1.5,
            angle: 0.0,
        };
        render(&mut buf, &p);
        let col = WIDTH / 2;
        let mut found_wall = false;
        for y in 0..HEIGHT {
            let px = pixel(&buf, col, y);
            if px != CEIL && px != FLOOR {
                found_wall = true;
                break;
            }
        }
        assert!(found_wall, "center column should contain wall pixels");
    }
}
