use crate::player::Player;
use crate::raycast::{WallSide, raycast_dda};

pub const WIDTH: u32 = 320;
pub const HEIGHT: u32 = 200;

pub fn render(frame: &mut [u8], player: &Player) {
    let ceil = [0x20u8, 0x20, 0x30, 0xFF];
    let floor = [0x30u8, 0x30, 0x20, 0xFF];

    for (i, pixel) in frame.chunks_exact_mut(4).enumerate() {
        let y = i / WIDTH as usize;
        if y < (HEIGHT / 2) as usize {
            pixel.copy_from_slice(&ceil);
        } else {
            pixel.copy_from_slice(&floor);
        }
    }

    let fov = std::f32::consts::PI / 3.0;

    for x in 0..WIDTH {
        let ray_angle = player.angle - fov / 2.0 + (x as f32 / WIDTH as f32) * fov;
        let hit = raycast_dda(player, ray_angle);

        let perp_dist = hit.distance;
        if perp_dist < 0.001 {
            continue;
        }

        let line_height = (HEIGHT as f32 / perp_dist) as i32;
        let draw_start = ((HEIGHT as i32 - line_height) / 2).max(0) as u32;
        let draw_end = ((HEIGHT as i32 + line_height) / 2).min(HEIGHT as i32 - 1) as u32;

        let shade = (255.0 / perp_dist).clamp(0.0, 255.0) as u8;
        let shade = match hit.side {
            WallSide::NorthSouth => shade,
            WallSide::EastWest => shade >> 1,
        };

        draw_vertical_line(frame, x, draw_start, draw_end, shade, shade, shade);
    }
}

fn draw_vertical_line(frame: &mut [u8], x: u32, y_start: u32, y_end: u32, r: u8, g: u8, b: u8) {
    for y in y_start..=y_end {
        let idx = ((y * WIDTH + x) * 4) as usize;
        frame[idx] = r;
        frame[idx + 1] = g;
        frame[idx + 2] = b;
        frame[idx + 3] = 0xFF;
    }
}
