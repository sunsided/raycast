use crate::map::is_wall;
use crate::player::Player;

#[derive(PartialEq)]
pub enum WallSide {
    NorthSouth,
    EastWest,
}

pub struct RayHit {
    pub distance: f32,
    pub side: WallSide,
}

pub fn raycast_dda(player: &Player, ray_angle: f32) -> RayHit {
    let dir_x = ray_angle.cos();
    let dir_y = ray_angle.sin();

    let mut map_x = player.x as i32;
    let mut map_y = player.y as i32;

    let delta_dist_x = 1.0 / dir_x.abs();
    let delta_dist_y = 1.0 / dir_y.abs();

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
