use crate::map::is_wall;

pub struct Player {
    pub x: f32,
    pub y: f32,
    pub angle: f32,
}

pub fn move_forward(player: &mut Player, amount: f32) {
    let dx = player.angle.cos() * amount;
    let dy = player.angle.sin() * amount;
    if !is_wall(player.x + dx, player.y) {
        player.x += dx;
    }
    if !is_wall(player.x, player.y + dy) {
        player.y += dy;
    }
}

pub fn move_backward(player: &mut Player, amount: f32) {
    move_forward(player, -amount);
}

pub fn turn_left(player: &mut Player, amount: f32) {
    player.angle -= amount;
}

pub fn turn_right(player: &mut Player, amount: f32) {
    player.angle += amount;
}
