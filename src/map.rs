pub const MAP: [&str; 8] = [
    "########", "#......#", "#..##..#", "#......#", "#......#", "#..##..#", "#......#", "########",
];

pub const MAP_WIDTH: usize = 8;
pub const MAP_HEIGHT: usize = 8;

pub fn is_wall(x: f32, y: f32) -> bool {
    let mx = x.floor() as i32;
    let my = y.floor() as i32;
    if mx < 0 || mx >= MAP_WIDTH as i32 || my < 0 || my >= MAP_HEIGHT as i32 {
        return true;
    }
    MAP[my as usize].as_bytes()[mx as usize] == b'#'
}
