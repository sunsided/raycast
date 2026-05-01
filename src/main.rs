mod map;
mod player;
mod raycast;
mod render;

use std::sync::Arc;

use pixels::{Pixels, SurfaceTexture};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::Window;

use player::{Player, move_backward, move_forward, turn_left, turn_right};
use render::{HEIGHT, WIDTH, render};

struct KeyState {
    forward: bool,
    back: bool,
    left: bool,
    right: bool,
}

impl KeyState {
    fn new() -> Self {
        Self {
            forward: false,
            back: false,
            left: false,
            right: false,
        }
    }

    fn update(&mut self, key: KeyCode, pressed: bool) {
        match key {
            KeyCode::KeyW | KeyCode::ArrowUp => self.forward = pressed,
            KeyCode::KeyS | KeyCode::ArrowDown => self.back = pressed,
            KeyCode::KeyA | KeyCode::ArrowLeft => self.left = pressed,
            KeyCode::KeyD | KeyCode::ArrowRight => self.right = pressed,
            _ => {}
        }
    }
}

struct AppInner {
    pixels: Pixels<'static>,
    window: Arc<Window>,
}

struct App {
    inner: Option<AppInner>,
    player: Player,
    keys: KeyState,
}

impl App {
    fn new() -> Self {
        Self {
            inner: None,
            player: Player {
                x: 2.0,
                y: 2.0,
                angle: 0.0,
            },
            keys: KeyState::new(),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.inner.is_some() {
            return;
        }

        let size = LogicalSize::new(WIDTH as f64 * 3.0, HEIGHT as f64 * 3.0);
        let window = Arc::new(
            event_loop
                .create_window(
                    winit::window::Window::default_attributes()
                        .with_title("Raycast")
                        .with_inner_size(size),
                )
                .unwrap(),
        );

        let window_size = window.inner_size();
        let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &window);

        let pixels_raw = Pixels::new(WIDTH, HEIGHT, surface_texture).unwrap();

        // SAFETY: The Arc<Window> is stored in AppInner alongside Pixels.
        // The wgpu Surface internally stores only a raw window handle,
        // not a Rust reference, so the lifetime annotation is purely
        // a compile-time check. The Arc keeps the Window alive.
        let pixels: Pixels<'static> = unsafe { std::mem::transmute(pixels_raw) };

        self.inner = Some(AppInner { pixels, window });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let Some(AppInner { pixels, window }) = self.inner.as_mut() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let KeyEvent {
                    physical_key,
                    state,
                    ..
                } = event;
                if let PhysicalKey::Code(key_code) = physical_key {
                    if key_code == KeyCode::Escape {
                        event_loop.exit();
                    } else {
                        self.keys.update(key_code, state == ElementState::Pressed);
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                apply_input(&mut self.player, &self.keys);
                render(pixels.frame_mut(), &self.player);
                pixels.render().unwrap();
                window.request_redraw();
            }
            _ => {}
        }
    }
}

fn apply_input(player: &mut Player, keys: &KeyState) {
    if keys.forward {
        move_forward(player, 0.05);
    }
    if keys.back {
        move_backward(player, 0.05);
    }
    if keys.left {
        turn_left(player, 0.03);
    }
    if keys.right {
        turn_right(player, 0.03);
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
