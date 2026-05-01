//! # Raycast - A Wolfenstein 3D-Style Raycasting Engine
//!
//! This is a classic raycasting engine that renders a pseudo-3D first-person view
//! from a 2D grid-based map, similar to the technique used in Wolfenstein 3D (1992).
//!
//! ## Architecture Overview
//!
//! The engine is split into four modules, each handling a distinct concern:
//!
//! - **`map`** - Defines the 2D world as a grid of wall (`#`) and empty (`.`) cells.
//!   Provides collision detection by checking whether arbitrary coordinates fall
//!   inside a wall cell.
//!
//! - **`player`** - Tracks the player's position (`x`, `y`) and viewing direction
//!   (`angle`). Handles movement with sliding wall collision: if the player tries
//!   to walk into a wall on one axis, they still slide along the other axis.
//!
//! - **`raycast`** - Implements the Digital Differential Analyzer (DDA) algorithm
//!   to cast a single ray from the player's position and find the first wall hit.
//!   Returns the perpendicular distance (not Euclidean) to avoid the "fisheye"
//!   distortion and which wall side was hit (for shading).
//!
//! - **`render`** - Renders one frame per vertical column of the screen. For each
//!   column it casts a ray, computes how tall the wall slice should appear based
//!   on distance, and draws a vertical line with distance-based shading. Ceiling
//!   and floor are drawn as solid colors.
//!
//! ## How Raycasting Works
//!
//! The core idea is simple: for each vertical column of pixels on the screen, cast
//! a single ray from the player's eye position in the direction that column
//! represents. Where the ray hits a wall determines how tall that wall slice
//! appears - closer walls are taller, farther walls are shorter. This creates the
//! illusion of 3D from purely 2D data.
//!
//! ```text
//!     Top-down view:          Resulting view:
//!     ┌─────────┐             ┌───────────────┐
//!     │#.......#│             │  ███████████  │
//!     │#  P  > #│    ───►     │  ██ WALL ██   │
//!     │#.......#│             │  ███████████  │
//!     └─────────┘             └───────────────┘
//! ```
//!
//! The DDA algorithm steps through the grid cell by cell (not pixel by pixel),
//! making it extremely efficient - it runs in O(N) where N is the number of
//! cells traversed, typically just a handful.
//!
//! ## Dependencies
//!
//! - **`pixels`** - A minimal 2D pixel buffer with wgpu backend. Provides a simple
//!   `Pixels::frame_mut()` that returns a `&mut [u8]` RGBA buffer we write into.
//!
//! - **`winit`** - Cross-platform window creation and event loop. Handles keyboard
//!   input, window lifecycle, and the render-request callback.

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

/// Tracks which movement keys are currently held down.
///
/// Decouples key-press *events* from per-frame movement logic. Without this,
/// the player would only move one step per press instead of continuously.
///
/// Supports WASD and arrow keys: W/↑=forward, S/↓=back, A/←=left, D/→=right.
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

    /// Maps a physical key code to the corresponding movement direction.
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

/// Holds the window and pixel buffer after they are created.
///
/// The `'static` lifetime on `Pixels` (obtained via `unsafe { std::mem::transmute }`)
/// is safe because the `Arc<Window>` is stored alongside `Pixels` in this struct,
/// and wgpu's surface only holds a raw OS window handle, not a Rust reference.
struct AppInner {
    pixels: Pixels<'static>,
    window: Arc<Window>,
}

/// Top-level application state implementing winit's [`ApplicationHandler`].
///
/// # Lifecycle
///
/// 1. [`App::new()`] — creates the app with no window yet.
/// 2. [`App::resumed()`] — creates the window and pixel buffer.
/// 3. [`App::window_event()`] — handles input and rendering on `RedrawRequested`.
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
    /// Called when the application should create (or recreate) its window.
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.inner.is_some() {
            return;
        }

        // 3× render resolution for pixel-art scaling.
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

        // SAFETY: Arc<Window> is stored alongside Pixels; wgpu Surface holds
        // only a raw OS window handle, not a Rust reference.
        let pixels: Pixels<'static> = unsafe { std::mem::transmute(pixels_raw) };

        self.inner = Some(AppInner { pixels, window });
    }

    /// Handles window events. The three we care about:
    /// `CloseRequested` (exit), `KeyboardInput` (update keys / exit on Escape),
    /// and `RedrawRequested` (apply input, render, request next frame).
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

/// Applies movement input based on currently held keys.
///
/// Forward/backward speed: `0.05` units/frame (~3.0 units/s at 60 FPS).
/// Turn speed: `0.03` rad/frame (~103°/s). Values are hardcoded; a production
/// engine would use delta-time for framerate independence.
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

/// Entry point. Creates the winit event loop, instantiates the app, and runs.
/// Exits on window close or Escape key.
fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
