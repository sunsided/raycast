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
/// This decouples key-press *events* (which fire once per state change) from
/// the per-frame movement logic (which needs to know if a key is *still held*).
/// Without this, the player would only move one step per key press instead of
/// continuously while the key is held.
///
/// Supports both WASD and arrow keys as aliases:
/// - `W` / `↑` - move forward
/// - `S` / `↓` - move backward
/// - `A` / `←` - turn left
/// - `D` / `→` - turn right
struct KeyState {
    /// `true` while W or Up arrow is held.
    forward: bool,
    /// `true` while S or Down arrow is held.
    back: bool,
    /// `true` while A or Left arrow is held.
    left: bool,
    /// `true` while D or Right arrow is held.
    right: bool,
}

impl KeyState {
    /// Creates a new `KeyState` with all keys released.
    fn new() -> Self {
        Self {
            forward: false,
            back: false,
            left: false,
            right: false,
        }
    }

    /// Updates the state of a single key.
    ///
    /// Called once for each `KeyboardInput` event. Maps the physical key code
    /// to the corresponding movement direction and sets the boolean to match
    /// the key's pressed/released state.
    ///
    /// # Arguments
    ///
    /// * `key` - The physical key code (layout-independent, so W is always W
    ///   regardless of keyboard language).
    /// * `pressed` - `true` if the key was just pressed, `false` if released.
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
/// The [`Pixels`] struct contains a `wgpu` surface that holds a raw window
/// handle internally. The lifetime parameter on `Pixels<'a>` expresses that
/// the surface must not outlive the window it renders to. Since both the
/// `Arc<Window>` and `Pixels` live together in this struct, the window is
/// guaranteed to outlive the pixels - but the compiler cannot prove this
/// through the type system alone.
///
/// The `'static` lifetime (obtained via `unsafe { std::mem::transmute }` in
/// [`App::resumed`](App::resumed)) is safe here because:
/// 1. The `Arc<Window>` is stored alongside `Pixels`, so the window lives as
///    long as the pixel buffer.
/// 2. The wgpu surface only stores a raw OS window handle, not a Rust reference.
/// 3. Both are destroyed together when `AppInner` is dropped.
struct AppInner {
    /// The pixel buffer backed by a wgpu swap chain. We mutate this buffer
    /// directly each frame via [`Pixels::frame_mut()`], then call
    /// [`Pixels::render()`] to present it to the window.
    pixels: Pixels<'static>,
    /// The OS window, kept alive by an [`Arc`] so it survives even if the
    /// internal wgpu surface only holds a raw handle to it.
    window: Arc<Window>,
}

/// The top-level application state managed by winit's event loop.
///
/// Implements [`ApplicationHandler`], the trait winit calls back into for every
/// event (window creation requests, keyboard input, redraw requests, etc.).
///
/// # Lifecycle
///
/// 1. [`App::new()`] - Creates the app with no window yet. The `inner` field is
///    `None` because winit hasn't asked us to create a window at this point.
/// 2. [`App::resumed()`] - Called when winit wants us to create (or recreate)
///    the window. We build the `Pixels` buffer and store everything in `inner`.
/// 3. [`App::window_event()`] - Called for every window event. On
///    `RedrawRequested`, we process input, render a frame, and request the next
///    redraw - forming the game loop.
struct App {
    /// The window and pixel buffer, created lazily in [`resumed()`](App::resumed).
    /// Uses `Option` so we can take ownership during teardown and support
    /// suspend/resume cycles on some platforms.
    inner: Option<AppInner>,
    /// The player whose position and viewing direction determine what the
    /// camera sees.
    player: Player,
    /// The current state of movement keys, updated by keyboard events and
    /// consumed each frame in [`apply_input()`].
    keys: KeyState,
}

impl App {
    /// Creates a new application with default player state.
    ///
    /// The player starts at position `(2.0, 2.0)` facing angle `0.0` (east /
    /// right along the positive X axis). These coordinates are in "world space"
    /// where each map cell is 1×1 unit.
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
    /// Called when the application is (re)activated and should have a window.
    ///
    /// This is the proper place to create the window and GPU resources in winit
    /// 0.30+, rather than in `main()`. It handles both initial startup and
    /// cases where the app was suspended and needs to recreate its window (e.g.,
    /// on mobile or some Wayland compositors).
    ///
    /// The window is sized to 3× the render resolution (`WIDTH × 3`, `HEIGHT × 3`)
    /// so the pixel art is scaled up while keeping the internal buffer small for
    /// performance. The GPU handles the upscaling via nearest-neighbor by default.
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Guard against duplicate window creation (resumed can be called multiple times).
        if self.inner.is_some() {
            return;
        }

        // Create a window 3× the internal render resolution for pixel-art scaling.
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

        // Create the pixel buffer. The surface texture is sized to the window's
        // physical pixel dimensions (which may differ from LogicalSize on HiDPI
        // displays, but Pixels handles this internally).
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

    /// Handles events for a specific window.
    ///
    /// This is the core of the game loop, called by winit for every event
    /// dispatched to our window. The three events we care about:
    ///
    /// - **`CloseRequested`** - The user clicked the window close button.
    ///   We tell the event loop to exit.
    /// - **`KeyboardInput`** - A key was pressed or released. We update
    ///   [`KeyState`] for continuous movement, or exit on Escape.
    /// - **`RedrawRequested`** - The window needs a new frame. This is where
    ///   we apply player movement, render the scene into the pixel buffer,
    ///   and present it. We then immediately request another redraw to form
    ///   a continuous game loop.
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

/// Applies pending movement input to the player based on currently held keys.
///
/// This function is called once per frame during the [`RedrawRequested`](WindowEvent::RedrawRequested)
/// event, right before rendering. It checks each direction in [`KeyState`] and
/// calls the corresponding movement function with a fixed step size.
///
/// # Movement parameters
///
/// - **Forward/backward speed**: `0.05` world units per frame. At 60 FPS this
///   is `3.0` units/second - fast enough to cross the 8×8 map in about 2.5 seconds.
/// - **Turn speed**: `0.03` radians per frame. At 60 FPS this is `1.8` rad/s,
///   or about 103°/s - a full 360° turn takes roughly 3.5 seconds.
///
/// The values are hardcoded for simplicity. A production engine would use
/// delta-time to make movement framerate-independent.
///
/// # Arguments
///
/// * `player` - Mutable reference to the player to modify.
/// * `keys` - The current key state, read but not modified.
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

/// Entry point of the application.
///
/// Creates the winit event loop, instantiates the application state, and hands
/// control over to winit's event loop. From this point on, the program is
/// entirely event-driven - winit calls back into [`App`] methods as events occur.
///
/// The event loop runs until either:
/// - The user closes the window (`CloseRequested`)
/// - The user presses Escape (`KeyboardInput` with Escape key)
fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
