//! Rootstar BCI Application
//!
//! Unified entry point for the Rootstar Brain-Computer Interface platform.
//! Supports native desktop visualization and WebSocket server modes.
//!
//! # Usage
//!
//! ```bash
//! # Native desktop visualization (default)
//! rootstar
//!
//! # Native viz with USB hardware connection
//! rootstar --device usb
//!
//! # WebSocket server mode for web UI
//! rootstar server --port 8080
//!
//! # Server mode with simulated data
//! rootstar server --simulate
//! ```

use clap::{Parser, Subcommand};
use tracing::{info, warn, Level};
use tracing_subscriber::FmtSubscriber;

/// Rootstar BCI Application
#[derive(Parser, Debug)]
#[command(name = "rootstar")]
#[command(author, version, about = "Rootstar Brain-Computer Interface Platform", long_about = None)]
struct Cli {
    /// Logging verbosity level
    #[arg(short, long, default_value = "info")]
    log_level: String,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run native desktop visualization (default if no subcommand)
    Native {
        /// Device connection type: usb, ble, or simulate
        #[arg(short, long, default_value = "simulate")]
        device: String,

        /// USB port path (e.g., /dev/ttyUSB0 or COM3)
        #[arg(long)]
        port: Option<String>,

        /// Window width
        #[arg(long, default_value = "1280")]
        width: u32,

        /// Window height
        #[arg(long, default_value = "720")]
        height: u32,
    },

    /// Run WebSocket server for web UI
    Server {
        /// Server port
        #[arg(short, long, default_value = "8080")]
        port: u16,

        /// Bind address
        #[arg(short, long, default_value = "127.0.0.1")]
        bind: String,

        /// Device connection type: usb, ble, or simulate
        #[arg(short, long, default_value = "simulate")]
        device: String,

        /// USB port path
        #[arg(long)]
        usb_port: Option<String>,

        /// Serve static web files from this directory
        #[arg(long)]
        static_dir: Option<String>,
    },

    /// List available devices
    Devices {
        /// Device type to scan: usb, ble, or all
        #[arg(short, long, default_value = "all")]
        device_type: String,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let log_level = match cli.log_level.to_lowercase().as_str() {
        "trace" => Level::TRACE,
        "debug" => Level::DEBUG,
        "info" => Level::INFO,
        "warn" => Level::WARN,
        "error" => Level::ERROR,
        _ => Level::INFO,
    };

    let subscriber = FmtSubscriber::builder()
        .with_max_level(log_level)
        .with_target(true)
        .with_thread_ids(false)
        .with_file(false)
        .with_line_number(false)
        .finish();

    tracing::subscriber::set_global_default(subscriber)?;

    info!("Rootstar BCI v{}", env!("CARGO_PKG_VERSION"));

    match cli.command {
        None | Some(Commands::Native { .. }) => {
            run_native(cli.command)?;
        }
        Some(Commands::Server {
            port,
            bind,
            device,
            usb_port,
            static_dir,
        }) => {
            run_server(port, bind, device, usb_port, static_dir)?;
        }
        Some(Commands::Devices { device_type }) => {
            list_devices(&device_type)?;
        }
    }

    Ok(())
}

/// Run native desktop visualization
fn run_native(command: Option<Commands>) -> anyhow::Result<()> {
    #[cfg(feature = "native")]
    {
        use rootstar_bci_native::viz::{run_app, VizConfig};

        let (device, _port, width, height) = match command {
            Some(Commands::Native {
                device,
                port,
                width,
                height,
            }) => (device, port, width, height),
            _ => ("simulate".to_string(), None, 1280, 720),
        };

        info!("Starting native visualization");
        info!("Device mode: {}", device);

        // TODO: Initialize device connection based on mode
        // For now, we run the visualization with simulated data
        if device != "simulate" {
            warn!("Hardware connection not yet implemented, falling back to simulation");
        }

        let config = VizConfig {
            width,
            height,
            title: "Rootstar BCI - SNS Visualization".to_string(),
            ..Default::default()
        };

        run_app(config).map_err(|e| anyhow::anyhow!("{}", e))?;
    }

    #[cfg(not(feature = "native"))]
    {
        let _ = command;
        anyhow::bail!(
            "Native visualization not enabled. Rebuild with --features native:\n\
             cargo run -p rootstar-bci-app --features native"
        );
    }

    Ok(())
}

/// Run WebSocket server for web UI
fn run_server(
    port: u16,
    bind: String,
    device: String,
    usb_port: Option<String>,
    static_dir: Option<String>,
) -> anyhow::Result<()> {
    #[cfg(feature = "server")]
    {
        use std::net::SocketAddr;
        use tokio::runtime::Runtime;

        info!("Starting WebSocket server on {}:{}", bind, port);
        info!("Device mode: {}", device);

        if device != "simulate" {
            if let Some(ref port_path) = usb_port {
                info!("USB port: {}", port_path);
            }
            warn!("Hardware connection not yet implemented, falling back to simulation");
        }

        let rt = Runtime::new()?;
        rt.block_on(async {
            let addr: SocketAddr = format!("{}:{}", bind, port).parse()?;
            run_server_async(addr, device, usb_port, static_dir).await
        })?;
    }

    #[cfg(not(feature = "server"))]
    {
        let _ = (port, bind, device, usb_port, static_dir);
        anyhow::bail!(
            "WebSocket server not enabled. Rebuild with --features server:\n\
             cargo run -p rootstar-bci-app --features server"
        );
    }

    Ok(())
}

/// Async server implementation
#[cfg(feature = "server")]
async fn run_server_async(
    addr: std::net::SocketAddr,
    _device: String,
    _usb_port: Option<String>,
    static_dir: Option<String>,
) -> anyhow::Result<()> {
    use axum::{
        extract::ws::{Message, WebSocket, WebSocketUpgrade},
        extract::State,
        response::IntoResponse,
        routing::get,
        Router,
    };
    use std::sync::Arc;
    use tokio::sync::broadcast;
    use tower_http::cors::{Any, CorsLayer};

    /// Shared application state
    struct AppState {
        /// Broadcast channel for BCI data
        tx: broadcast::Sender<BciFrame>,
    }

    /// BCI data frame sent over WebSocket
    #[derive(Clone, Debug, serde::Serialize)]
    struct BciFrame {
        timestamp_us: u64,
        frame_type: String,
        channels: Vec<f32>,
    }

    /// WebSocket handler
    async fn ws_handler(
        ws: WebSocketUpgrade,
        State(state): State<Arc<AppState>>,
    ) -> impl IntoResponse {
        ws.on_upgrade(move |socket| handle_socket(socket, state))
    }

    /// Handle individual WebSocket connection
    async fn handle_socket(mut socket: WebSocket, state: Arc<AppState>) {
        let mut rx = state.tx.subscribe();

        info!("WebSocket client connected");

        // Send welcome message
        let welcome = serde_json::json!({
            "type": "welcome",
            "version": env!("CARGO_PKG_VERSION"),
            "capabilities": ["eeg", "fnirs", "emg", "eda", "sns"]
        });

        if socket
            .send(Message::Text(welcome.to_string().into()))
            .await
            .is_err()
        {
            return;
        }

        // Forward BCI frames to client
        loop {
            tokio::select! {
                // Receive from broadcast channel
                result = rx.recv() => {
                    match result {
                        Ok(frame) => {
                            let json = match serde_json::to_string(&frame) {
                                Ok(j) => j,
                                Err(_) => continue,
                            };
                            if socket.send(Message::Text(json.into())).await.is_err() {
                                break;
                            }
                        }
                        Err(broadcast::error::RecvError::Lagged(n)) => {
                            warn!("Client lagged, dropped {} frames", n);
                        }
                        Err(broadcast::error::RecvError::Closed) => {
                            break;
                        }
                    }
                }

                // Receive from WebSocket (for control messages)
                result = socket.recv() => {
                    match result {
                        Some(Ok(Message::Text(text))) => {
                            info!("Received control message: {}", text);
                            // TODO: Handle control messages (stim params, etc.)
                        }
                        Some(Ok(Message::Close(_))) | None => {
                            break;
                        }
                        _ => {}
                    }
                }
            }
        }

        info!("WebSocket client disconnected");
    }

    /// Simulated data generator
    async fn simulate_data(tx: broadcast::Sender<BciFrame>) {
        use std::time::{Duration, Instant};

        let start = Instant::now();
        let mut sequence = 0u64;

        loop {
            let elapsed = start.elapsed();
            let timestamp_us = elapsed.as_micros() as u64;

            // Generate simulated EEG (8 channels, ~250 Hz)
            let t = elapsed.as_secs_f32();
            let eeg_channels: Vec<f32> = (0..8)
                .map(|ch| {
                    let phase = ch as f32 * 0.5;
                    // Mix of alpha (10 Hz), beta (20 Hz), and noise
                    10.0 * (2.0 * std::f32::consts::PI * 10.0 * t + phase).sin()
                        + 5.0 * (2.0 * std::f32::consts::PI * 20.0 * t + phase * 2.0).sin()
                        + 2.0 * ((sequence as f32 * 0.1 + ch as f32).sin())
                })
                .collect();

            let frame = BciFrame {
                timestamp_us,
                frame_type: "eeg".to_string(),
                channels: eeg_channels,
            };

            // Ignore send errors (no subscribers)
            let _ = tx.send(frame);

            // Generate fNIRS at lower rate (~10 Hz)
            if sequence % 25 == 0 {
                let fnirs_channels: Vec<f32> = (0..8)
                    .map(|ch| {
                        let phase = ch as f32 * 0.3;
                        // Slow hemodynamic response
                        0.5 + 0.3 * (2.0 * std::f32::consts::PI * 0.1 * t + phase).sin()
                    })
                    .collect();

                let frame = BciFrame {
                    timestamp_us,
                    frame_type: "fnirs".to_string(),
                    channels: fnirs_channels,
                };

                let _ = tx.send(frame);
            }

            sequence += 1;
            tokio::time::sleep(Duration::from_micros(4000)).await; // ~250 Hz
        }
    }

    // Create broadcast channel for BCI data
    let (tx, _) = broadcast::channel::<BciFrame>(1024);

    let state = Arc::new(AppState { tx: tx.clone() });

    // Start simulated data generator
    // TODO: Replace with real device connection
    tokio::spawn(simulate_data(tx));

    // Build router
    let mut app = Router::new()
        .route("/ws", get(ws_handler))
        .route("/bci", get(ws_handler)) // Alias
        .layer(CorsLayer::new().allow_origin(Any).allow_methods(Any))
        .with_state(state);

    // Optionally serve static files
    if let Some(dir) = static_dir {
        use tower_http::services::ServeDir;
        info!("Serving static files from: {}", dir);
        app = app.fallback_service(ServeDir::new(dir));
    }

    // Health check endpoint
    let app = app.route(
        "/health",
        get(|| async { axum::Json(serde_json::json!({"status": "ok"})) }),
    );

    info!("Server listening on http://{}", addr);
    info!("WebSocket endpoint: ws://{}/ws", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

/// List available devices
fn list_devices(device_type: &str) -> anyhow::Result<()> {
    info!("Scanning for devices (type: {})...", device_type);

    #[cfg(feature = "usb")]
    if device_type == "usb" || device_type == "all" {
        info!("USB devices:");
        match serialport::available_ports() {
            Ok(ports) => {
                if ports.is_empty() {
                    info!("  (none found)");
                } else {
                    for port in ports {
                        info!("  {} - {:?}", port.port_name, port.port_type);
                    }
                }
            }
            Err(e) => {
                warn!("  Error scanning USB: {}", e);
            }
        }
    }

    #[cfg(not(feature = "usb"))]
    if device_type == "usb" || device_type == "all" {
        warn!("USB support not enabled. Rebuild with --features usb");
    }

    #[cfg(feature = "ble")]
    if device_type == "ble" || device_type == "all" {
        info!("BLE scanning not yet implemented");
        // TODO: Implement BLE device scanning
    }

    #[cfg(not(feature = "ble"))]
    if device_type == "ble" || device_type == "all" {
        warn!("BLE support not enabled. Rebuild with --features ble");
    }

    Ok(())
}
