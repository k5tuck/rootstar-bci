//! Rootstar BCI Application
//!
//! Unified entry point for the Rootstar Brain-Computer Interface platform.
//! Supports native desktop visualization and WebSocket server modes.
//!
//! # Installation
//!
//! ## From Source (Recommended)
//!
//! ```bash
//! # Clone the repository
//! git clone https://github.com/k5tuck/rootstar-bci
//! cd rootstar-bci
//!
//! # Build and install the CLI globally
//! cargo install --path crates/rootstar-bci-app --features "native,server,usb,ble"
//!
//! # Or install with specific features only
//! cargo install --path crates/rootstar-bci-app --features server  # Server only
//! cargo install --path crates/rootstar-bci-app --features native  # Native viz only
//!
//! # The binary is installed to ~/.cargo/bin/rootstar
//! # Ensure ~/.cargo/bin is in your PATH
//! ```
//!
//! ## From crates.io (when published)
//!
//! ```bash
//! cargo install rootstar-bci-app --features "native,server,usb,ble"
//! ```
//!
//! ## Docker
//!
//! ```bash
//! # Build the Docker image
//! docker build -t rootstar-bci .
//!
//! # Run server mode
//! docker run -p 8080:8080 rootstar-bci
//!
//! # Or use docker compose
//! docker compose up
//! ```
//!
//! ## Running Without Installation
//!
//! ```bash
//! # Run directly with cargo
//! cargo run -p rootstar-bci-app --features native
//! cargo run -p rootstar-bci-app --features server -- server --port 8080
//! ```
//!
//! # Usage
//!
//! ```bash
//! # Native desktop visualization with simulated data (default)
//! rootstar
//! rootstar native --device simulate
//!
//! # Native viz with simulation parameters
//! rootstar native --sim-alpha 0.8 --sim-noise 0.05
//!
//! # Native viz with USB hardware
//! rootstar native --device usb --port /dev/ttyUSB0
//!
//! # Native viz with BLE hardware
//! rootstar native --device ble --ble-device "RootstarBCI"
//!
//! # WebSocket server mode (simulated)
//! rootstar server --port 8080
//!
//! # Server with USB device
//! rootstar server --device usb --usb-port /dev/ttyUSB0
//!
//! # Server with BLE device
//! rootstar server --device ble --ble-device "RootstarBCI"
//!
//! # List available devices
//! rootstar devices
//! rootstar devices --type usb
//! rootstar devices --type ble --scan-time 10
//! ```

use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, mpsc, RwLock};
use tracing::{debug, error, info, warn, Level};
use tracing_subscriber::FmtSubscriber;

// ============================================================================
// CLI Definition
// ============================================================================

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

#[derive(Subcommand, Debug, Clone)]
enum Commands {
    /// Run native desktop visualization (default if no subcommand)
    Native {
        /// Device connection type: usb, ble, or simulate
        #[arg(short, long, default_value = "simulate")]
        device: String,

        /// USB port path (e.g., /dev/ttyUSB0 or COM3)
        #[arg(long)]
        port: Option<String>,

        /// BLE device name or address to connect to
        #[arg(long)]
        ble_device: Option<String>,

        /// Window width
        #[arg(long, default_value = "1280")]
        width: u32,

        /// Window height
        #[arg(long, default_value = "720")]
        height: u32,

        /// Simulation: alpha band power (0.0-1.0)
        #[arg(long, default_value = "0.5")]
        sim_alpha: f32,

        /// Simulation: noise level (0.0-1.0)
        #[arg(long, default_value = "0.1")]
        sim_noise: f32,
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

        /// BLE device name or address
        #[arg(long)]
        ble_device: Option<String>,

        /// Serve static web files from this directory
        #[arg(long)]
        static_dir: Option<String>,

        /// Simulation sample rate in Hz
        #[arg(long, default_value = "250")]
        sim_rate: u32,
    },

    /// List available devices
    Devices {
        /// Device type to scan: usb, ble, or all
        #[arg(short = 't', long = "type", default_value = "all")]
        device_type: String,

        /// BLE scan duration in seconds
        #[arg(long, default_value = "5")]
        scan_time: u64,
    },
}

// ============================================================================
// Data Types
// ============================================================================

/// BCI data frame for streaming
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BciFrame {
    /// Timestamp in microseconds since start
    pub timestamp_us: u64,
    /// Frame type: "eeg", "fnirs", "emg", "eda"
    pub frame_type: String,
    /// Channel data
    pub channels: Vec<f32>,
    /// Optional metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<FrameMetadata>,
}

/// Optional frame metadata
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrameMetadata {
    /// Sequence number
    pub sequence: u32,
    /// Device ID
    pub device_id: Option<String>,
    /// Signal quality indicators (0.0-1.0 per channel)
    pub quality: Option<Vec<f32>>,
}

/// Control message from client
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ControlMessage {
    /// Start data acquisition
    #[serde(rename = "start")]
    Start,
    /// Stop data acquisition
    #[serde(rename = "stop")]
    Stop,
    /// Set stimulation parameters
    #[serde(rename = "stim")]
    Stimulation {
        amplitude_ua: f32,
        frequency_hz: f32,
        duration_ms: u32,
    },
    /// Request device status
    #[serde(rename = "status")]
    Status,
    /// Ping/keepalive
    #[serde(rename = "ping")]
    Ping,
}

/// Data source configuration
#[derive(Clone, Debug)]
pub enum DataSourceConfig {
    /// Simulated EEG/fNIRS data
    Simulate {
        sample_rate_hz: u32,
        alpha_power: f32,
        noise_level: f32,
    },
    /// USB serial connection
    Usb { port: String, baud_rate: u32 },
    /// Bluetooth Low Energy connection
    Ble {
        device_name: Option<String>,
        device_address: Option<String>,
    },
}

// ============================================================================
// Main Entry Point
// ============================================================================

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

    match cli.command.clone() {
        None => {
            // Default to native mode with simulation
            run_native(Commands::Native {
                device: "simulate".to_string(),
                port: None,
                ble_device: None,
                width: 1280,
                height: 720,
                sim_alpha: 0.5,
                sim_noise: 0.1,
            })?;
        }
        Some(cmd @ Commands::Native { .. }) => {
            run_native(cmd)?;
        }
        Some(Commands::Server {
            port,
            bind,
            device,
            usb_port,
            ble_device,
            static_dir,
            sim_rate,
        }) => {
            run_server(
                port, bind, device, usb_port, ble_device, static_dir, sim_rate,
            )?;
        }
        Some(Commands::Devices {
            device_type,
            scan_time,
        }) => {
            list_devices(&device_type, scan_time)?;
        }
    }

    Ok(())
}

// ============================================================================
// Native Visualization Mode
// ============================================================================

fn run_native(command: Commands) -> anyhow::Result<()> {
    let Commands::Native {
        device,
        port,
        ble_device,
        width,
        height,
        sim_alpha,
        sim_noise,
    } = command
    else {
        unreachable!()
    };

    info!("Starting native visualization");
    info!("Device mode: {}", device);

    // Build data source configuration
    let data_source = build_data_source_config(&device, port.clone(), ble_device.clone(), sim_alpha, sim_noise, 250);

    match &data_source {
        DataSourceConfig::Simulate { alpha_power, noise_level, .. } => {
            info!("Using simulated data (alpha={}, noise={})", alpha_power, noise_level);
        }
        DataSourceConfig::Usb { port, .. } => {
            info!("Using USB device at {}", port);
        }
        DataSourceConfig::Ble { device_name, .. } => {
            info!("Using BLE device: {:?}", device_name);
        }
    }

    #[cfg(feature = "native")]
    {
        use rootstar_bci_native::viz::{run_app, VizConfig};

        // For native viz, the app has built-in simulation
        // For hardware mode, we would need to start a background data thread
        // that feeds data to a shared state (future enhancement)
        if !matches!(data_source, DataSourceConfig::Simulate { .. }) {
            warn!("Native visualization currently uses built-in simulation");
            warn!("Hardware integration with native viz requires shared state (future work)");
            warn!("For real hardware, use 'rootstar server' mode instead");
        }

        let title = match &data_source {
            DataSourceConfig::Simulate { .. } => "Rootstar BCI - Simulation".to_string(),
            DataSourceConfig::Usb { port, .. } => format!("Rootstar BCI - USB: {}", port),
            DataSourceConfig::Ble { device_name, .. } => {
                format!("Rootstar BCI - BLE: {}", device_name.as_deref().unwrap_or("scanning..."))
            }
        };

        let config = VizConfig {
            width,
            height,
            title,
            ..Default::default()
        };

        run_app(config).map_err(|e| anyhow::anyhow!("{}", e))?;
    }

    #[cfg(not(feature = "native"))]
    {
        let _ = (width, height, data_source, port, ble_device, sim_alpha, sim_noise);
        anyhow::bail!(
            "Native visualization not enabled. Rebuild with --features native:\n\
             cargo install --path crates/rootstar-bci-app --features native"
        );
    }

    Ok(())
}

// ============================================================================
// WebSocket Server Mode
// ============================================================================

fn run_server(
    port: u16,
    bind: String,
    device: String,
    usb_port: Option<String>,
    ble_device: Option<String>,
    static_dir: Option<String>,
    sim_rate: u32,
) -> anyhow::Result<()> {
    #[cfg(feature = "server")]
    {
        use std::net::SocketAddr;
        use tokio::runtime::Runtime;

        info!("Starting WebSocket server on {}:{}", bind, port);

        let data_source = build_data_source_config(&device, usb_port, ble_device, 0.5, 0.1, sim_rate);

        match &data_source {
            DataSourceConfig::Simulate { sample_rate_hz, .. } => {
                info!("Data source: Simulated @ {} Hz", sample_rate_hz);
            }
            DataSourceConfig::Usb { port, baud_rate } => {
                info!("Data source: USB {} @ {} baud", port, baud_rate);
            }
            DataSourceConfig::Ble { device_name, .. } => {
                info!("Data source: BLE {:?}", device_name);
            }
        }

        let rt = Runtime::new()?;
        rt.block_on(async {
            let addr: SocketAddr = format!("{}:{}", bind, port).parse()?;
            run_server_async(addr, data_source, static_dir).await
        })?;
    }

    #[cfg(not(feature = "server"))]
    {
        let _ = (port, bind, device, usb_port, ble_device, static_dir, sim_rate);
        anyhow::bail!(
            "WebSocket server not enabled. Rebuild with --features server:\n\
             cargo install --path crates/rootstar-bci-app --features server"
        );
    }

    Ok(())
}

/// Build data source configuration from CLI args
fn build_data_source_config(
    device: &str,
    usb_port: Option<String>,
    ble_device: Option<String>,
    alpha_power: f32,
    noise_level: f32,
    sample_rate_hz: u32,
) -> DataSourceConfig {
    match device {
        "usb" => {
            let port = usb_port.unwrap_or_else(|| {
                if let Some(p) = find_first_usb_port() {
                    info!("Auto-detected USB port: {}", p);
                    p
                } else {
                    warn!("No USB port specified or detected");
                    "/dev/ttyUSB0".to_string()
                }
            });
            DataSourceConfig::Usb {
                port,
                baud_rate: 921600,
            }
        }
        "ble" => DataSourceConfig::Ble {
            device_name: ble_device,
            device_address: None,
        },
        _ => DataSourceConfig::Simulate {
            sample_rate_hz,
            alpha_power,
            noise_level,
        },
    }
}

/// Async server implementation
#[cfg(feature = "server")]
async fn run_server_async(
    addr: std::net::SocketAddr,
    data_source: DataSourceConfig,
    static_dir: Option<String>,
) -> anyhow::Result<()> {
    use axum::{
        extract::ws::{Message, WebSocket, WebSocketUpgrade},
        extract::State,
        response::IntoResponse,
        routing::get,
        Router,
    };
    use tower_http::cors::{Any, CorsLayer};

    /// Shared application state
    struct AppState {
        tx: broadcast::Sender<BciFrame>,
        control_tx: mpsc::Sender<ControlMessage>,
        acquiring: Arc<RwLock<bool>>,
    }

    async fn ws_handler(
        ws: WebSocketUpgrade,
        State(state): State<Arc<AppState>>,
    ) -> impl IntoResponse {
        ws.on_upgrade(move |socket| handle_socket(socket, state))
    }

    async fn handle_socket(mut socket: WebSocket, state: Arc<AppState>) {
        let mut rx = state.tx.subscribe();

        info!("WebSocket client connected");

        let welcome = serde_json::json!({
            "type": "welcome",
            "version": env!("CARGO_PKG_VERSION"),
            "capabilities": ["eeg", "fnirs", "emg", "eda", "sns"],
            "acquiring": *state.acquiring.read().await
        });

        if socket
            .send(Message::Text(welcome.to_string().into()))
            .await
            .is_err()
        {
            return;
        }

        loop {
            tokio::select! {
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
                        Err(broadcast::error::RecvError::Closed) => break,
                    }
                }

                result = socket.recv() => {
                    match result {
                        Some(Ok(Message::Text(text))) => {
                            debug!("Control message: {}", text);
                            if let Ok(msg) = serde_json::from_str::<ControlMessage>(&text) {
                                handle_control_message(&state, &mut socket, msg).await;
                            }
                        }
                        Some(Ok(Message::Close(_))) | None => break,
                        _ => {}
                    }
                }
            }
        }

        info!("WebSocket client disconnected");
    }

    async fn handle_control_message(
        state: &Arc<AppState>,
        socket: &mut WebSocket,
        msg: ControlMessage,
    ) {
        let response = match msg {
            ControlMessage::Start => {
                info!("Starting acquisition");
                *state.acquiring.write().await = true;
                let _ = state.control_tx.send(ControlMessage::Start).await;
                serde_json::json!({"type": "ack", "command": "start"})
            }
            ControlMessage::Stop => {
                info!("Stopping acquisition");
                *state.acquiring.write().await = false;
                let _ = state.control_tx.send(ControlMessage::Stop).await;
                serde_json::json!({"type": "ack", "command": "stop"})
            }
            ControlMessage::Stimulation {
                amplitude_ua,
                frequency_hz,
                duration_ms,
            } => {
                info!("Stim: {}uA @ {}Hz for {}ms", amplitude_ua, frequency_hz, duration_ms);
                if amplitude_ua > 2000.0 {
                    serde_json::json!({"type": "error", "message": "Amplitude exceeds 2000uA limit"})
                } else {
                    let _ = state.control_tx.send(ControlMessage::Stimulation {
                        amplitude_ua,
                        frequency_hz,
                        duration_ms,
                    }).await;
                    serde_json::json!({"type": "ack", "command": "stim"})
                }
            }
            ControlMessage::Status => {
                serde_json::json!({
                    "type": "status",
                    "acquiring": *state.acquiring.read().await,
                    "version": env!("CARGO_PKG_VERSION")
                })
            }
            ControlMessage::Ping => serde_json::json!({"type": "pong"}),
        };
        let _ = socket.send(Message::Text(response.to_string().into())).await;
    }

    let (tx, _) = broadcast::channel::<BciFrame>(1024);
    let (control_tx, control_rx) = mpsc::channel::<ControlMessage>(64);

    let state = Arc::new(AppState {
        tx: tx.clone(),
        control_tx,
        acquiring: Arc::new(RwLock::new(true)),
    });

    // Start appropriate data source
    match data_source {
        DataSourceConfig::Simulate {
            sample_rate_hz,
            alpha_power,
            noise_level,
        } => {
            tokio::spawn(run_simulation(tx, sample_rate_hz, alpha_power, noise_level));
        }
        DataSourceConfig::Usb { port, baud_rate } => {
            tokio::spawn(run_usb_source(tx, control_rx, port, baud_rate));
        }
        DataSourceConfig::Ble {
            device_name,
            device_address,
        } => {
            tokio::spawn(run_ble_source(tx, control_rx, device_name, device_address));
        }
    }

    let mut app = Router::new()
        .route("/ws", get(ws_handler))
        .route("/bci", get(ws_handler))
        .layer(CorsLayer::new().allow_origin(Any).allow_methods(Any))
        .with_state(state);

    if let Some(dir) = static_dir {
        use tower_http::services::ServeDir;
        info!("Serving static files from: {}", dir);
        app = app.fallback_service(ServeDir::new(dir));
    }

    let app = app.route(
        "/health",
        get(|| async { axum::Json(serde_json::json!({"status": "ok"})) }),
    );

    info!("Listening on http://{}", addr);
    info!("WebSocket: ws://{}/ws", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

// ============================================================================
// Data Sources
// ============================================================================

/// Simulated EEG/fNIRS data generator
#[cfg(feature = "server")]
async fn run_simulation(
    tx: broadcast::Sender<BciFrame>,
    sample_rate_hz: u32,
    alpha_power: f32,
    noise_level: f32,
) {
    let start = Instant::now();
    let mut sequence = 0u32;
    let interval = Duration::from_micros(1_000_000 / u64::from(sample_rate_hz));

    loop {
        let elapsed = start.elapsed();
        let timestamp_us = elapsed.as_micros() as u64;
        let t = elapsed.as_secs_f32();

        // Generate EEG with multiple frequency bands
        let eeg: Vec<f32> = (0..8)
            .map(|ch| {
                let phase = ch as f32 * 0.5;
                let delta = 5.0 * (2.0 * std::f32::consts::PI * 2.0 * t + phase).sin();
                let theta = 3.0 * (2.0 * std::f32::consts::PI * 6.0 * t + phase * 0.7).sin();
                let alpha = 10.0 * alpha_power * (2.0 * std::f32::consts::PI * 10.0 * t + phase).sin();
                let beta = 5.0 * (2.0 * std::f32::consts::PI * 20.0 * t + phase * 1.3).sin();
                let noise = noise_level * 10.0 * ((sequence as f32 * 0.1 + ch as f32 * 17.3).sin());
                delta + theta + alpha + beta + noise
            })
            .collect();

        let _ = tx.send(BciFrame {
            timestamp_us,
            frame_type: "eeg".to_string(),
            channels: eeg,
            metadata: Some(FrameMetadata {
                sequence,
                device_id: Some("sim-0".to_string()),
                quality: Some(vec![0.95; 8]),
            }),
        });

        // fNIRS at 10 Hz
        if sequence % (sample_rate_hz / 10) == 0 {
            let fnirs: Vec<f32> = (0..8)
                .map(|ch| {
                    let phase = ch as f32 * 0.3;
                    if ch % 2 == 0 {
                        0.5 + 0.3 * (2.0 * std::f32::consts::PI * 0.1 * t + phase).sin()
                    } else {
                        0.3 - 0.2 * (2.0 * std::f32::consts::PI * 0.1 * t + phase).sin()
                    }
                })
                .collect();

            let _ = tx.send(BciFrame {
                timestamp_us,
                frame_type: "fnirs".to_string(),
                channels: fnirs,
                metadata: Some(FrameMetadata {
                    sequence,
                    device_id: Some("sim-0".to_string()),
                    quality: None,
                }),
            });
        }

        sequence = sequence.wrapping_add(1);
        tokio::time::sleep(interval).await;
    }
}

/// USB serial data source
#[cfg(feature = "server")]
async fn run_usb_source(
    tx: broadcast::Sender<BciFrame>,
    mut control_rx: mpsc::Receiver<ControlMessage>,
    port: String,
    baud_rate: u32,
) {
    #[cfg(feature = "usb")]
    {
        use rootstar_bci_native::bridge::UsbBridge;
        use rootstar_bci_core::protocol::PacketType;

        let mut bridge = match UsbBridge::open(&port, baud_rate) {
            Ok(b) => {
                info!("USB connected to {}", port);
                b
            }
            Err(e) => {
                error!("Failed to open USB {}: {}", port, e);
                return;
            }
        };

        if let Err(e) = bridge.start_acquisition() {
            error!("Failed to start acquisition: {}", e);
            return;
        }

        let start = Instant::now();
        let mut sequence = 0u32;

        loop {
            // Handle control messages
            if let Ok(msg) = control_rx.try_recv() {
                match msg {
                    ControlMessage::Stop => {
                        info!("Stopping USB acquisition");
                        let _ = bridge.stop_acquisition();
                    }
                    ControlMessage::Start => {
                        info!("Starting USB acquisition");
                        let _ = bridge.start_acquisition();
                    }
                    _ => {}
                }
            }

            match bridge.read_packet() {
                Ok(Some((packet_type, payload))) => {
                    let frame_type = match packet_type {
                        PacketType::EegData => "eeg",
                        PacketType::FnirsData => "fnirs",
                        PacketType::EmgData => "emg",
                        PacketType::EdaData => "eda",
                        _ => continue,
                    };

                    let channels = parse_packet_payload(&payload, packet_type);

                    let _ = tx.send(BciFrame {
                        timestamp_us: start.elapsed().as_micros() as u64,
                        frame_type: frame_type.to_string(),
                        channels,
                        metadata: Some(FrameMetadata {
                            sequence,
                            device_id: Some(format!("usb:{}", port)),
                            quality: None,
                        }),
                    });

                    sequence = sequence.wrapping_add(1);
                }
                Ok(None) => {
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
                Err(e) => {
                    error!("USB error: {}", e);
                    break;
                }
            }
        }
    }

    #[cfg(not(feature = "usb"))]
    {
        let _ = (tx, control_rx, port, baud_rate);
        error!("USB feature not compiled");
    }
}

/// Parse raw packet into float channels
#[cfg(all(feature = "server", feature = "usb"))]
fn parse_packet_payload(payload: &[u8], packet_type: rootstar_bci_core::protocol::PacketType) -> Vec<f32> {
    use rootstar_bci_core::protocol::PacketType;

    match packet_type {
        PacketType::EegData => {
            // 8ch × 3 bytes (24-bit signed)
            (0..8)
                .filter_map(|i| {
                    let off = i * 3;
                    if off + 3 <= payload.len() {
                        let sign = if payload[off] & 0x80 != 0 { 0xFF } else { 0x00 };
                        let raw = i32::from_be_bytes([sign, payload[off], payload[off + 1], payload[off + 2]]);
                        Some(raw as f32 * 0.022351) // µV/LSB
                    } else {
                        None
                    }
                })
                .collect()
        }
        PacketType::FnirsData => {
            payload
                .chunks_exact(4)
                .take(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        }
        _ => {
            payload
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        }
    }
}

/// BLE data source
#[cfg(feature = "server")]
async fn run_ble_source(
    tx: broadcast::Sender<BciFrame>,
    mut control_rx: mpsc::Receiver<ControlMessage>,
    device_name: Option<String>,
    _device_address: Option<String>,
) {
    #[cfg(feature = "ble")]
    {
        use rootstar_bci_native::bridge::{BleBridge, BleEvent};
        use rootstar_bci_core::protocol::PacketType;

        let (mut bridge, mut rx) = match BleBridge::new().await {
            Ok(b) => b,
            Err(e) => {
                error!("BLE init failed: {}", e);
                return;
            }
        };

        info!("BLE initialized, scanning...");

        if let Err(e) = bridge.start_scan().await {
            error!("BLE scan failed: {}", e);
            return;
        }

        let start = Instant::now();
        let mut sequence = 0u32;
        let target = device_name.as_deref();

        loop {
            tokio::select! {
                Some(msg) = control_rx.recv() => {
                    match msg {
                        ControlMessage::Stop => info!("BLE: stop requested"),
                        ControlMessage::Start => info!("BLE: start requested"),
                        _ => {}
                    }
                }

                Some(event) = rx.recv() => {
                    match event {
                        BleEvent::DeviceDiscovered(dev) => {
                            let matches = target.map(|n| dev.name.as_deref() == Some(n))
                                .unwrap_or(dev.has_bci_service);
                            if matches {
                                info!("Found device: {:?}", dev);
                                let _ = bridge.connect(&dev.address).await;
                            }
                        }
                        BleEvent::Connected { device_id } => {
                            info!("Connected: {:?}", device_id);
                            let _ = bridge.start_notifications().await;
                        }
                        BleEvent::Data { packet_type, payload, device_id } => {
                            let frame_type = match packet_type {
                                PacketType::EegData => "eeg",
                                PacketType::FnirsData => "fnirs",
                                PacketType::EmgData => "emg",
                                PacketType::EdaData => "eda",
                                _ => continue,
                            };

                            let channels: Vec<f32> = payload
                                .chunks_exact(4)
                                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                                .collect();

                            let _ = tx.send(BciFrame {
                                timestamp_us: start.elapsed().as_micros() as u64,
                                frame_type: frame_type.to_string(),
                                channels,
                                metadata: Some(FrameMetadata {
                                    sequence,
                                    device_id: Some(format!("{:?}", device_id)),
                                    quality: None,
                                }),
                            });

                            sequence = sequence.wrapping_add(1);
                        }
                        BleEvent::Disconnected { device_id, reason } => {
                            warn!("Disconnected {:?}: {:?}", device_id, reason);
                            let _ = bridge.start_scan().await;
                        }
                        BleEvent::Error { message, .. } => {
                            error!("BLE error: {}", message);
                        }
                    }
                }
            }
        }
    }

    #[cfg(not(feature = "ble"))]
    {
        let _ = (tx, control_rx, device_name, _device_address);
        error!("BLE feature not compiled");
    }
}

// ============================================================================
// Device Discovery
// ============================================================================

fn list_devices(device_type: &str, scan_time: u64) -> anyhow::Result<()> {
    info!("Scanning for devices...");

    if device_type == "usb" || device_type == "all" {
        list_usb_devices();
    }

    if device_type == "ble" || device_type == "all" {
        list_ble_devices(scan_time)?;
    }

    Ok(())
}

fn list_usb_devices() {
    #[cfg(feature = "usb")]
    {
        use serialport::SerialPortType;

        info!("USB devices:");
        match serialport::available_ports() {
            Ok(ports) if ports.is_empty() => info!("  (none)"),
            Ok(ports) => {
                for p in ports {
                    let desc = match &p.port_type {
                        SerialPortType::UsbPort(u) => format!(
                            "VID:{:04X} PID:{:04X} {}",
                            u.vid,
                            u.pid,
                            u.product.as_deref().unwrap_or("")
                        ),
                        SerialPortType::BluetoothPort => "Bluetooth".into(),
                        SerialPortType::PciPort => "PCI".into(),
                        SerialPortType::Unknown => "Unknown".into(),
                    };
                    info!("  {} - {}", p.port_name, desc);
                }
            }
            Err(e) => warn!("  Error: {}", e),
        }
    }

    #[cfg(not(feature = "usb"))]
    warn!("USB not enabled (--features usb)");
}

fn list_ble_devices(scan_time: u64) -> anyhow::Result<()> {
    #[cfg(feature = "ble")]
    {
        use tokio::runtime::Runtime;

        let rt = Runtime::new()?;
        rt.block_on(async {
            use rootstar_bci_native::bridge::{BleBridge, BleEvent};

            let (mut bridge, mut rx) = match BleBridge::new().await {
                Ok(b) => b,
                Err(e) => {
                    warn!("BLE init failed: {}", e);
                    return;
                }
            };

            info!("BLE devices ({}s scan):", scan_time);

            bridge.set_scan_duration(Duration::from_secs(scan_time));
            if let Err(e) = bridge.start_scan().await {
                warn!("  Scan error: {}", e);
                return;
            }

            let mut found = Vec::new();
            let deadline = Instant::now() + Duration::from_secs(scan_time);

            while Instant::now() < deadline {
                match tokio::time::timeout(Duration::from_millis(100), rx.recv()).await {
                    Ok(Some(BleEvent::DeviceDiscovered(d))) => {
                        if !found.contains(&d.address) {
                            let bci = if d.has_bci_service { " [BCI]" } else { "" };
                            info!("  {} - {} ({}dBm){}", d.address, d.name.as_deref().unwrap_or("?"), d.rssi, bci);
                            found.push(d.address);
                        }
                    }
                    _ => {}
                }
            }

            if found.is_empty() {
                info!("  (none)");
            }
        });
    }

    #[cfg(not(feature = "ble"))]
    {
        let _ = scan_time;
        warn!("BLE not enabled (--features ble)");
    }

    Ok(())
}

/// Auto-detect first USB port that looks like a BCI device
fn find_first_usb_port() -> Option<String> {
    #[cfg(feature = "usb")]
    {
        use serialport::SerialPortType;

        serialport::available_ports().ok()?.into_iter().find(|p| {
            matches!(&p.port_type, SerialPortType::UsbPort(u) if
                u.vid == 0x10C4 ||  // CP210x
                u.vid == 0x1A86 ||  // CH340
                u.vid == 0x303A     // Espressif
            )
        }).map(|p| p.port_name)
    }

    #[cfg(not(feature = "usb"))]
    None
}
