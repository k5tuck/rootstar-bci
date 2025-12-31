//! Open Sound Control (OSC) streaming
//!
//! OSC is widely used in music and audio applications for real-time
//! control data. This module enables integration with:
//!
//! - Max/MSP, Pure Data, SuperCollider
//! - TouchDesigner, Resolume
//! - MIDI controllers and visual performance software
//! - VJing and live performance tools
//!
//! # Address Patterns
//!
//! The default address space follows this pattern:
//!
//! ```text
//! /bci/eeg/raw          - Raw EEG samples (8 floats)
//! /bci/eeg/alpha        - Alpha band power per channel
//! /bci/eeg/beta         - Beta band power per channel
//! /bci/eeg/theta        - Theta band power per channel
//! /bci/eeg/gamma        - Gamma band power per channel
//! /bci/fnirs/hbo        - HbO concentration (4 floats)
//! /bci/fnirs/hbr        - HbR concentration (4 floats)
//! /bci/emg/envelope     - EMG envelope value
//! /bci/emg/valence      - Facial expression valence
//! /bci/eda/scl          - Skin conductance level
//! /bci/eda/scr          - SCR (phasic) response
//! /bci/markers/event    - Event markers (string)
//! /bci/attention        - Attention metric (0-1)
//! /bci/relaxation       - Relaxation metric (0-1)
//! ```

use std::net::{SocketAddr, UdpSocket};
use thiserror::Error;
use rosc::{OscBundle, OscMessage as RoscMessage, OscPacket, OscType, encoder, decoder};

/// OSC errors
#[derive(Debug, Error)]
pub enum OscError {
    /// Network I/O error
    #[error("Network error: {0}")]
    Network(#[from] std::io::Error),

    /// OSC encoding error
    #[error("OSC encoding error: {0}")]
    Encoding(String),

    /// OSC decoding error
    #[error("OSC decoding error: {0}")]
    Decoding(String),

    /// Invalid address pattern
    #[error("Invalid OSC address: {0}")]
    InvalidAddress(String),
}

/// OSC result type
pub type OscResult<T> = Result<T, OscError>;

/// OSC message wrapper
#[derive(Debug, Clone)]
pub struct OscMessage {
    /// OSC address pattern
    pub address: String,
    /// Message arguments
    pub args: Vec<OscArg>,
}

/// OSC argument types
#[derive(Debug, Clone)]
pub enum OscArg {
    /// Float value
    Float(f32),
    /// Integer value
    Int(i32),
    /// String value
    String(String),
    /// Boolean value
    Bool(bool),
    /// Blob (binary data)
    Blob(Vec<u8>),
    /// Double precision float
    Double(f64),
    /// Long integer
    Long(i64),
}

impl From<f32> for OscArg {
    fn from(v: f32) -> Self {
        Self::Float(v)
    }
}

impl From<i32> for OscArg {
    fn from(v: i32) -> Self {
        Self::Int(v)
    }
}

impl From<&str> for OscArg {
    fn from(v: &str) -> Self {
        Self::String(v.to_string())
    }
}

impl From<String> for OscArg {
    fn from(v: String) -> Self {
        Self::String(v)
    }
}

impl From<bool> for OscArg {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}

impl OscArg {
    fn to_rosc_type(&self) -> OscType {
        match self {
            Self::Float(v) => OscType::Float(*v),
            Self::Int(v) => OscType::Int(*v),
            Self::String(v) => OscType::String(v.clone()),
            Self::Bool(v) => OscType::Bool(*v),
            Self::Blob(v) => OscType::Blob(v.clone()),
            Self::Double(v) => OscType::Double(*v),
            Self::Long(v) => OscType::Long(*v),
        }
    }
}

/// OSC sender for streaming BCI data
pub struct OscSender {
    socket: UdpSocket,
    target: SocketAddr,
    prefix: String,
}

impl OscSender {
    /// Create a new OSC sender
    ///
    /// # Arguments
    ///
    /// * `target` - Target address (e.g., "127.0.0.1:9000")
    pub fn new(target: &str) -> OscResult<Self> {
        let socket = UdpSocket::bind("0.0.0.0:0")?;
        let target: SocketAddr = target.parse().map_err(|_| {
            OscError::InvalidAddress(format!("Invalid target address: {}", target))
        })?;

        Ok(Self {
            socket,
            target,
            prefix: "/bci".to_string(),
        })
    }

    /// Set address prefix (default: "/bci")
    #[must_use]
    pub fn with_prefix(mut self, prefix: &str) -> Self {
        self.prefix = prefix.to_string();
        self
    }

    /// Send a raw OSC message
    pub fn send(&self, message: &OscMessage) -> OscResult<()> {
        let msg = RoscMessage {
            addr: message.address.clone(),
            args: message.args.iter().map(OscArg::to_rosc_type).collect(),
        };

        let packet = OscPacket::Message(msg);
        let bytes = encoder::encode(&packet)
            .map_err(|e| OscError::Encoding(format!("{:?}", e)))?;

        self.socket.send_to(&bytes, self.target)?;
        Ok(())
    }

    /// Send raw EEG samples
    pub fn send_eeg_raw(&self, channels: &[f32]) -> OscResult<()> {
        self.send(&OscMessage {
            address: format!("{}/eeg/raw", self.prefix),
            args: channels.iter().map(|&v| OscArg::Float(v)).collect(),
        })
    }

    /// Send EEG band powers
    pub fn send_band_powers(&self, band: &str, powers: &[f32]) -> OscResult<()> {
        self.send(&OscMessage {
            address: format!("{}/eeg/{}", self.prefix, band),
            args: powers.iter().map(|&v| OscArg::Float(v)).collect(),
        })
    }

    /// Send all EEG bands at once
    pub fn send_all_bands(&self, alpha: &[f32], beta: &[f32], theta: &[f32], gamma: &[f32]) -> OscResult<()> {
        self.send_band_powers("alpha", alpha)?;
        self.send_band_powers("beta", beta)?;
        self.send_band_powers("theta", theta)?;
        self.send_band_powers("gamma", gamma)?;
        Ok(())
    }

    /// Send fNIRS HbO/HbR concentrations
    pub fn send_fnirs(&self, hbo: &[f32], hbr: &[f32]) -> OscResult<()> {
        self.send(&OscMessage {
            address: format!("{}/fnirs/hbo", self.prefix),
            args: hbo.iter().map(|&v| OscArg::Float(v)).collect(),
        })?;
        self.send(&OscMessage {
            address: format!("{}/fnirs/hbr", self.prefix),
            args: hbr.iter().map(|&v| OscArg::Float(v)).collect(),
        })?;
        Ok(())
    }

    /// Send EMG data
    pub fn send_emg(&self, envelope: f32, valence: f32) -> OscResult<()> {
        self.send(&OscMessage {
            address: format!("{}/emg/envelope", self.prefix),
            args: vec![OscArg::Float(envelope)],
        })?;
        self.send(&OscMessage {
            address: format!("{}/emg/valence", self.prefix),
            args: vec![OscArg::Float(valence)],
        })?;
        Ok(())
    }

    /// Send EDA data
    pub fn send_eda(&self, scl: f32, scr: f32) -> OscResult<()> {
        self.send(&OscMessage {
            address: format!("{}/eda/scl", self.prefix),
            args: vec![OscArg::Float(scl)],
        })?;
        self.send(&OscMessage {
            address: format!("{}/eda/scr", self.prefix),
            args: vec![OscArg::Float(scr)],
        })?;
        Ok(())
    }

    /// Send an event marker
    pub fn send_marker(&self, event: &str) -> OscResult<()> {
        self.send(&OscMessage {
            address: format!("{}/markers/event", self.prefix),
            args: vec![OscArg::String(event.to_string())],
        })
    }

    /// Send attention metric (0-1)
    pub fn send_attention(&self, value: f32) -> OscResult<()> {
        self.send(&OscMessage {
            address: format!("{}/attention", self.prefix),
            args: vec![OscArg::Float(value.clamp(0.0, 1.0))],
        })
    }

    /// Send relaxation metric (0-1)
    pub fn send_relaxation(&self, value: f32) -> OscResult<()> {
        self.send(&OscMessage {
            address: format!("{}/relaxation", self.prefix),
            args: vec![OscArg::Float(value.clamp(0.0, 1.0))],
        })
    }

    /// Send a bundle of messages (for synchronized data)
    pub fn send_bundle(&self, messages: &[OscMessage]) -> OscResult<()> {
        use rosc::OscTime;

        let osc_messages: Vec<OscPacket> = messages
            .iter()
            .map(|m| {
                OscPacket::Message(RoscMessage {
                    addr: m.address.clone(),
                    args: m.args.iter().map(OscArg::to_rosc_type).collect(),
                })
            })
            .collect();

        let bundle = OscBundle {
            timetag: OscTime::try_from(std::time::UNIX_EPOCH)
                .unwrap_or(OscTime::try_from(std::time::UNIX_EPOCH).expect("epoch")),
            content: osc_messages,
        };

        let packet = OscPacket::Bundle(bundle);
        let bytes = encoder::encode(&packet)
            .map_err(|e| OscError::Encoding(format!("{:?}", e)))?;

        self.socket.send_to(&bytes, self.target)?;
        Ok(())
    }
}

/// OSC receiver for incoming control messages
pub struct OscReceiver {
    socket: UdpSocket,
    buffer: Vec<u8>,
}

impl OscReceiver {
    /// Create a new OSC receiver
    ///
    /// # Arguments
    ///
    /// * `bind_addr` - Local address to bind (e.g., "0.0.0.0:8000")
    pub fn new(bind_addr: &str) -> OscResult<Self> {
        let socket = UdpSocket::bind(bind_addr)?;
        socket.set_nonblocking(true)?;

        Ok(Self {
            socket,
            buffer: vec![0u8; 4096],
        })
    }

    /// Try to receive a message (non-blocking)
    pub fn try_recv(&mut self) -> OscResult<Option<OscMessage>> {
        match self.socket.recv_from(&mut self.buffer) {
            Ok((len, _addr)) => {
                let packet = decoder::decode_udp(&self.buffer[..len])
                    .map_err(|e| OscError::Decoding(format!("{:?}", e)))?;

                match packet.1 {
                    OscPacket::Message(msg) => Ok(Some(self.convert_message(&msg))),
                    OscPacket::Bundle(bundle) => {
                        // Return first message from bundle
                        for content in bundle.content {
                            if let OscPacket::Message(msg) = content {
                                return Ok(Some(self.convert_message(&msg)));
                            }
                        }
                        Ok(None)
                    }
                }
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Receive a message (blocking)
    pub fn recv(&mut self) -> OscResult<OscMessage> {
        self.socket.set_nonblocking(false)?;
        let (len, _addr) = self.socket.recv_from(&mut self.buffer)?;
        self.socket.set_nonblocking(true)?;

        let packet = decoder::decode_udp(&self.buffer[..len])
            .map_err(|e| OscError::Decoding(format!("{:?}", e)))?;

        match packet.1 {
            OscPacket::Message(msg) => Ok(self.convert_message(&msg)),
            OscPacket::Bundle(bundle) => {
                for content in bundle.content {
                    if let OscPacket::Message(msg) = content {
                        return Ok(self.convert_message(&msg));
                    }
                }
                Err(OscError::Decoding("Empty bundle".to_string()))
            }
        }
    }

    fn convert_message(&self, msg: &RoscMessage) -> OscMessage {
        let args = msg
            .args
            .iter()
            .map(|arg| match arg {
                OscType::Float(v) => OscArg::Float(*v),
                OscType::Int(v) => OscArg::Int(*v),
                OscType::String(v) => OscArg::String(v.clone()),
                OscType::Bool(v) => OscArg::Bool(*v),
                OscType::Blob(v) => OscArg::Blob(v.clone()),
                OscType::Double(v) => OscArg::Double(*v),
                OscType::Long(v) => OscArg::Long(*v),
                _ => OscArg::Int(0), // Fallback for unsupported types
            })
            .collect();

        OscMessage {
            address: msg.addr.clone(),
            args,
        }
    }
}

/// Convenience struct for bidirectional OSC communication
pub struct OscBridge {
    /// Sender for outgoing BCI data
    pub sender: OscSender,
    /// Receiver for incoming control messages
    pub receiver: OscReceiver,
}

impl OscBridge {
    /// Create a new OSC bridge
    ///
    /// # Arguments
    ///
    /// * `send_to` - Target address for outgoing data
    /// * `listen_on` - Local port for incoming messages
    pub fn new(send_to: &str, listen_on: &str) -> OscResult<Self> {
        Ok(Self {
            sender: OscSender::new(send_to)?,
            receiver: OscReceiver::new(listen_on)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_osc_arg_conversion() {
        let f = OscArg::from(3.14f32);
        assert!(matches!(f, OscArg::Float(_)));

        let i = OscArg::from(42i32);
        assert!(matches!(i, OscArg::Int(_)));

        let s = OscArg::from("hello");
        assert!(matches!(s, OscArg::String(_)));
    }

    #[test]
    fn test_osc_message() {
        let msg = OscMessage {
            address: "/bci/eeg/alpha".to_string(),
            args: vec![
                OscArg::Float(0.5),
                OscArg::Float(0.6),
                OscArg::Float(0.4),
            ],
        };

        assert_eq!(msg.address, "/bci/eeg/alpha");
        assert_eq!(msg.args.len(), 3);
    }
}
