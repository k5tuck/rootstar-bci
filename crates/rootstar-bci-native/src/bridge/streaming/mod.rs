//! External application streaming protocols
//!
//! This module provides integration with external BCI applications and
//! neuroscience tools, similar to what OpenBCI offers:
//!
//! - [`lsl`]: Lab Streaming Layer (LSL) compatible streaming
//! - [`osc`]: Open Sound Control (OSC) for audio/music applications
//! - [`brainflow`]: BrainFlow-compatible data format
//!
//! # Examples
//!
//! ## Streaming to LSL-compatible applications
//!
//! ```rust,ignore
//! use rootstar_bci_native::bridge::streaming::{LslOutlet, StreamInfo};
//!
//! // Create an EEG stream
//! let info = StreamInfo::new("RootstarEEG", "EEG", 8, 250.0);
//! let outlet = LslOutlet::new(&info).await?;
//!
//! // Push samples as they arrive
//! outlet.push_sample(&eeg_data)?;
//! ```
//!
//! ## OSC streaming for audio applications
//!
//! ```rust,ignore
//! use rootstar_bci_native::bridge::streaming::OscSender;
//!
//! let sender = OscSender::new("127.0.0.1:9000")?;
//! sender.send_band_powers("/bci/alpha", &[0.5, 0.6, 0.4, 0.7])?;
//! ```

pub mod lsl;
pub mod brainflow;

#[cfg(feature = "osc")]
pub mod osc;

pub use lsl::{
    LslOutlet, LslInlet, StreamInfo, StreamType, ChannelFormat,
    LslError, LslResult,
};

pub use brainflow::{
    BrainFlowFormat, BrainFlowPacket, BoardId,
};

#[cfg(feature = "osc")]
pub use osc::{OscSender, OscReceiver, OscMessage, OscError, OscResult};
