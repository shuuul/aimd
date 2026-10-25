//! Hand-written server-sent-events framing and a channel-backed body.

use std::pin::Pin;
use std::task::{Context, Poll};

use axum::body::Bytes;
use http_body::{Body, Frame};
use tokio::sync::mpsc;

use crate::schemas::JobEvent;

/// `id: <sequence>\nevent: job\ndata: <json>\n\n`.
pub fn frame_job_event(event: &JobEvent) -> String {
    let data = serde_json::to_string(event).unwrap_or_else(|_| "{}".to_string());
    format!("id: {}\nevent: job\ndata: {data}\n\n", event.sequence)
}

/// Streaming body fed by a producer task; dropping the body (client
/// disconnect) closes the channel and stops the producer.
pub struct ChannelBody {
    rx: mpsc::Receiver<Bytes>,
}

impl ChannelBody {
    pub fn new(rx: mpsc::Receiver<Bytes>) -> Self {
        Self { rx }
    }
}

impl Body for ChannelBody {
    type Data = Bytes;
    type Error = std::convert::Infallible;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Self::Error>>> {
        self.rx
            .poll_recv(cx)
            .map(|chunk| chunk.map(|bytes| Ok(Frame::data(bytes))))
    }
}
