use tracing_subscriber::{fmt, EnvFilter};
use std::io;

/// Initializes the global tracing subscriber.
/// All logs, diagnostics, and progress outputs will be routed strictly to STDERR.
pub fn init_logger() {
    fmt()
        .with_writer(io::stderr) // Enforce Unix Philosophy: Logs go to STDERR
        .with_env_filter(
            EnvFilter::builder()
                .with_default_directive(tracing::Level::INFO.into())
                .from_env_lossy(),
        )
        .init();
}
