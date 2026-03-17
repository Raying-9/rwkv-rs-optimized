#[cfg(feature = "trace")]
use tracing_subscriber::layer::SubscriberExt;
#[cfg(feature = "trace")]
use tracing_subscriber::util::SubscriberInitExt;
#[cfg(feature = "trace")]
use tracing_subscriber::{EnvFilter, fmt};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TraceMode {
    Off,
    Console,
    Tracy,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TraceLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl TraceLevel {
    #[cfg(feature = "trace")]
    fn as_directive(self) -> &'static str {
        match self {
            Self::Error => "error",
            Self::Warn => "warn",
            Self::Info => "info",
            Self::Debug => "debug",
            Self::Trace => "trace",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TraceConfig {
    pub mode: TraceMode,
    pub level: TraceLevel,
    pub with_ansi: bool,
}

impl Default for TraceConfig {
    fn default() -> Self {
        Self {
            mode: TraceMode::Tracy,
            level: TraceLevel::Trace,
            with_ansi: true,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TraceInitError {
    #[error("tracing feature is not enabled")]
    TraceFeatureDisabled,
    #[error("tracy mode requested, but tracing subscriber is already initialized")]
    TracySubscriberAlreadySet,
    #[error("failed to initialize tracing subscriber: {0}")]
    SubscriberInit(String),
}

pub fn init_tracing(service: &'static str, cfg: TraceConfig) -> Result<TraceMode, TraceInitError> {
    if cfg.mode == TraceMode::Off {
        return Ok(cfg.mode);
    }

    #[cfg(not(feature = "trace"))]
    {
        let _ = service;
        let _ = cfg;
        return Err(TraceInitError::TraceFeatureDisabled);
    }

    #[cfg(feature = "trace")]
    {
        let filter = EnvFilter::new(cfg.level.as_directive());
        let fmt_layer = fmt::layer()
            .with_target(true)
            .with_thread_ids(true)
            .with_thread_names(true)
            .with_ansi(cfg.with_ansi && cfg.mode == TraceMode::Console);

        let init_result = match cfg.mode {
            TraceMode::Off => Ok(()),
            TraceMode::Console => tracing_subscriber::registry()
                .with(filter)
                .with(fmt_layer)
                .try_init(),
            TraceMode::Tracy => tracing_subscriber::registry()
                .with(filter)
                .with(fmt_layer)
                .with(tracing_tracy::TracyLayer::default())
                .try_init(),
        };

        if let Err(err) = init_result {
            let msg = err.to_string();
            if msg.contains("global default trace dispatcher has already been set") {
                if cfg.mode == TraceMode::Tracy {
                    return Err(TraceInitError::TracySubscriberAlreadySet);
                }
                tracing::warn!(
                    service,
                    mode = ?cfg.mode,
                    "tracing subscriber already initialized, reusing existing subscriber"
                );
                return Ok(cfg.mode);
            }
            return Err(TraceInitError::SubscriberInit(msg));
        }

        tracing::info!(service, mode = ?cfg.mode, "tracing initialized");
        Ok(cfg.mode)
    }
}
