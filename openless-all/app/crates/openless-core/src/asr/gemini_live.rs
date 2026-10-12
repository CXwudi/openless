//! Gemini Live transcription over the bidirectional Live API WebSocket.
//!
//! This provider is deliberately separate from [`super::gemini`].  The batch
//! Gemini transcriber uploads a WAV file and polls an Interaction; Live accepts
//! raw 16 kHz PCM frames and emits interim and final transcript events.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use base64::Engine;
use futures_util::{SinkExt, StreamExt};
use parking_lot::Mutex as ParkingMutex;
use serde_json::{json, Value};
use tokio::net::TcpStream;
use tokio::sync::{mpsc, oneshot, Mutex as AsyncMutex, Notify};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};

use crate::config::{TaskSpawner, TokioTaskSpawner};
use crate::ports::{TextStreamChunk, TextStreamSink};

use super::qwen_realtime::join_segments;
use super::{AudioConsumer, RawTranscript};

pub const PROVIDER_ID: &str = "gemini-live";
pub const DEFAULT_ENDPOINT: &str =
    "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent";
pub const DEFAULT_MODEL: &str = "gemini-3.5-transcribe-live";
pub const TARGET_AUDIO_CHUNK_BYTES: usize = 3_200;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const SESSION_READY_TIMEOUT: Duration = Duration::from_secs(5);
const WRITE_TIMEOUT: Duration = Duration::from_secs(5);
const FINAL_RESULT_TIMEOUT: Duration = Duration::from_secs(12);
const FINISH_HARD_DEADLINE: Duration = Duration::from_secs(12);
const BYTES_PER_MS: u64 = 32;

pub type WsStream = WebSocketStream<MaybeTlsStream<TcpStream>>;
type WsSink = futures_util::stream::SplitSink<WsStream, Message>;
type SharedWriter = Arc<AsyncMutex<Option<WsSink>>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeminiLiveMode {
    Verbatim,
    Smart,
}

impl GeminiLiveMode {
    fn as_wire(self) -> &'static str {
        match self {
            Self::Verbatim => "VERBATIM",
            Self::Smart => "SMART",
        }
    }
}

#[derive(Debug, Clone)]
pub struct GeminiLiveCredentials {
    pub api_key: String,
    pub endpoint: String,
    pub model: String,
    pub language_codes: Vec<String>,
    pub custom_vocabulary: Vec<String>,
    pub mode: GeminiLiveMode,
}

impl GeminiLiveCredentials {
    pub fn normalized_model(&self) -> String {
        let model = self.model.trim();
        let model = model.strip_prefix("models/").unwrap_or(model);
        if model.is_empty() {
            DEFAULT_MODEL.to_string()
        } else {
            format!("models/{model}")
        }
    }

    pub fn connect_url(&self) -> Result<String, GeminiLiveASRError> {
        let endpoint = self.endpoint.trim();
        let mut url = url::Url::parse(if endpoint.is_empty() {
            DEFAULT_ENDPOINT
        } else {
            endpoint
        })
        .map_err(|_| GeminiLiveASRError::ConnectionFailed("invalid WebSocket endpoint".into()))?;
        if !matches!(url.scheme(), "ws" | "wss") {
            return Err(GeminiLiveASRError::ConnectionFailed(
                "Gemini Live endpoint must use ws or wss".into(),
            ));
        }
        let existing = url
            .query_pairs()
            .filter(|(key, _)| key != "key" && key != "access_token")
            .map(|(key, value)| (key.into_owned(), value.into_owned()))
            .collect::<Vec<_>>();
        {
            let mut query = url.query_pairs_mut();
            query.clear();
            for (key, value) in existing {
                query.append_pair(&key, &value);
            }
            query.append_pair("key", self.api_key.trim());
        }
        Ok(url.to_string())
    }

    fn setup_message(&self) -> String {
        build_setup_message(
            &self.normalized_model(),
            &self.language_codes,
            &self.custom_vocabulary,
            self.mode,
        )
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GeminiLiveASRError {
    #[error("credentials missing")]
    CredentialsMissing,
    #[error("connection failed: {0}")]
    ConnectionFailed(String),
    #[error("send failed: {0}")]
    SendFailed(String),
    #[error("task failed: {0}")]
    TaskFailed(String),
    #[error("no final result")]
    NoFinalResult,
    #[error("final result timed out")]
    FinalResultTimeout,
}

enum SendItem {
    Audio(Vec<u8>),
    End(oneshot::Sender<Result<(), String>>),
}

#[derive(Default)]
struct SyncState {
    pending_audio: Vec<u8>,
    audio_scratch: Vec<u8>,
    bytes_received: u64,
    session_started: bool,
    session_start_error: Option<String>,
    session_finished: bool,
    ending: bool,
    audio_stream_end_sent: bool,
    audio_stream_end_written: bool,
    final_received: bool,
    turn_complete_received: bool,
    start: Option<Instant>,
    final_tx: Option<oneshot::Sender<Result<RawTranscript, GeminiLiveASRError>>>,
    send_tx: Option<mpsc::UnboundedSender<SendItem>>,
    completed_segments: Vec<String>,
    partial_text: String,
    last_final_text: Option<String>,
}

pub struct GeminiLiveASR {
    credentials: GeminiLiveCredentials,
    task_spawner: Arc<dyn TaskSpawner>,
    state: ParkingMutex<SyncState>,
    writer: SharedWriter,
    final_rx: ParkingMutex<Option<oneshot::Receiver<Result<RawTranscript, GeminiLiveASRError>>>>,
    session_started: Arc<Notify>,
    session_finished: Arc<Notify>,
    shutdown: Arc<Notify>,
    shutdown_requested: Arc<AtomicBool>,
    partial_sink: ParkingMutex<Option<Arc<dyn TextStreamSink>>>,
}

impl GeminiLiveASR {
    pub fn new(credentials: GeminiLiveCredentials) -> Self {
        Self::with_task_spawner(credentials, Arc::new(TokioTaskSpawner))
    }

    pub fn with_task_spawner(
        credentials: GeminiLiveCredentials,
        task_spawner: Arc<dyn TaskSpawner>,
    ) -> Self {
        Self {
            credentials,
            task_spawner,
            state: ParkingMutex::new(SyncState::default()),
            writer: Arc::new(AsyncMutex::new(None)),
            final_rx: ParkingMutex::new(None),
            session_started: Arc::new(Notify::new()),
            session_finished: Arc::new(Notify::new()),
            shutdown: Arc::new(Notify::new()),
            shutdown_requested: Arc::new(AtomicBool::new(false)),
            partial_sink: ParkingMutex::new(None),
        }
    }

    pub fn set_partial_sink(&self, sink: Arc<dyn TextStreamSink>) {
        *self.partial_sink.lock() = Some(sink);
    }

    pub async fn open_session(self: &Arc<Self>) -> Result<(), GeminiLiveASRError> {
        if self.credentials.api_key.trim().is_empty() {
            return Err(GeminiLiveASRError::CredentialsMissing);
        }
        let url = self.credentials.connect_url()?;
        let (ws, _) = tokio::time::timeout(CONNECT_TIMEOUT, connect_async(url))
            .await
            .map_err(|_| {
                GeminiLiveASRError::ConnectionFailed("WebSocket connection timed out".into())
            })?
            .map_err(|_| {
                GeminiLiveASRError::ConnectionFailed("WebSocket connection failed".into())
            })?;
        let (write, read) = ws.split();
        *self.writer.lock().await = Some(write);

        let (final_tx, final_rx) = oneshot::channel();
        let (send_tx, mut send_rx) = mpsc::unbounded_channel::<SendItem>();
        {
            let mut state = self.state.lock();
            *state = SyncState {
                final_tx: Some(final_tx),
                send_tx: Some(send_tx),
                start: Some(Instant::now()),
                ..SyncState::default()
            };
        }
        *self.final_rx.lock() = Some(final_rx);

        let writer_for_worker = Arc::clone(&self.writer);
        let weak_self = Arc::downgrade(self);
        let shutdown = Arc::clone(&self.shutdown);
        let shutdown_requested = Arc::clone(&self.shutdown_requested);
        let spawner = Arc::clone(&self.task_spawner);
        spawner.spawn(Box::pin(async move {
            loop {
                if shutdown_requested.load(Ordering::Acquire) {
                    break;
                }
                let item = tokio::select! {
                    _ = shutdown.notified() => break,
                    item = send_rx.recv() => item,
                };
                let Some(item) = item else { break };
                let result = match item {
                    SendItem::Audio(chunk) => {
                        send_text(&writer_for_worker, audio_message(&chunk)).await
                    }
                    SendItem::End(done) => {
                        let result = send_text(&writer_for_worker, audio_stream_end_message())
                            .await
                            .map_err(|error| error.to_string());
                        let _ = done.send(result);
                        continue;
                    }
                };
                if let Err(error) = result {
                    if let Some(this) = weak_self.upgrade() {
                        this.finish_error(error);
                    }
                    break;
                }
            }
        }));

        let weak_self = Arc::downgrade(self);
        let shutdown = Arc::clone(&self.shutdown);
        let shutdown_requested = Arc::clone(&self.shutdown_requested);
        let spawner = Arc::clone(&self.task_spawner);
        spawner.spawn(Box::pin(async move {
            let mut read = read;
            loop {
                if shutdown_requested.load(Ordering::Acquire) {
                    break;
                }
                let message = tokio::select! {
                    _ = shutdown.notified() => break,
                    message = read.next() => message,
                };
                let Some(message) = message else {
                    if let Some(this) = weak_self.upgrade() {
                        this.fail_session_start("WebSocket closed before setup completed");
                        this.finish_with_partial_or_error(GeminiLiveASRError::ConnectionFailed(
                            "WebSocket closed".into(),
                        ));
                    }
                    break;
                };
                let Some(this) = weak_self.upgrade() else {
                    break;
                };
                match message {
                    Ok(Message::Text(text)) => {
                        if !this.handle_server_message(&text) {
                            break;
                        }
                    }
                    Ok(Message::Close(_)) => {
                        this.fail_session_start("WebSocket closed before setup completed");
                        this.finish_with_partial_or_error(GeminiLiveASRError::NoFinalResult);
                        break;
                    }
                    Ok(_) => {}
                    Err(_) => {
                        this.fail_session_start("WebSocket receive failed");
                        this.finish_with_partial_or_error(GeminiLiveASRError::ConnectionFailed(
                            "WebSocket receive failed".into(),
                        ));
                        break;
                    }
                }
            }
        }));

        let started = self.session_started.notified();
        tokio::pin!(started);
        started.as_mut().enable();
        let setup = self.credentials.setup_message();
        if let Err(error) = send_text(&self.writer, setup).await {
            self.cancel();
            return Err(error);
        }
        let ready = if !self.state.lock().session_started {
            tokio::time::timeout(SESSION_READY_TIMEOUT, started)
                .await
                .map_err(|_| GeminiLiveASRError::FinalResultTimeout)
        } else {
            Ok(())
        };
        if let Err(error) = ready {
            self.cancel();
            return Err(error);
        }
        if let Some(error) = self.state.lock().session_start_error.clone() {
            self.cancel();
            return Err(GeminiLiveASRError::TaskFailed(error));
        }
        Ok(())
    }

    pub async fn send_last_frame(&self) -> Result<(), GeminiLiveASRError> {
        let finished = self.session_finished.notified();
        tokio::pin!(finished);
        finished.as_mut().enable();
        let (send_tx, tail) = {
            let mut state = self.state.lock();
            if state.session_finished {
                return Ok(());
            }
            if state.audio_stream_end_sent {
                return Ok(());
            }
            state.ending = true;
            if !state.pending_audio.is_empty() {
                let pending = std::mem::take(&mut state.pending_audio);
                state.audio_scratch.extend_from_slice(&pending);
            }
            let tail = std::mem::take(&mut state.audio_scratch);
            (state.send_tx.clone(), tail)
        };
        let Some(send_tx) = send_tx else {
            return Ok(());
        };
        if !tail.is_empty() {
            send_audio_chunks(&send_tx, &tail)?;
        }
        let (done_tx, done_rx) = oneshot::channel();
        {
            let mut state = self.state.lock();
            state.audio_stream_end_sent = true;
        }
        send_tx
            .send(SendItem::End(done_tx))
            .map_err(|_| GeminiLiveASRError::SendFailed("audio writer is not available".into()))?;
        done_rx
            .await
            .map_err(|_| {
                GeminiLiveASRError::SendFailed("audio end acknowledgement dropped".into())
            })?
            .map_err(GeminiLiveASRError::SendFailed)?;
        let complete_without_final = {
            let mut state = self.state.lock();
            state.audio_stream_end_written = true;
            state.turn_complete_received && !state.final_received
        };
        let complete_with_final = {
            let state = self.state.lock();
            state.turn_complete_received && state.final_received
        };
        if complete_with_final {
            self.finish_success();
            return Ok(());
        }
        if complete_without_final {
            self.finish_error(GeminiLiveASRError::NoFinalResult);
            return Ok(());
        }

        if !self.state.lock().session_finished {
            let _ = tokio::time::timeout(FINISH_HARD_DEADLINE, finished).await;
        }
        if !self.state.lock().session_finished {
            self.finish_with_partial_or_error(GeminiLiveASRError::FinalResultTimeout);
        }
        Ok(())
    }

    pub async fn await_final_result(&self) -> Result<RawTranscript, GeminiLiveASRError> {
        let rx = self
            .final_rx
            .lock()
            .take()
            .ok_or(GeminiLiveASRError::NoFinalResult)?;
        tokio::time::timeout(FINAL_RESULT_TIMEOUT, rx)
            .await
            .map_err(|_| GeminiLiveASRError::FinalResultTimeout)?
            .map_err(|_| GeminiLiveASRError::NoFinalResult)?
    }

    pub fn cancel(&self) {
        let mut state = self.state.lock();
        if state.session_finished {
            return;
        }
        state.pending_audio.clear();
        state.audio_scratch.clear();
        state.send_tx.take();
        state.session_finished = true;
        state.final_tx.take();
        drop(state);
        self.shutdown_requested.store(true, Ordering::Release);
        self.shutdown.notify_waiters();
        self.session_started.notify_waiters();
        self.session_finished.notify_waiters();
        self.close_on_runtime();
    }

    fn handle_server_message(&self, text: &str) -> bool {
        let value: Value = match serde_json::from_str(text) {
            Ok(value) => value,
            Err(_) => return true,
        };
        if value.get("setupComplete").is_some() {
            self.mark_session_started();
            return true;
        }
        if let Some(error) = value.get("error") {
            let message = error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("Gemini Live session error");
            self.finish_with_partial_or_error(GeminiLiveASRError::TaskFailed(self.redact(message)));
            return false;
        }
        if value.get("goAway").is_some() {
            self.finish_with_partial_or_error(GeminiLiveASRError::ConnectionFailed(
                "Gemini Live server requested disconnect".into(),
            ));
            return false;
        }
        let Some(content) = value.get("serverContent") else {
            return true;
        };
        if let Some(text) = content
            .get("interimInputTranscription")
            .and_then(|item| item.get("text"))
            .and_then(Value::as_str)
        {
            self.record_partial(text);
        }
        if let Some(text) = content
            .get("inputTranscription")
            .and_then(|item| item.get("text"))
            .and_then(Value::as_str)
        {
            self.record_final(text);
        }
        if content
            .get("turnComplete")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            let (should_finish, has_final) = {
                let mut state = self.state.lock();
                if !state.audio_stream_end_sent {
                    return true;
                }
                state.turn_complete_received = true;
                (
                    state.ending && state.audio_stream_end_written,
                    state.final_received,
                )
            };
            if should_finish {
                if has_final {
                    self.finish_success();
                } else {
                    self.finish_error(GeminiLiveASRError::NoFinalResult);
                }
                return false;
            }
        }
        true
    }

    fn mark_session_started(&self) {
        let (send_tx, chunks) = {
            let mut state = self.state.lock();
            if state.session_started {
                return;
            }
            state.session_started = true;
            if !state.pending_audio.is_empty() {
                let pending = std::mem::take(&mut state.pending_audio);
                state.audio_scratch.extend_from_slice(&pending);
            }
            let chunks = drain_audio_chunks(&mut state.audio_scratch);
            (state.send_tx.clone(), chunks)
        };
        if let Some(send_tx) = send_tx {
            for chunk in chunks {
                let _ = send_tx.send(SendItem::Audio(chunk));
            }
        }
        self.session_started.notify_waiters();
    }

    fn fail_session_start(&self, error: &str) {
        let mut state = self.state.lock();
        if !state.session_started && state.session_start_error.is_none() {
            state.session_start_error = Some(error.to_string());
            self.session_started.notify_waiters();
        }
    }

    fn record_partial(&self, text: &str) {
        let text = text.trim();
        if text.is_empty() {
            return;
        }
        let snapshot = {
            let mut state = self.state.lock();
            if state.session_finished {
                return;
            }
            state.partial_text = text.to_string();
            let mut segments = state.completed_segments.clone();
            segments.push(state.partial_text.clone());
            join_segments(&segments)
        };
        self.publish(snapshot);
    }

    fn record_final(&self, text: &str) {
        let text = text.trim();
        if text.is_empty() {
            return;
        }
        let (snapshot, should_finish) = {
            let mut state = self.state.lock();
            if state.session_finished || state.last_final_text.as_deref() == Some(text) {
                return;
            }
            state.last_final_text = Some(text.to_string());
            state.final_received = true;
            state.completed_segments.push(text.to_string());
            state.partial_text.clear();
            (
                join_segments(&state.completed_segments),
                state.ending && state.audio_stream_end_written && state.turn_complete_received,
            )
        };
        self.publish(snapshot);
        if should_finish {
            self.finish_success();
        }
    }

    fn publish(&self, text: String) {
        if let Some(sink) = self.partial_sink.lock().clone() {
            let _ = sink.publish(TextStreamChunk { text, offset: 0 });
        }
    }

    fn finish_success(&self) {
        let (tx, text, duration_ms) = {
            let mut state = self.state.lock();
            if state.session_finished {
                return;
            }
            state.session_finished = true;
            state.send_tx.take();
            if !state.partial_text.trim().is_empty() {
                let partial = std::mem::take(&mut state.partial_text);
                if state.last_final_text.as_deref() != Some(partial.as_str()) {
                    state.completed_segments.push(partial);
                }
            }
            let text = join_segments(&state.completed_segments);
            let duration_ms = if state.bytes_received > 0 {
                state.bytes_received / BYTES_PER_MS
            } else {
                state
                    .start
                    .map(|start| start.elapsed().as_millis() as u64)
                    .unwrap_or_default()
            };
            (state.final_tx.take(), text, duration_ms)
        };
        if let Some(tx) = tx {
            let _ = tx.send(Ok(RawTranscript { text, duration_ms }));
        }
        self.session_finished.notify_waiters();
        self.close_on_runtime();
    }

    fn finish_with_partial_or_error(&self, error: GeminiLiveASRError) {
        let has_text = {
            let state = self.state.lock();
            !state.completed_segments.is_empty() || !state.partial_text.trim().is_empty()
        };
        if has_text {
            self.finish_success();
        } else {
            self.finish_error(error);
        }
    }

    fn finish_error(&self, error: GeminiLiveASRError) {
        self.fail_session_start(&error.to_string());
        let tx = {
            let mut state = self.state.lock();
            if state.session_finished {
                return;
            }
            state.session_finished = true;
            state.send_tx.take();
            state.final_tx.take()
        };
        if let Some(tx) = tx {
            let _ = tx.send(Err(error));
        }
        self.session_started.notify_waiters();
        self.session_finished.notify_waiters();
        self.close_on_runtime();
    }

    fn close_on_runtime(&self) {
        let writer = Arc::clone(&self.writer);
        self.task_spawner.spawn(Box::pin(async move {
            let _ = close_writer(&writer).await;
        }));
    }

    fn redact(&self, message: &str) -> String {
        let key = self.credentials.api_key.trim();
        if key.is_empty() {
            message.to_string()
        } else {
            message.replace(key, "[redacted]")
        }
    }
}

impl Drop for GeminiLiveASR {
    fn drop(&mut self) {
        self.shutdown_requested.store(true, Ordering::Release);
        self.shutdown.notify_waiters();
    }
}

impl AudioConsumer for GeminiLiveASR {
    fn consume_pcm_chunk(&self, pcm: &[u8]) {
        if pcm.is_empty() {
            return;
        }
        let (send_tx, chunks) = {
            let mut state = self.state.lock();
            if state.session_finished || state.ending {
                return;
            }
            state.last_final_text = None;
            state.bytes_received = state.bytes_received.saturating_add(pcm.len() as u64);
            if !state.session_started {
                state.pending_audio.extend_from_slice(pcm);
                return;
            }
            state.audio_scratch.extend_from_slice(pcm);
            let chunks = drain_audio_chunks(&mut state.audio_scratch);
            (state.send_tx.clone(), chunks)
        };
        if let Some(send_tx) = send_tx {
            for chunk in chunks {
                let _ = send_tx.send(SendItem::Audio(chunk));
            }
        }
    }
}

fn drain_audio_chunks(buffer: &mut Vec<u8>) -> Vec<Vec<u8>> {
    let mut chunks = Vec::new();
    while buffer.len() >= TARGET_AUDIO_CHUNK_BYTES {
        chunks.push(buffer.drain(..TARGET_AUDIO_CHUNK_BYTES).collect());
    }
    chunks
}

fn send_audio_chunks(
    send_tx: &mpsc::UnboundedSender<SendItem>,
    audio: &[u8],
) -> Result<(), GeminiLiveASRError> {
    let mut buffer = audio.to_vec();
    for chunk in drain_audio_chunks(&mut buffer) {
        send_tx
            .send(SendItem::Audio(chunk))
            .map_err(|_| GeminiLiveASRError::SendFailed("audio writer is not available".into()))?;
    }
    if !buffer.is_empty() {
        send_tx
            .send(SendItem::Audio(buffer))
            .map_err(|_| GeminiLiveASRError::SendFailed("audio writer is not available".into()))?;
    }
    Ok(())
}

pub fn build_setup_message(
    model: &str,
    language_codes: &[String],
    custom_vocabulary: &[String],
    mode: GeminiLiveMode,
) -> String {
    let model = model.strip_prefix("models/").unwrap_or(model);
    let languages = language_codes
        .iter()
        .map(|language| normalize_language_code(language))
        .filter(|language| !language.is_empty())
        .collect::<Vec<_>>();
    let vocabulary = normalize_vocabulary(custom_vocabulary);
    let mut transcription = json!({"mode": mode.as_wire()});
    if !languages.is_empty() {
        transcription["languageCodes"] = json!(languages);
    }
    if !vocabulary.is_empty() {
        transcription["customVocabulary"] = json!(vocabulary);
    }
    json!({
        "setup": {
            "model": format!("models/{model}"),
            "generationConfig": {"responseModalities": ["TEXT"]},
            "inputAudioTranscription": transcription,
        }
    })
    .to_string()
}

fn normalize_vocabulary(values: &[String]) -> Vec<String> {
    let mut output = Vec::new();
    for value in values {
        let value = value.trim();
        if value.is_empty() || output.iter().any(|item: &String| item == value) {
            continue;
        }
        output.push(value.to_string());
        if output.len() == 1_000 {
            break;
        }
    }
    output
}

fn normalize_language_code(language: &str) -> String {
    match language.trim().to_ascii_lowercase().as_str() {
        "zh" | "zh-cn" | "zh-hans" => "cmn-Hans-CN".into(),
        "zh-tw" | "zh-hant" => "cmn-Hant-TW".into(),
        "en" => "en-US".into(),
        "ja" => "ja-JP".into(),
        "ko" => "ko-KR".into(),
        "" => String::new(),
        _ => language.trim().to_string(),
    }
}

fn audio_message(pcm: &[u8]) -> String {
    json!({
        "realtimeInput": {
            "audio": {
                "data": base64::engine::general_purpose::STANDARD.encode(pcm),
                "mimeType": "audio/pcm;rate=16000",
            }
        }
    })
    .to_string()
}

fn audio_stream_end_message() -> String {
    json!({"realtimeInput": {"audioStreamEnd": true}}).to_string()
}

async fn send_text(writer: &SharedWriter, text: String) -> Result<(), GeminiLiveASRError> {
    tokio::time::timeout(WRITE_TIMEOUT, async {
        let mut guard = writer.lock().await;
        let Some(ws) = guard.as_mut() else {
            return Err(GeminiLiveASRError::ConnectionFailed(
                "WebSocket writer is not available".into(),
            ));
        };
        ws.send(Message::Text(text))
            .await
            .map_err(|_| GeminiLiveASRError::SendFailed("WebSocket write failed".into()))
    })
    .await
    .map_err(|_| GeminiLiveASRError::SendFailed("WebSocket write timed out".into()))?
}

async fn close_writer(writer: &SharedWriter) -> Result<(), GeminiLiveASRError> {
    let mut guard = writer.lock().await;
    if let Some(mut ws) = guard.take() {
        let _ = ws.close().await;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setup_uses_live_model_and_transcription_options() {
        let message = build_setup_message(
            "gemini-3.5-transcribe-live",
            &["zh".into(), "en-US".into()],
            &[" Gemini ".into(), "Gemini".into()],
            GeminiLiveMode::Verbatim,
        );
        let value: Value = serde_json::from_str(&message).unwrap();
        assert_eq!(value["setup"]["model"], "models/gemini-3.5-transcribe-live");
        assert_eq!(
            value["setup"]["generationConfig"]["responseModalities"][0],
            "TEXT"
        );
        assert_eq!(
            value["setup"]["inputAudioTranscription"]["languageCodes"][0],
            "cmn-Hans-CN"
        );
        assert_eq!(
            value["setup"]["inputAudioTranscription"]["customVocabulary"],
            json!(["Gemini"])
        );
    }

    #[test]
    fn audio_is_raw_pcm_and_stream_end_is_distinct() {
        let audio: Value = serde_json::from_str(&audio_message(&[1, 2, 3])).unwrap();
        assert_eq!(audio["realtimeInput"]["audio"]["data"], "AQID");
        assert_eq!(
            audio["realtimeInput"]["audio"]["mimeType"],
            "audio/pcm;rate=16000"
        );
        assert!(audio["realtimeInput"]["audio"]["data"] != "UklGRg==");
        let end: Value = serde_json::from_str(&audio_stream_end_message()).unwrap();
        assert_eq!(end["realtimeInput"]["audioStreamEnd"], true);
    }

    #[test]
    fn url_encodes_key_and_does_not_duplicate_auth_query() {
        let credentials = GeminiLiveCredentials {
            api_key: "a+b&c".into(),
            endpoint: "ws://localhost/live?key=old&x=1".into(),
            model: DEFAULT_MODEL.into(),
            language_codes: Vec::new(),
            custom_vocabulary: Vec::new(),
            mode: GeminiLiveMode::Smart,
        };
        let url = credentials.connect_url().unwrap();
        let parsed = url::Url::parse(&url).unwrap();
        let pairs = parsed.query_pairs().collect::<Vec<_>>();
        assert_eq!(pairs.iter().filter(|(key, _)| key == "key").count(), 1);
        assert_eq!(
            pairs.iter().find(|(key, _)| key == "key").unwrap().1,
            "a+b&c"
        );
        assert_eq!(pairs.iter().find(|(key, _)| key == "x").unwrap().1, "1");
    }

    #[test]
    fn turn_complete_before_audio_stream_end_is_ignored() {
        let provider = GeminiLiveASR::new(GeminiLiveCredentials {
            api_key: "test-key".into(),
            endpoint: "ws://localhost/live".into(),
            model: DEFAULT_MODEL.into(),
            language_codes: Vec::new(),
            custom_vocabulary: Vec::new(),
            mode: GeminiLiveMode::Verbatim,
        });
        let message = r#"{"serverContent":{"turnComplete":true}}"#;

        assert!(provider.handle_server_message(message));
        assert!(!provider.state.lock().turn_complete_received);

        provider.state.lock().audio_stream_end_sent = true;
        assert!(provider.handle_server_message(message));
        assert!(provider.state.lock().turn_complete_received);
    }

    #[tokio::test]
    async fn websocket_streams_audio_and_returns_final_transcript() {
        use futures_util::StreamExt;
        use tokio::net::TcpListener;
        use tokio_tungstenite::accept_async;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("ws://{}/live", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut socket = accept_async(stream).await.unwrap();
            let setup = socket.next().await.unwrap().unwrap();
            let setup = match setup {
                Message::Text(text) => serde_json::from_str::<Value>(text.as_ref()).unwrap(),
                other => panic!("unexpected setup frame: {other:?}"),
            };
            assert_eq!(setup["setup"]["model"], "models/gemini-3.5-transcribe-live");
            socket
                .send(Message::Text(r#"{"setupComplete":{}}"#.to_string()))
                .await
                .unwrap();

            let mut saw_audio = false;
            while let Some(message) = socket.next().await {
                let message = message.unwrap();
                let Message::Text(text) = message else {
                    continue;
                };
                let value: Value = serde_json::from_str(text.as_ref()).unwrap();
                if value["realtimeInput"]["audio"]["data"].is_string() {
                    saw_audio = true;
                    continue;
                }
                if value["realtimeInput"]["audioStreamEnd"] == true {
                    assert!(saw_audio);
                    socket
                        .send(Message::Text(
                            r#"{"serverContent":{"interimInputTranscription":{"text":"hello wor"}}}"#
                                .to_string(),
                        ))
                        .await
                        .unwrap();
                    socket
                        .send(Message::Text(
                            r#"{"serverContent":{"inputTranscription":{"text":"hello world"},"turnComplete":true}}"#
                                .to_string(),
                        ))
                        .await
                        .unwrap();
                    break;
                }
            }
        });

        let provider = Arc::new(GeminiLiveASR::new(GeminiLiveCredentials {
            api_key: "test-key".into(),
            endpoint,
            model: DEFAULT_MODEL.into(),
            language_codes: Vec::new(),
            custom_vocabulary: Vec::new(),
            mode: GeminiLiveMode::Verbatim,
        }));
        let sink = Arc::new(super::super::TranscriptCapture::default());
        provider.set_partial_sink(sink.clone());
        provider.open_session().await.unwrap();
        provider.consume_pcm_chunk(&vec![1; TARGET_AUDIO_CHUNK_BYTES]);
        provider.send_last_frame().await.unwrap();
        let result = provider.await_final_result().await.unwrap();
        assert_eq!(result.text, "hello world");
        sink.assert_snapshots(&["hello wor", "hello world"]);
        tokio::time::timeout(Duration::from_secs(2), server)
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn cancellation_closes_live_session_without_sending_cancel_message() {
        use futures_util::StreamExt;
        use tokio::net::TcpListener;
        use tokio_tungstenite::accept_async;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("ws://{}/live", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut socket = accept_async(stream).await.unwrap();
            let _ = socket.next().await;
            socket
                .send(Message::Text(r#"{"setupComplete":{}}"#.to_string()))
                .await
                .unwrap();
            let _ = tokio::time::timeout(Duration::from_secs(2), socket.next()).await;
        });
        let provider = Arc::new(GeminiLiveASR::new(GeminiLiveCredentials {
            api_key: "test-key".into(),
            endpoint,
            model: DEFAULT_MODEL.into(),
            language_codes: Vec::new(),
            custom_vocabulary: Vec::new(),
            mode: GeminiLiveMode::Verbatim,
        }));
        provider.open_session().await.unwrap();
        provider.consume_pcm_chunk(&[1, 2, 3]);
        provider.cancel();
        assert!(provider.await_final_result().await.is_err());
        tokio::time::timeout(Duration::from_secs(2), server)
            .await
            .unwrap()
            .unwrap();
    }
}
