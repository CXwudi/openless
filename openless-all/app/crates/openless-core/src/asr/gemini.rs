//! Gemini dedicated batch transcription client.
//!
//! `gemini-3.5-transcribe` uses the Files API followed by the Interactions API.
//! It is intentionally separate from the native Gemini generateContent client:
//! the request envelope and response lifecycle are different.

use anyhow::{Context, Result};
use futures_util::StreamExt;
use parking_lot::Mutex;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::asr::wav::encode_wav_16k_mono;
use crate::asr::RawTranscript;

pub const PROVIDER_ID: &str = "gemini";
pub const DEFAULT_ENDPOINT: &str = "https://generativelanguage.googleapis.com/v1beta";
pub const DEFAULT_MODEL: &str = "gemini-3.5-transcribe";
const MAX_RESPONSE_BYTES: usize = 4 * 1024 * 1024;
const UPLOAD_TIMEOUT: Duration = Duration::from_secs(90);
const INTERACTION_TIMEOUT: Duration = Duration::from_secs(180);
const POLL_INTERVAL: Duration = Duration::from_millis(500);
const MAX_POLL_ATTEMPTS: usize = 240;
const CANCEL_TIMEOUT: Duration = Duration::from_secs(10);
const FILE_CLEANUP_TIMEOUT: Duration = Duration::from_secs(10);

struct UploadedFile {
    name: String,
}

pub struct GeminiBatchASR {
    api_key: String,
    base_url: String,
    model: String,
    language_codes: Vec<String>,
    custom_vocabulary: Vec<String>,
    buffer: Mutex<Vec<u8>>,
    active_interaction: Mutex<Option<String>>,
    cancelled: AtomicBool,
}

impl GeminiBatchASR {
    pub fn new(
        api_key: String,
        base_url: String,
        model: String,
        language: Option<String>,
        hotwords: Vec<String>,
    ) -> Self {
        Self {
            api_key,
            base_url,
            model,
            language_codes: language
                .and_then(|language| language_code(&language))
                .into_iter()
                .collect(),
            custom_vocabulary: normalize_vocabulary(hotwords),
            buffer: Mutex::new(Vec::new()),
            active_interaction: Mutex::new(None),
            cancelled: AtomicBool::new(false),
        }
    }

    pub fn buffer_duration_ms(&self) -> u64 {
        crate::asr::pcm::pcm_duration_ms(&self.buffer.lock())
    }

    pub fn transcribe_timeout(&self) -> Duration {
        INTERACTION_TIMEOUT + UPLOAD_TIMEOUT
    }

    pub async fn transcribe(&self) -> Result<RawTranscript> {
        let pcm = self.buffer.lock().clone();
        if pcm.is_empty() {
            return Ok(RawTranscript {
                text: String::new(),
                duration_ms: 0,
            });
        }
        if self.cancelled.load(Ordering::Acquire) {
            anyhow::bail!("Gemini ASR request cancelled");
        }

        let result = self
            .transcribe_inner(&pcm, Instant::now() + self.transcribe_timeout())
            .await;
        if result.is_ok() {
            self.buffer.lock().clear();
        }
        result
    }

    async fn transcribe_inner(&self, pcm: &[u8], deadline: Instant) -> Result<RawTranscript> {
        if self.api_key.trim().is_empty() {
            anyhow::bail!("Gemini API key missing");
        }
        if self.model.trim().is_empty() {
            anyhow::bail!("Gemini transcription model missing");
        }
        if self.cancelled.load(Ordering::Acquire) {
            anyhow::bail!("Gemini ASR request cancelled");
        }

        let duration_ms = crate::asr::pcm::pcm_duration_ms(pcm);
        let samples: Vec<i16> = pcm
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
            .collect();
        let wav = encode_wav_16k_mono(&samples);
        let file = self.upload_wav(&wav, deadline).await?;
        let result = async {
            self.wait_for_file_ready(&file.name, deadline).await?;
            if self.cancelled.load(Ordering::Acquire) {
                anyhow::bail!("Gemini ASR request cancelled");
            }
            let interaction = self.create_interaction(&file.name, deadline).await?;
            if let Some(id) = interaction_id(&interaction) {
                *self.active_interaction.lock() = Some(id.clone());
                if self.cancelled.load(Ordering::Acquire) {
                    let _ = self.cancel_interaction(&id).await;
                    anyhow::bail!("Gemini ASR request cancelled");
                }
            }
            self.resolve_interaction(interaction, deadline).await
        }
        .await;
        let interaction = self.active_interaction.lock().take();
        if result.is_err() {
            if let Some(interaction) = interaction {
                let _ = self.cancel_interaction(&interaction).await;
            }
        }
        // Files are temporary provider-side resources. Cleanup is best effort so a
        // successful transcript is not discarded because deletion was unavailable.
        let _ = self.delete_file(&file.name, FILE_CLEANUP_TIMEOUT).await;
        let text = result?;
        Ok(RawTranscript { text, duration_ms })
    }

    async fn upload_wav(&self, wav: &[u8], deadline: Instant) -> Result<UploadedFile> {
        let url = files_upload_url(&self.base_url)?;
        let client = crate::net::credential_http_for_url(&url);
        let start = client
            .post(&url)
            .header("x-goog-api-key", self.api_key.trim())
            .header("X-Goog-Upload-Protocol", "resumable")
            .header("X-Goog-Upload-Command", "start")
            .header("X-Goog-Upload-Header-Content-Length", wav.len().to_string())
            .header("X-Goog-Upload-Header-Content-Type", "audio/wav")
            .json(&json!({
                "file": { "display_name": "openless-dictation.wav" }
            }))
            .timeout(remaining_timeout(deadline, UPLOAD_TIMEOUT)?)
            .send()
            .await
            .context("Gemini Files API upload initialization failed")?;
        if !start.status().is_success() {
            anyhow::bail!(
                "Gemini Files API initialization failed ({})",
                start.status()
            );
        }
        let upload_url = start
            .headers()
            .get("x-goog-upload-url")
            .context("Gemini Files API response missing upload URL")?
            .to_str()
            .context("Gemini Files API upload URL is invalid")?
            .to_owned();
        validate_upload_url(&url, &upload_url)?;

        let response = client
            .post(&upload_url)
            .header("x-goog-api-key", self.api_key.trim())
            .header("X-Goog-Upload-Offset", "0")
            .header("X-Goog-Upload-Command", "upload, finalize")
            .header("Content-Type", "audio/wav")
            .body(wav.to_vec())
            .timeout(remaining_timeout(deadline, UPLOAD_TIMEOUT)?)
            .send()
            .await
            .context("Gemini Files API audio upload failed")?;
        let status = response.status();
        let body = read_response_limited(response)
            .await
            .context("read Gemini Files API response")?;
        if !status.is_success() {
            anyhow::bail!("Gemini Files API upload failed ({status})");
        }
        let json: Value =
            serde_json::from_slice(&body).context("parse Gemini Files API response")?;
        let uri = json
            .pointer("/file/uri")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .filter(|uri| !uri.trim().is_empty())
            .context("Gemini Files API response missing file URI")?;
        let name = json
            .pointer("/file/name")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .filter(|name| !name.trim().is_empty())
            .or_else(|| uri.rsplit('/').next().map(|id| format!("files/{id}")))
            .context("Gemini Files API response missing file name")?;
        Ok(UploadedFile { name })
    }

    async fn wait_for_file_ready(&self, name: &str, deadline: Instant) -> Result<()> {
        let url = file_url(&self.base_url, name)?;
        for _ in 0..=60 {
            if self.cancelled.load(Ordering::Acquire) {
                anyhow::bail!("Gemini ASR request cancelled");
            }
            let response = crate::net::credential_http_for_url(&url)
                .get(&url)
                .header("x-goog-api-key", self.api_key.trim())
                .timeout(remaining_timeout(deadline, UPLOAD_TIMEOUT)?)
                .send()
                .await
                .context("Gemini Files API state request failed")?;
            let status = response.status();
            let body = read_response_limited(response)
                .await
                .context("read Gemini Files API state response")?;
            if !status.is_success() {
                anyhow::bail!("Gemini Files API state request failed ({status})");
            }
            let json: Value =
                serde_json::from_slice(&body).context("parse Gemini Files API state response")?;
            let state = json
                .get("state")
                .or_else(|| json.pointer("/file/state"))
                .and_then(Value::as_str);
            match state {
                Some("ACTIVE") => return Ok(()),
                Some("FAILED") => anyhow::bail!("Gemini uploaded audio failed processing"),
                Some(_) => tokio::time::sleep(remaining_sleep(deadline, POLL_INTERVAL)?).await,
                None => anyhow::bail!("Gemini Files API state response missing state"),
            }
        }
        anyhow::bail!("Gemini uploaded audio was not ready in time")
    }

    async fn delete_file(&self, name: &str, timeout: Duration) -> Result<()> {
        let url = file_url(&self.base_url, name)?;
        let response = crate::net::credential_http_for_url(&url)
            .delete(&url)
            .header("x-goog-api-key", self.api_key.trim())
            .timeout(timeout)
            .send()
            .await?;
        if response.status().is_success() || response.status() == reqwest::StatusCode::NOT_FOUND {
            Ok(())
        } else {
            anyhow::bail!("Gemini Files API deletion failed ({})", response.status())
        }
    }

    async fn create_interaction(&self, file_uri: &str, deadline: Instant) -> Result<Value> {
        let url = interactions_url(&self.base_url)?;
        let body = interaction_body(
            &self.model,
            file_uri,
            &self.language_codes,
            &self.custom_vocabulary,
        );
        let client = crate::net::credential_http_for_url(&url);
        let response = client
            .post(&url)
            .header("x-goog-api-key", self.api_key.trim())
            .json(&body)
            .timeout(remaining_timeout(deadline, INTERACTION_TIMEOUT)?)
            .send()
            .await
            .context("Gemini Interactions API request failed")?;
        let status = response.status();
        let body = read_response_limited(response)
            .await
            .context("read Gemini Interactions API response")?;
        if !status.is_success() {
            anyhow::bail!("Gemini Interactions API request failed ({status})");
        }
        serde_json::from_slice(&body).context("parse Gemini Interaction response")
    }

    async fn resolve_interaction(
        &self,
        mut interaction: Value,
        deadline: Instant,
    ) -> Result<String> {
        for _ in 0..=MAX_POLL_ATTEMPTS {
            if self.cancelled.load(Ordering::Acquire) {
                anyhow::bail!("Gemini ASR request cancelled");
            }
            match interaction_status(&interaction) {
                InteractionStatus::Completed => {
                    return extract_interaction_text(&interaction)
                        .context("Gemini Interaction completed without transcript");
                }
                InteractionStatus::Failed => anyhow::bail!("Gemini Interaction failed"),
                InteractionStatus::Cancelled => anyhow::bail!("Gemini Interaction was cancelled"),
                InteractionStatus::Incomplete => {
                    anyhow::bail!("Gemini Interaction was incomplete")
                }
                InteractionStatus::Pending => {
                    let id = interaction_id(&interaction)
                        .context("Gemini Interaction response missing id")?;
                    tokio::time::sleep(remaining_sleep(deadline, POLL_INTERVAL)?).await;
                    interaction = self.get_interaction(&id, deadline).await?;
                }
            }
        }
        anyhow::bail!("Gemini Interaction timed out")
    }

    async fn get_interaction(&self, id: &str, deadline: Instant) -> Result<Value> {
        let url = interaction_url(&self.base_url, id)?;
        let response = crate::net::credential_http_for_url(&url)
            .get(&url)
            .header("x-goog-api-key", self.api_key.trim())
            .timeout(remaining_timeout(deadline, INTERACTION_TIMEOUT)?)
            .send()
            .await
            .context("Gemini Interaction status request failed")?;
        let status = response.status();
        let body = read_response_limited(response)
            .await
            .context("read Gemini Interaction status response")?;
        if !status.is_success() {
            anyhow::bail!("Gemini Interaction status request failed ({status})");
        }
        serde_json::from_slice(&body).context("parse Gemini Interaction status response")
    }

    async fn cancel_interaction(&self, id: &str) -> Result<()> {
        let url = interaction_url(&self.base_url, id)?;
        let cancel_url = format!("{url}/cancel");
        let response = crate::net::credential_http_for_url(&cancel_url)
            .post(&cancel_url)
            .header("x-goog-api-key", self.api_key.trim())
            .timeout(CANCEL_TIMEOUT)
            .send()
            .await
            .context("Gemini Interaction cancellation request failed")?;
        if response.status().is_success()
            || response.status() == reqwest::StatusCode::NOT_FOUND
            || response.status() == reqwest::StatusCode::CONFLICT
        {
            Ok(())
        } else {
            anyhow::bail!(
                "Gemini Interaction cancellation failed ({})",
                response.status()
            )
        }
    }

    pub async fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        self.buffer.lock().clear();
        let interaction = self.active_interaction.lock().clone();
        if let Some(interaction) = interaction {
            let _ = self.cancel_interaction(&interaction).await;
        }
    }
}

impl super::AudioConsumer for GeminiBatchASR {
    fn consume_pcm_chunk(&self, pcm: &[u8]) {
        if !self.cancelled.load(Ordering::Acquire) {
            self.buffer.lock().extend_from_slice(pcm);
        }
    }
}

pub fn interactions_url(base_url: &str) -> Result<String> {
    endpoint_url(base_url, "/interactions")
}

pub fn files_upload_url(base_url: &str) -> Result<String> {
    let parsed = reqwest::Url::parse(base_url.trim()).context("parse Gemini endpoint")?;
    let path = parsed.path().trim_end_matches('/');
    let path = if path.ends_with("/files") && path.contains("/upload/") {
        path.to_string()
    } else {
        format!("/upload{path}/files")
    };
    let mut url = parsed;
    url.set_path(&path);
    url.set_query(None);
    url.set_fragment(None);
    Ok(url.to_string())
}

fn validate_upload_url(source: &str, upload: &str) -> Result<()> {
    let source = reqwest::Url::parse(source).context("parse Gemini upload endpoint")?;
    let upload = reqwest::Url::parse(upload).context("parse Gemini resumable upload URL")?;
    if source.origin() != upload.origin() || !matches!(upload.scheme(), "http" | "https") {
        anyhow::bail!("Gemini Files API returned an unsafe upload URL");
    }
    Ok(())
}

fn file_url(base_url: &str, name: &str) -> Result<String> {
    let name = name.trim().trim_start_matches("files/");
    if name.is_empty() || name.contains(['/', '\\', '?', '#']) {
        anyhow::bail!("Gemini file name is invalid");
    }
    endpoint_url(base_url, &format!("/files/{name}"))
}

fn endpoint_url(base_url: &str, suffix: &str) -> Result<String> {
    let mut url = reqwest::Url::parse(base_url.trim()).context("parse Gemini endpoint")?;
    let path = url.path().trim_end_matches('/');
    let path = if path.ends_with(suffix) {
        path.to_string()
    } else {
        format!("{path}{suffix}")
    };
    url.set_path(&path);
    url.set_query(None);
    url.set_fragment(None);
    Ok(url.to_string())
}

pub fn interaction_body(
    model: &str,
    file_id: &str,
    language_codes: &[String],
    custom_vocabulary: &[String],
) -> Value {
    let mut transcription_config = json!({
        "mode": { "type": "verbatim" }
    });
    if !language_codes.is_empty() {
        transcription_config["language_codes"] = json!(language_codes);
    }
    if !custom_vocabulary.is_empty() {
        transcription_config["custom_vocabulary"] = json!(custom_vocabulary);
    }
    json!({
        "model": model,
        "input": [{
            "type": "audio",
            "audio": { "file_id": file_id }
        }],
        "generation_config": {
            "transcription_config": transcription_config
        }
    })
}

fn interaction_url(base_url: &str, id: &str) -> Result<String> {
    let id = id.strip_prefix("interactions/").unwrap_or(id);
    if id.is_empty() || id == "." || id == ".." || id.contains(['/', '\\', '?', '#']) {
        anyhow::bail!("Gemini Interaction id is invalid");
    }
    endpoint_url(base_url, &format!("/interactions/{id}"))
}

fn interaction_id(interaction: &Value) -> Option<String> {
    interaction
        .get("id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .map(str::to_owned)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InteractionStatus {
    Pending,
    Completed,
    Failed,
    Cancelled,
    Incomplete,
}

fn interaction_status(interaction: &Value) -> InteractionStatus {
    match interaction
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "completed" => InteractionStatus::Completed,
        "failed" => InteractionStatus::Failed,
        "cancelled" | "canceled" => InteractionStatus::Cancelled,
        "incomplete" => InteractionStatus::Incomplete,
        "queued" | "in_progress" | "requires_action" => InteractionStatus::Pending,
        _ if extract_interaction_text(interaction).is_some() => InteractionStatus::Completed,
        _ => InteractionStatus::Pending,
    }
}

fn extract_interaction_text(interaction: &Value) -> Option<String> {
    if let Some(text) = interaction.get("output_text").and_then(Value::as_str) {
        if !text.trim().is_empty() {
            return Some(text.trim().to_owned());
        }
    }
    for key in ["outputs", "steps", "output"] {
        let Some(value) = interaction.get(key) else {
            continue;
        };
        let mut pieces = Vec::new();
        collect_text(value, &mut pieces);
        let text = pieces.join("").trim().to_owned();
        if !text.is_empty() {
            return Some(text);
        }
    }
    None
}

fn collect_text(value: &Value, pieces: &mut Vec<String>) {
    match value {
        Value::Array(items) => items.iter().for_each(|item| collect_text(item, pieces)),
        Value::Object(object) => {
            if let Some(text) = object.get("text").and_then(Value::as_str) {
                if !text.is_empty() {
                    pieces.push(text.to_owned());
                }
            } else if let Some(content) = object.get("content") {
                collect_text(content, pieces);
            }
            if object.get("text").is_none() {
                if let Some(output) = object.get("output") {
                    collect_text(output, pieces);
                }
            }
        }
        _ => {}
    }
}

fn normalize_vocabulary(words: Vec<String>) -> Vec<String> {
    let mut normalized = Vec::new();
    for word in words {
        let word = word.trim();
        if word.is_empty() || normalized.iter().any(|item: &String| item == word) {
            continue;
        }
        normalized.push(word.to_owned());
        if normalized.len() == 1_000 {
            break;
        }
    }
    normalized
}

fn language_code(language: &str) -> Option<String> {
    let language = language.trim();
    if language.is_empty() {
        return None;
    }
    if let Some(code) = crate::language_catalog::asr_language_code(language) {
        return Some(code);
    }
    let valid = language.split('-').all(|part| {
        !part.is_empty()
            && part
                .chars()
                .all(|character| character.is_ascii_alphanumeric())
    });
    valid.then(|| language.to_owned())
}

fn remaining_timeout(deadline: Instant, maximum: Duration) -> Result<Duration> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining.is_zero() {
        anyhow::bail!("Gemini ASR request timed out");
    }
    Ok(remaining.min(maximum))
}

fn remaining_sleep(deadline: Instant, maximum: Duration) -> Result<Duration> {
    remaining_timeout(deadline, maximum)
}

async fn read_response_limited(response: reqwest::Response) -> Result<Vec<u8>> {
    let mut stream = response.bytes_stream();
    let mut body = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("read Gemini response")?;
        if body.len().saturating_add(chunk.len()) > MAX_RESPONSE_BYTES {
            anyhow::bail!("Gemini response too large");
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ports::AudioConsumer;
    use base64::Engine;
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::thread;

    #[test]
    fn builds_files_and_interaction_urls() {
        assert_eq!(
            files_upload_url(DEFAULT_ENDPOINT).unwrap(),
            "https://generativelanguage.googleapis.com/upload/v1beta/files"
        );
        assert_eq!(
            interactions_url(DEFAULT_ENDPOINT).unwrap(),
            "https://generativelanguage.googleapis.com/v1beta/interactions"
        );
    }

    #[test]
    fn serializes_documented_interaction_shape() {
        let body = interaction_body(
            DEFAULT_MODEL,
            "files/abc",
            &["zh".to_string()],
            &["Kubernetes".to_string()],
        );
        assert_eq!(body["model"], DEFAULT_MODEL);
        assert_eq!(body["input"][0]["type"], "audio");
        assert_eq!(body["input"][0]["audio"]["file_id"], "files/abc");
        assert_eq!(
            body["generation_config"]["transcription_config"]["language_codes"][0],
            "zh"
        );
        assert_eq!(
            body["generation_config"]["transcription_config"]["custom_vocabulary"][0],
            "Kubernetes"
        );
        assert_eq!(
            body["generation_config"]["transcription_config"]["mode"]["type"],
            "verbatim"
        );
    }

    #[tokio::test]
    async fn transcribes_through_files_and_interactions_lifecycle() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let mut request = accept_request(&listener);
            assert!(request.0.starts_with("POST /upload/v1beta/files HTTP/1.1"));
            assert!(request
                .0
                .to_ascii_lowercase()
                .contains("x-goog-api-key: key"));
            assert!(request
                .0
                .to_ascii_lowercase()
                .contains("x-goog-upload-command: start"));
            assert!(String::from_utf8_lossy(&request.1).contains("openless-dictation.wav"));
            write_response(
                &mut request.2,
                "200 OK",
                &[(
                    "X-Goog-Upload-URL",
                    &format!("http://{address}/upload-session"),
                )],
                "",
            );

            request = accept_request(&listener);
            assert!(request.0.starts_with("POST /upload-session HTTP/1.1"));
            assert!(request
                .0
                .to_ascii_lowercase()
                .contains("x-goog-upload-command: upload, finalize"));
            assert!(request
                .0
                .to_ascii_lowercase()
                .contains("x-goog-upload-offset: 0"));
            assert!(request.1.starts_with(b"RIFF"));
            write_response(
                &mut request.2,
                "200 OK",
                &[],
                &format!(
                    r#"{{"file":{{"name":"files/file-1","uri":"http://{address}/v1beta/files/file-1"}},"state":"ACTIVE"}}"#
                ),
            );

            request = accept_request(&listener);
            assert!(request.0.starts_with("GET /v1beta/files/file-1 HTTP/1.1"));
            write_response(
                &mut request.2,
                "200 OK",
                &[],
                r#"{"name":"files/file-1","state":"ACTIVE"}"#,
            );

            request = accept_request(&listener);
            assert!(request.0.starts_with("POST /v1beta/interactions HTTP/1.1"));
            let body: Value = serde_json::from_slice(&request.1).unwrap();
            assert_eq!(body["model"], DEFAULT_MODEL);
            assert_eq!(body["input"][0]["type"], "audio");
            assert_eq!(body["input"][0]["audio"]["file_id"], "files/file-1");
            assert_eq!(
                body["generation_config"]["transcription_config"]["language_codes"][0],
                "en"
            );
            assert_eq!(
                body["generation_config"]["transcription_config"]["custom_vocabulary"][0],
                "Kubernetes"
            );
            write_response(
                &mut request.2,
                "200 OK",
                &[],
                r#"{"id":"interactions/i-1","status":"completed","steps":[{"type":"model_output","content":[{"type":"text","text":"hello Gemini"}]}]}"#,
            );

            request = accept_request(&listener);
            assert!(request
                .0
                .starts_with("DELETE /v1beta/files/file-1 HTTP/1.1"));
            write_response(&mut request.2, "200 OK", &[], "");
        });

        let provider = GeminiBatchASR::new(
            "key".to_string(),
            format!("http://{address}/v1beta"),
            DEFAULT_MODEL.to_string(),
            Some("en".to_string()),
            vec![" Kubernetes ".to_string()],
        );
        provider.consume_pcm_chunk(&vec![0_u8; 32_000]);
        let transcript = provider.transcribe().await.unwrap();
        assert_eq!(transcript.text, "hello Gemini");
        assert_eq!(transcript.duration_ms, 1_000);
        assert_eq!(provider.buffer_duration_ms(), 0);
        server.join().unwrap();
    }

    #[test]
    fn extracts_output_text_and_steps_text() {
        assert_eq!(
            extract_interaction_text(&json!({ "output_text": " hello " })),
            Some("hello".to_string())
        );
        assert_eq!(
            extract_interaction_text(&json!({
                "steps": [{ "content": [{ "type": "text", "text": "你好" }] }]
            })),
            Some("你好".to_string())
        );
    }

    #[test]
    fn pending_interaction_requires_polling() {
        assert_eq!(
            interaction_status(&json!({ "status": "in_progress", "id": "abc" })),
            InteractionStatus::Pending
        );
        assert_eq!(
            interaction_status(&json!({ "status": "completed", "output_text": "ok" })),
            InteractionStatus::Completed
        );
    }

    #[test]
    fn vocabulary_is_bounded_and_deduplicated() {
        let words = normalize_vocabulary(vec![" foo ".into(), "foo".into(), "bar".into()]);
        assert_eq!(words, ["foo", "bar"]);
    }

    #[test]
    fn encodes_canonical_pcm_as_wav_before_upload() {
        let samples = vec![0_i16; 16_000];
        let wav = encode_wav_16k_mono(&samples);
        assert_eq!(&wav[..4], b"RIFF");
        assert_eq!(wav.len(), 44 + 32_000);
        assert_eq!(
            base64::engine::general_purpose::STANDARD.encode(&wav).len(),
            42_728
        );
    }

    fn accept_request(listener: &TcpListener) -> (String, Vec<u8>, TcpStream) {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut bytes = Vec::new();
        let mut buffer = [0_u8; 4096];
        let header_end = loop {
            let count = stream.read(&mut buffer).unwrap();
            assert!(count > 0, "client closed before sending HTTP headers");
            bytes.extend_from_slice(&buffer[..count]);
            if let Some(index) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
                break index + 4;
            }
        };
        let header = String::from_utf8_lossy(&bytes[..header_end]).to_string();
        let content_length = header
            .lines()
            .find_map(|line| {
                let line = line.to_ascii_lowercase();
                line.strip_prefix("content-length:")
                    .map(str::trim)
                    .and_then(|value| value.parse::<usize>().ok())
            })
            .unwrap_or(0);
        while bytes.len() < header_end + content_length {
            let count = stream.read(&mut buffer).unwrap();
            assert!(count > 0, "client closed before sending HTTP body");
            bytes.extend_from_slice(&buffer[..count]);
        }
        (
            header,
            bytes[header_end..header_end + content_length].to_vec(),
            stream,
        )
    }

    fn write_response(stream: &mut TcpStream, status: &str, headers: &[(&str, &str)], body: &str) {
        let mut response = format!(
            "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n",
            body.len()
        );
        for (name, value) in headers {
            response.push_str(&format!("{name}: {value}\r\n"));
        }
        response.push_str("\r\n");
        stream.write_all(response.as_bytes()).unwrap();
        stream.write_all(body.as_bytes()).unwrap();
        stream.flush().unwrap();
    }
}
