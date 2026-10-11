//! Smart transcription adapter for the single-pass multimodal pipeline.

use std::sync::Arc;
use std::time::Duration;

use super::OmniConfig;
use crate::asr::gemini::{GeminiBatchASR, TranscriptionMode};
use crate::polish::LLMError;

/// Uses the dedicated audio API instead of Gemini's general-purpose chat API.
pub struct GeminiTranscribeOmni {
    config: OmniConfig,
    languages: Vec<String>,
    hotwords: Vec<String>,
}

impl GeminiTranscribeOmni {
    pub fn new(config: OmniConfig) -> Self {
        Self {
            config,
            languages: Vec::new(),
            hotwords: Vec::new(),
        }
    }

    pub fn set_hints(&mut self, languages: &[String], hotwords: &[String]) {
        self.languages = languages.to_vec();
        self.hotwords = hotwords.to_vec();
    }

    /// Smart mode supplies its own cleanup rules. The API rejects system instructions.
    /// Reject text tasks locally so a transcript cannot be mistaken for a Q&A answer.
    pub async fn complete(
        &self,
        user_text: &str,
        wav_bytes: Option<&[u8]>,
    ) -> Result<String, LLMError> {
        let wav = wav_bytes.filter(|wav| !wav.is_empty()).ok_or_else(|| {
            LLMError::ParseError(
                "Gemini Transcribe requires audio; choose a general-purpose model for text tasks"
                    .into(),
            )
        })?;
        if !user_text.trim().is_empty() {
            return Err(LLMError::ParseError(
                "Gemini Transcribe supports dictation only; choose a general-purpose model for Q&A"
                    .into(),
            ));
        }
        let provider = Arc::new(
            GeminiBatchASR::new(
                self.config.api_key.clone(),
                self.config.base_url.clone(),
                self.config.model.trim().to_string(),
                None,
                self.hotwords.clone(),
            )
            .with_mode(TranscriptionMode::Smart)
            .with_language_hints(&self.languages),
        );
        let worker = Arc::clone(&provider);
        let wav = wav.to_vec();
        // The engine may drop this future on cancellation. Keep the worker alive
        // long enough to cancel the interaction and delete its temporary file.
        let task = tokio::spawn(async move { worker.transcribe_wav(&wav).await });
        let mut cleanup = CancelOnDrop(Some(provider));
        let result = task.await;
        cleanup.0 = None;
        result
            .map_err(|_| LLMError::Network("Gemini transcription task failed".into()))?
            .map_err(|error| LLMError::Network(error.to_string()))
    }

    /// Recorded-audio transcription returns one final chunk after recording stops.
    pub async fn complete_streaming<F, C>(
        &self,
        user_text: &str,
        wav_bytes: Option<&[u8]>,
        on_delta: F,
        should_cancel: C,
    ) -> Result<String, LLMError>
    where
        F: Fn(&str) + Send + Sync,
        C: Fn() -> bool + Send + Sync,
    {
        let cancelled = async {
            loop {
                if should_cancel() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        };
        let text = tokio::select! {
            biased;
            _ = cancelled => return Err(LLMError::Network("Gemini transcription cancelled".into())),
            result = self.complete(user_text, wav_bytes) => result?,
        };
        if should_cancel() {
            return Err(LLMError::Network("Gemini transcription cancelled".into()));
        }
        if !text.is_empty() {
            on_delta(&text);
        }
        Ok(text)
    }
}

/// Cancels the detached worker if its caller stops awaiting the result.
struct CancelOnDrop(Option<Arc<GeminiBatchASR>>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if let Some(provider) = self.0.take() {
            tokio::spawn(async move { provider.cancel().await });
        }
    }
}
