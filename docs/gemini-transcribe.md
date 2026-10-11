# Gemini dedicated transcription

OpenLess supports `gemini-3.5-transcribe` for recorded audio in the Traditional ASR + LLM pipeline. Transcription starts when recording stops. The separate `gemini-3.5-transcribe-live` model is not supported yet.

## Configuration

1. Select Traditional mode.
2. Add a speech recognition channel and choose Google Gemini Transcribe.
3. Enter your Gemini API key in that channel.
4. Use `gemini-3.5-transcribe` as the model.
5. Use `https://generativelanguage.googleapis.com/v1beta` as the Base URL. The complete `https://generativelanguage.googleapis.com/v1beta/interactions` endpoint is also accepted.
6. Click Verify and enable the channel. Configure the language model channel separately if you want polishing.

This ASR channel uses the dedicated transcription API. Selecting this model in the multimodal or language model channel does not configure dedicated ASR.

## Request and verification behavior

OpenLess encodes recorded PCM as a 16 kHz mono WAV, uploads it through the Files API, and passes the returned file URI and `audio/wav` MIME type to the Interactions API. Temporary uploaded files are deleted after the request, including on transcription failure. Language preferences and dictionary terms are sent as transcription hints; transcription uses verbatim mode before any separately configured polishing.

Verify exercises this same ASR path using 500 milliseconds of silence. A completed response without text is a valid result for silence. HTTP failures retain their status for the settings error display; provider response bodies and credentials are not logged.

## Live smoke test

The following example reads `GEMINI_API_KEY` from the environment and uses in-memory channel credentials. It runs the real settings Verify service, then uploads and transcribes the supplied audio with the production ASR client. It makes billable API requests when applicable.

From `openless-all/app`, supply a raw signed 16-bit little-endian, 16 kHz mono PCM file containing speech:

```sh
cargo run -p openless-core --locked --example gemini_transcribe_smoke -- speech.pcm
cargo run -p openless-core --locked --example gemini_transcribe_smoke -- speech.pcm https://generativelanguage.googleapis.com/v1beta/interactions
```

For example, FFmpeg can convert an existing recording with `ffmpeg -i speech.wav -ar 16000 -ac 1 -f s16le speech.pcm`.

Offline regression tests require no API key:

```sh
cargo test -p openless-core --locked --lib asr::gemini
```

## Official references

- [Audio transcription guide](https://ai.google.dev/gemini-api/docs/transcribe)
- [Interactions API reference](https://ai.google.dev/api/interactions-api)
- [Interactions OpenAPI schema](https://ai.google.dev/static/api/interactions.openapi.json)
- [Live transcription guide](https://ai.google.dev/gemini-api/docs/live-api/live-transcribe)
