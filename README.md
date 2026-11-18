# Extract call insights with AssemblyAI-STT-3.7 and LLM Gateway (Python)

A small call analytics pipeline: transcribe a recorded call with [AssemblyAI-STT-3.7](https://www.assemblyai.com/docs/pre-recorded-audio/select-the-speech-model) (`assemblyai-stt-3-7`) and speaker diarization, redact PII during transcription, add entity detection and sentiment analysis, then send the redacted, speaker-labeled transcript to [LLM Gateway](https://www.assemblyai.com/docs/llm-gateway/quickstart) for a structured JSON record you can write to a CRM.

Companion repo for the tutorial [Extract phone call insights with LLMs in Python](https://www.assemblyai.com/blog/extract-call-insights-llms-python).

## What's in the repo

| File | What it does |
|------|--------------|
| `transcribe.py` | Transcribes the sample call with `assemblyai-stt-3-7` and `speaker_labels`, then prints a speaker-labeled transcript |
| `redact_pii.py` | PII redaction in the transcript text plus a redacted copy of the audio (PII replaced with silence) |
| `speech_understanding.py` | Entity detection, sentiment analysis, topic detection and key phrases on the same request |
| `pipeline.py` | The full pipeline: transcription + redaction + Speech Understanding, then LLM Gateway. Default output is schema-constrained JSON from `claude-haiku-5-5`; `--text` prints free-text insights from `qwen3.5-4b-32k-fast` |
| `webhook_server.py` | Production pattern: submit with a webhook, a Flask receiver that validates and enqueues, and a worker that runs the LLM step |

## Quick start

Requires Python 3.10 or later and an AssemblyAI API key.

```bash
git clone https://github.com/kelsey-aai/call-insights-llm-gateway-python
cd call-insights-llm-gateway-python

python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Configure your key:

```bash
export ASSEMBLYAI_API_KEY=<your-key>

# or copy the template and load it:
cp .env.example .env
set -a; source .env; set +a
```

If you use the template, edit `.env` and set `ASSEMBLYAI_API_KEY` before loading it. Every script exits with `Set ASSEMBLYAI_API_KEY before running this script.` if the key is missing.

## Run

```bash
python transcribe.py
python redact_pii.py
python speech_understanding.py

python pipeline.py                          # structured JSON for the sample call
python pipeline.py path/or/url/to/call.mp3  # your own recording
python pipeline.py --text                   # free-text insights instead of JSON
```

Webhooks (needs a public URL, for example through a tunnel):

```bash
python webhook_server.py serve
python webhook_server.py submit <AUDIO_URL> https://<your-public-host>/assemblyai-webhook
```

AssemblyAI sends `X-Webhook-Secret: $WEBHOOK_SECRET` back with each callback, and the receiver rejects requests without it.

## Notes

- `assemblyai-stt-3-7` covers 32 languages. Add `"universal-2"` to `speech_models` to fall back for the rest of the 99 languages.
- `qwen3.5-4b-32k-fast` accepts `max_tokens`, `temperature` and `stream` only, so the structured step uses a model that supports `response_format` (`claude-haiku-5-5`). Check the [Models page](https://www.assemblyai.com/docs/llm-gateway/available-models) before you swap models.
- For EU data residency, use `https://api.eu.assemblyai.com` for transcription (`aai.settings.base_url`) and `https://llm-gateway.eu.assemblyai.com/v1/chat/completions` for LLM Gateway. Only Claude and Gemini models are served from the EU endpoint.
- Prices are on [assemblyai.com/pricing](https://www.assemblyai.com/pricing).

## License

MIT
