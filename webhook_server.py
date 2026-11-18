"""Webhook flow for production volume.

    python webhook_server.py submit <AUDIO_URL> <PUBLIC_WEBHOOK_URL>
    python webhook_server.py serve            # listens on :8000/assemblyai-webhook

The receiver only validates and enqueues. A worker does the LLM step.
"""

import os
import queue
import sys
import threading

import assemblyai as aai
from flask import Flask, abort, request

import pipeline

WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "change-me")
app = Flask(__name__)
jobs: "queue.Queue[str]" = queue.Queue()  # swap for SQS, Celery or RQ in production


def submit(audio_url: str, webhook_url: str) -> str:
    config = pipeline.config.set_webhook(
        webhook_url,
        auth_header_name="X-Webhook-Secret",
        auth_header_value=WEBHOOK_SECRET,
    )
    transcript = aai.Transcriber(config=config).submit(audio_url)
    return transcript.id


@app.post("/assemblyai-webhook")
def assemblyai_webhook():
    if request.headers.get("X-Webhook-Secret") != WEBHOOK_SECRET:
        abort(401)
    payload = request.get_json(silent=True) or {}
    if payload.get("status") == "completed" and payload.get("transcript_id"):
        jobs.put(payload["transcript_id"])
    return "", 200


def worker() -> None:
    while True:
        transcript_id = jobs.get()
        transcript = aai.Transcript.get_by_id(transcript_id)
        insights = pipeline.extract_structured(
            pipeline.format_transcript(transcript),
            pipeline.summarize_fields(transcript),
        )
        print(transcript_id, insights)  # write to your CRM or warehouse here
        jobs.task_done()


if __name__ == "__main__":
    api_key = os.environ.get("ASSEMBLYAI_API_KEY")
    if not api_key:
        sys.exit("Set ASSEMBLYAI_API_KEY before running this script.")
    aai.settings.api_key = api_key

    if len(sys.argv) == 4 and sys.argv[1] == "submit":
        print("Submitted transcript", submit(sys.argv[2], sys.argv[3]))
    elif len(sys.argv) == 2 and sys.argv[1] == "serve":
        threading.Thread(target=worker, daemon=True).start()
        app.run(port=8000)
    else:
        sys.exit(__doc__)
