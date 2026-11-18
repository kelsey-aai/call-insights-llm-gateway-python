"""Call insights pipeline: AssemblyAI-STT-3.7 + PII redaction + Speech
Understanding + LLM Gateway structured output.

Usage:
    python pipeline.py [AUDIO_URL_OR_PATH]          # structured JSON for a CRM
    python pipeline.py --text [AUDIO_URL_OR_PATH]   # free-text insights
"""

import json
import os
import sys
from collections import Counter

import assemblyai as aai
import requests

SAMPLE_AUDIO = "https://storage.googleapis.com/aai-web-samples/Custom-Home-Builder.mp3"

config = aai.TranscriptionConfig(
    speech_models=["assemblyai-stt-3-7"],
    speaker_labels=True,
    redact_pii=True,
    redact_pii_sub=aai.PIISubstitutionPolicy.entity_name,
    redact_pii_policies=[
        aai.PIIRedactionPolicy.credit_card_number,
        aai.PIIRedactionPolicy.credit_card_cvv,
        aai.PIIRedactionPolicy.credit_card_expiration,
        aai.PIIRedactionPolicy.banking_information,
        aai.PIIRedactionPolicy.us_social_security_number,
        aai.PIIRedactionPolicy.phone_number,
        aai.PIIRedactionPolicy.email_address,
    ],
    entity_detection=True,
    sentiment_analysis=True,
)


def format_transcript(transcript: aai.Transcript) -> str:
    return "\n".join(
        f"Speaker {u.speaker}: {u.text}" for u in transcript.utterances
    )


def summarize_fields(transcript: aai.Transcript) -> str:
    """Turn the Speech Understanding output into compact context for the LLM."""
    entities = sorted({f"{e.entity_type.value}: {e.text}" for e in transcript.entities or []})
    sentiment = Counter(s.sentiment.value for s in transcript.sentiment_analysis or [])
    return json.dumps(
        {"entities": entities, "sentence_sentiment_counts": dict(sentiment)},
        indent=2,
    )


GATEWAY_URL = "https://llm-gateway.assemblyai.com/v1/chat/completions"
INSIGHTS_MODEL = "qwen3.5-4b-32k-fast"
EXTRACTION_MODEL = "claude-haiku-5-5"

PROMPT = """ROLE
You are a sales operations analyst reviewing an inbound customer call.

CONTEXT
Below is a speaker-labeled, PII-redacted transcript. Structured fields already
extracted from the audio are provided alongside it. Treat those fields as ground
truth and do not contradict them.

INSTRUCTION
Produce a concise summary, the concrete action items each party committed to,
the customer's primary objection if one was raised, and a recommended next step.
If something was not discussed, say so rather than inferring it.

TRANSCRIPT
{conversation}

STRUCTURED FIELDS
{structured}
"""


def gateway_headers() -> dict:
    return {"authorization": os.environ["ASSEMBLYAI_API_KEY"]}


def analyze(conversation: str, structured: str, model: str = INSIGHTS_MODEL) -> str:
    response = requests.post(
        GATEWAY_URL,
        headers=gateway_headers(),
        json={
            "model": model,
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": PROMPT.format(
                        conversation=conversation, structured=structured
                    ),
                }
            ],
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


CALL_INSIGHTS_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "action_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "enum": ["agent", "customer"]},
                    "task": {"type": "string"},
                    "due": {"type": "string"},
                },
                "required": ["owner", "task", "due"],
                "additionalProperties": False,
            },
        },
        "primary_objection": {"type": "string"},
        "call_outcome": {
            "type": "string",
            "enum": ["closed_won", "follow_up_scheduled", "no_next_step", "lost"],
        },
        "overall_sentiment": {
            "type": "string",
            "enum": ["positive", "neutral", "negative"],
        },
    },
    "required": [
        "summary",
        "action_items",
        "primary_objection",
        "call_outcome",
        "overall_sentiment",
    ],
    "additionalProperties": False,
}

SCHEMA_RULES = (
    "\nReturn JSON only. Use an empty string for primary_objection when the "
    "customer raised none, and for due when no date was agreed."
)


def extract_structured(conversation: str, structured: str) -> dict:
    response = requests.post(
        GATEWAY_URL,
        headers=gateway_headers(),
        json={
            "model": EXTRACTION_MODEL,
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": PROMPT.format(
                        conversation=conversation, structured=structured
                    )
                    + SCHEMA_RULES,
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "call_insights",
                    "strict": True,
                    "schema": CALL_INSIGHTS_SCHEMA,
                },
            },
            "post_processing_steps": [{"type": "json-repair"}],
        },
        timeout=60,
    )
    response.raise_for_status()
    return json.loads(response.json()["choices"][0]["message"]["content"])


def process_call(audio_url: str) -> dict | None:
    try:
        transcript = aai.Transcriber(config=config).transcribe(audio_url)
        if transcript.status == aai.TranscriptStatus.error:
            print(f"Transcription failed: {transcript.error}")
            return None

        conversation = format_transcript(transcript)
        insights = extract_structured(conversation, summarize_fields(transcript))
        insights["transcript_id"] = transcript.id
        return insights

    except aai.AssemblyAIError as e:
        print(f"AssemblyAI API error: {e}")
    except requests.exceptions.RequestException as e:
        print(f"LLM Gateway request failed: {e}")
    except json.JSONDecodeError as e:
        print(f"Model returned unparseable output: {e}")
    return None


def main() -> None:
    api_key = os.environ.get("ASSEMBLYAI_API_KEY")
    if not api_key:
        sys.exit("Set ASSEMBLYAI_API_KEY before running this script.")
    aai.settings.api_key = api_key

    args = [a for a in sys.argv[1:] if a != "--text"]
    audio = args[0] if args else SAMPLE_AUDIO

    if "--text" in sys.argv:
        # Free-text insights from the fast model instead of the JSON contract.
        transcript = aai.Transcriber(config=config).transcribe(audio)
        if transcript.status == aai.TranscriptStatus.error:
            sys.exit(f"Transcription failed: {transcript.error}")
        print(analyze(format_transcript(transcript), summarize_fields(transcript)))
        return

    result = process_call(audio)
    if result is None:
        sys.exit(1)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
