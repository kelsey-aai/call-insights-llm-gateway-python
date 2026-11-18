import os
import sys

import assemblyai as aai

api_key = os.environ.get("ASSEMBLYAI_API_KEY")
if not api_key:
    sys.exit("Set ASSEMBLYAI_API_KEY before running this script.")
aai.settings.api_key = api_key

AUDIO_URL = "https://storage.googleapis.com/aai-web-samples/Custom-Home-Builder.mp3"

config = aai.TranscriptionConfig(
    speech_models=["assemblyai-stt-3-7"],
    speaker_labels=True,
    redact_pii=True,
    redact_pii_sub=aai.PIISubstitutionPolicy.entity_name,
    redact_pii_policies=[aai.PIIRedactionPolicy.credit_card_number],
    entity_detection=True,
    sentiment_analysis=True,
    iab_categories=True,
    auto_highlights=True,
)

transcript = aai.Transcriber(config=config).transcribe(AUDIO_URL)

if transcript.status == aai.TranscriptStatus.error:
    sys.exit(f"Transcription failed: {transcript.error}")

for entity in transcript.entities:
    print(f"{entity.entity_type.value}: {entity.text}")

negative = [
    s for s in transcript.sentiment_analysis
    if s.sentiment == aai.SentimentType.negative
]
print(f"{len(negative)} negative sentences")

for label, relevance in transcript.iab_categories.summary.items():
    print(f"{label} ({relevance:.2f})")

for phrase in transcript.auto_highlights.results[:5]:
    print(f"{phrase.text} (mentioned {phrase.count}x)")
