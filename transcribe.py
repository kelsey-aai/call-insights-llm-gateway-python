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
)

transcript = aai.Transcriber(config=config).transcribe(AUDIO_URL)

if transcript.status == aai.TranscriptStatus.error:
    sys.exit(f"Transcription failed: {transcript.error}")

print("Model used:", transcript.speech_model_used)


def format_transcript(transcript: aai.Transcript) -> str:
    return "\n".join(
        f"Speaker {u.speaker}: {u.text}" for u in transcript.utterances
    )


conversation = format_transcript(transcript)
print(conversation[:800])
