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
    redact_pii_policies=[
        aai.PIIRedactionPolicy.credit_card_number,
        aai.PIIRedactionPolicy.credit_card_cvv,
        aai.PIIRedactionPolicy.credit_card_expiration,
        aai.PIIRedactionPolicy.banking_information,
        aai.PIIRedactionPolicy.account_number,
        aai.PIIRedactionPolicy.us_social_security_number,
        aai.PIIRedactionPolicy.phone_number,
        aai.PIIRedactionPolicy.email_address,
        aai.PIIRedactionPolicy.location_address,
    ],
    redact_pii_audio=True,
    redact_pii_audio_options={"override_audio_redaction_method": "silence"},
)

transcript = aai.Transcriber(config=config).transcribe(AUDIO_URL)

if transcript.status == aai.TranscriptStatus.error:
    sys.exit(f"Transcription failed: {transcript.error}")

print(transcript.text[:500])

# The redacted recording is a separate file. Download it within 24 hours.
print("Redacted audio:", transcript.get_redacted_audio_url())
