import os
import asyncio
from config.client import openai_client


class AudioTranscriber:
    @staticmethod
    async def transcribe(file_path: str) -> str:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        try:
            with open(file_path, "rb") as audio_file:
                response = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    timestamp_granularities=["word"],
                    response_format="verbose_json",
                )
                return response.text
        except Exception as e:
            raise RuntimeError(f"Error during transcription: {e}")
