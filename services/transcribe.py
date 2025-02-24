import asyncio
import os
from fireworks.client.audio import AudioInference
from openai import AsyncOpenAI, BaseModel
import logging
from config.settings import FIREWORKS_API_KEY, OPENAI_API_KEY
from utils.logging import log_execution_time
from typing import List


class WordTimeStamp(BaseModel):
    word: str
    start: float
    end: float


class WordTimeStampList(BaseModel):
    words: List[WordTimeStamp]


class AudioTranscribe(BaseModel):
    transcription: str
    word_timestamps: WordTimeStampList


class AudioProcessor:
    client = AudioInference(
        model="whisper-v3",
        base_url="https://audio-prod.us-virginia-1.direct.fireworks.ai",
        api_key=FIREWORKS_API_KEY,
        vad_model="whisperx-pyannet",
        alignment_model="tdnn_ffn",
    )

    @staticmethod
    @log_execution_time
    async def transcribe(file_path: str) -> AudioTranscribe:
        openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        try:
            with open(file_path, "rb") as audio_file:
                response = await openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    timestamp_granularities=["word"],
                    response_format="verbose_json",
                )

            transcript_text = response.text
            word_timestamps = [
                WordTimeStamp(
                    word=word_info.model_dump().get("word", ""),
                    start=round(word_info.model_dump().get("start", 0.0), 3),
                    end=round(word_info.model_dump().get("end", 0.0), 3),
                )
                for word_info in response.words
            ]

            return AudioTranscribe(
                transcription=transcript_text,
                word_timestamps=WordTimeStampList(words=word_timestamps),
            )
        except Exception as e:
            logging.error(f"Error during transcription: {e}")
            raise RuntimeError(f"Error during transcription: {e}")


async def main():
    result = await AudioProcessor.transcribe(
        "/home/xuananle/Documents/Linglooma/Linglooma-core/resources/audio/part2-2.mp3"
    )
    print(result.model_dump_json(indent=4))


if __name__ == "__main__":
    asyncio.run(main())
