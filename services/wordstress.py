import json
from typing import Dict, Any, List, Literal
from config import settings
from services.transcribe import AudioProcessor
from utils.audio import AudioReader
from openai import AsyncOpenAI, BaseModel

from utils.logging import log_execution_time
import json
from typing import List, Literal
from pydantic import BaseModel
from openai import AsyncOpenAI


class PronunciationError(BaseModel):
    word: str
    syllableBreakdown: List[str]
    errorType: Literal[
        "Stress Misplacement",
        "Vowel Reduction",
        "Consonant Substitution",
        "Insertion",
        "Omission",
        "None",
    ]
    actualStressedSyllableIndex: int
    expectedStressedSyllableIndex: int
    errorDescription: str
    improvementAdvice: str
    errorStartIndex: int
    errorEndIndex: int


class PronunciationErrorList(BaseModel):
    errors: List[PronunciationError]


class WordstressEvaluationService:
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    SYSTEM_PROMPT_TEMPLATE = """
    ### ROLE
    You are a phonetics and linguistics expert specializing in pronunciation analysis and stress pattern detection.
    Your task is to analyze speech recordings and provide detailed feedback ONLY for words with pronunciation errors.
    The errors type MUST be one of the following: ["Stress Misplacement", "Vowel Reduction", "Consonant Substitution", "Insertion", "Omission"].

    ### TASKS
    1. Analyze the pronunciation of each word in the recording
    2. ONLY report words that have pronunciation errors
    3. For each error, provide:
       - Word breakdown into syllables
       - Type of error
       - Actual vs expected stress patterns
       - Clear description and improvement advice
    4. Skip words with CORRECTLY STRESSED entirely
    5. DO NOT INCLUDE ```json in the output

    ### OUTPUT FORMAT
    Return a list of error objects ONLY. Each error object should have:
    ```
    {{
        "word": "example",
        "syllableBreakdown": ["ex", "am", "ple"],
        "errorType": "Stress Misplacement",
        "actualStressedSyllableIndex": 2,
        "expectedStressedSyllableIndex": 1,
        "errorDescription": "Concise description of the error",
        "improvementAdvice": "Brief, actionable advice for improvement",
        "errorStartIndex": 20,
        "errorEndIndex": 27
    }}
    ```

    ### REQUIREMENTS
    1. Only include words with actual pronunciation errors
    2. Ensure error descriptions are clear and specific
    3. Provide brief, actionable improvement advice
    4. Include accurate start and end indices for error highlighting
    5. DO NOT include correctly pronounced words in the output
    6. Return a JSON list with **ONLY stress misplacement errors**, following this structure:
    Actual text from audio: "{actual_text}"
    """

    @staticmethod
    @log_execution_time
    async def analyze_pronunciation(
        actual_text: str, audio_path: str
    ) -> List[PronunciationError]:
        audio_content = AudioReader.encode_audio_to_base64(audio_path)

        system_prompt = WordstressEvaluationService.SYSTEM_PROMPT_TEMPLATE.format(
            actual_text=actual_text
        )

        response = await WordstressEvaluationService.client.chat.completions.create(
            model="gpt-4o-mini-audio-preview-2024-12-17",
            modalities=["text"],
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze the audio and return ONLY pronunciation errors in a structured list. Skip correctly pronounced words entirely.",
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {"data": audio_content, "format": "mp3"},
                        },
                    ],
                },
            ],
            temperature=0,
        )

        analysis_result = response.choices[0].message.content
        print(analysis_result)
        errors = json.loads(analysis_result)
        return [PronunciationError(**error) for error in errors]


async def main():
    audio_path = (
        "/home/xuananle/Documents/Linglooma/Linglooma-core/resources/audio/part2-2.mp3"
    )
    actual_text = await AudioProcessor.transcribe(audio_path)
    result = [
        await WordstressEvaluationService.analyze_pronunciation(actual_text, audio_path)
    ]


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
