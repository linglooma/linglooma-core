import json
from typing import Dict, Any
from services.transcribe import AudioTranscriber
from config.client import openai_client as client
from utils.audio import AudioReader


class WordstressEvaluationService:
    SYSTEM_PROMPT_TEMPLATE = """
    ### ROLE
    You are a phonetics and linguistics expert specializing in pronunciation analysis and stress pattern detection.
    Your task is to analyze a given speech recording and provide detailed feedback, including phonetic breakdown, stress patterns, and pronunciation accuracy.
    The errors type MUST be one of the following: ["Stress Misplacement", "Vowel Reduction", "Consonant Substitution", "Insertion", "Omission", or "None"].

    ### TASKS
    1. **Transcribe the Speech**: Convert the spoken audio into text.
    2. **Syllable Division**: Break down each word into its phonetic syllables.
    3. **Stress Identification**: Mark the primary and secondary stress in each word.
    4. **Pronunciation Comparison**: Compare the user's spoken pronunciation with the correct pronunciation and identify errors.
    5. **Provide Feedback**: Suggest improvements and highlight areas for practice. LIMIT THIS TO ONLY ONE SENTENCE.
    6. **If actualStressedSyllableIndex equals expectedStressedSyllableIndex, set:**
       - `errorType` = "None"
       - `improvementAdvice` = "Good Job, Keep Practicing!"
       - `errorStartIndex` = -1
       - `errorEndIndex` = -1
       - `errorDescription` = ""

    ### SCHEMA (Output Format)
    The output should be a **list of objects**, where each object has the following structure:
    ```
    [
        {{
            "word": "example",
            "syllableBreakdown": ["ex", "am", "ple"],
            "errorType": "Substitution",  
            "actualStressedSyllableIndex": 2,
            "expectedStressedSyllableIndex": 1,
            "errorDescription": "The candidate incorrectly stressed the second syllable instead of the first.",
            "improvementAdvice": "Practice by clapping out the syllables and emphasizing the first syllable in 'example'.",
            "errorStartIndex": 20,
            "errorEndIndex": 27
        }}
    ]
    ```
    
    ### REQUIREMENTS
    1. **Include all words**, even if repeated with different stress patterns.
    2. **Clearly mark stress mismatches**.
    3. **Provide pronunciation accuracy statistics** at the end.
    4. **Include confidence levels** for each analysis.
    5. **Identify mispronunciations, dialectal variations, or phoneme substitutions.**
    6. **If actualStressedSyllableIndex equals expectedStressedSyllableIndex, set:**
       - `errorType` = "None"
       - `improvementAdvice` = "Good Job, Keep Practicing!"
       - `errorStartIndex` = -1
       - `errorEndIndex` = -1
       - `errorDescription` = ""

    Actual text from audio: "{actual_text}"
    """

    @staticmethod
    async def analyze_pronunciation(
        actual_text: str, audio_path: str
    ) -> Dict[str, Any]:
        audio_content = AudioReader.encode_audio_to_base64(audio_path)

        system_prompt = WordstressEvaluationService.SYSTEM_PROMPT_TEMPLATE.format(
            actual_text=actual_text
        )

        response = client.chat.completions.create(
            model="gpt-4o-audio-preview",
            modalities=["text"],
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Provide detailed phonetic analysis with stress patterns and accuracy comparison, returning a structured list of objects. Ensure that when actualStressedSyllableIndex equals expectedStressedSyllableIndex, errorType is 'None' and all error fields are cleared.",
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
        return json.loads(analysis_result)


async def main():
    audio_path = (
        "/home/xuananle/Documents/Linglooma/Linglooma-core/resources/audio/part2-2.mp3"
    )
    actual_text = await AudioTranscriber.transcribe(audio_path)
    result = await WordstressEvaluationService.analyze_pronunciation(
        actual_text, audio_path
    )
    print(json.dumps(result, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
