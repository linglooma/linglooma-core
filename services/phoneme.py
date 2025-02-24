from pydantic import Json, BaseModel, Field
from typing import Literal, List
from config.client import openai_client as client
import json
import eng_to_ipa as ipa
from utils.logging import log_execution_time
from utils.phoneme import update_transcription_error_indices


class PhonemeError(BaseModel):
    transcribedWord: str
    expectedWord: str
    expectedPronunciation: str
    actualPronunciation: str
    errorType: Literal["substitution", "omission"]
    errorStartIndexWord: int
    errorEndIndexWord: int
    substituted: str
    errorDescription: str = Field(..., max_length=150)
    improvementAdvice: str = Field(..., max_length=150)


class PronunciationAnalysisResponse(BaseModel):
    expectedText: str
    phonemeErrors: List[PhonemeError]


class PronunciationEvaluationService:
    UNIFIED_PROMPT = """You are an expert in phonetic transcription and pronunciation analysis. Your task is to analyze a given text and its IPA transcription to:
    1. Predict the intended text based on pronunciation errors
    2. Identify specific pronunciation errors

    ### Guidelines for Text Prediction:
    - Prioritize vowel shifts (e.g., /ɛr/ → /ɪə/ in 'rarely' vs. 'really')
    - Correct common consonant substitutions (e.g., /θ/ → /s/ in 'think' vs. 'sink')
    - Identify and correct phonetic omissions (e.g., /ˈæŋɡɚ/ → 'anger' instead of 'angry')
    - Correct phonetic additions (e.g., /ˈʌv/ → 'of' instead of 'have')
    - Ensure sentences remain grammatically and contextually meaningful
    - **ERROR INDEX START FROM ZERO NOT ONE**

    ### Guidelines for Error Analysis:
    - Focus on significant pronunciation errors
    - Provide precise error positions
    - Use IPA notation in descriptions
    - Keep improvement advice concise and specific
    
    ### Required Output Schema:
    {
        "expectedText": string,  // The corrected text based on pronunciation analysis
        "phonemeErrors": [{
            "transcribedWord": string,
            "expectedWord": string,
            "expectedPronunciation": string,
            "actualPronunciation": string,
            "errorType": "substitution" | "omission",
            "errorStartIndex": number,
            "errorEndIndex": number,
            "substituted": string,
            "errorDescription": string,
            "improvementAdvice": string
        }]
    }

    ### Examples:
    Input:
    Text: "I rarely like reading English. I find it youthful."
    IPA: "aɪ ˈrɛrli laɪk ˈrɛdɪŋ ˈɪŋlɪʃ. aɪ faɪnd ɪt ˈjuθfəl."

    Output:
    {
        "expectedText": "I really like reading English. I find it useful.",
        "phonemeErrors": [
            {
                "transcribedWord": "rarely",
                "expectedWord": "really",
                "expectedPronunciation": "ˈrɪli",
                "actualPronunciation": "ˈrɛrli",
                "errorType": "substitution",
                "errorStartIndex": 2,
                "errorEndIndex": 3,
                "substituted": "ɛr",
                "errorDescription": "/ˈrɛrli/ was pronounced instead of /ˈrɪli/",
                "improvementAdvice": "Replace /ɛr/ with /ɪ/ in 'really'"
            },
            {
                "transcribedWord": "youthful",
                "expectedWord": "useful",
                "expectedPronunciation": "ˈjusfəl",
                "actualPronunciation": "ˈjuθfəl",
                "errorType": "substitution",
                "errorStartIndex": 2,
                "errorEndIndex": 3,
                "substituted": "juθ",
                "errorDescription": "/ˈjuθfəl/ was pronounced instead of /ˈjusfəl/",
                "improvementAdvice": "Replace /θ/ with /s/ in 'useful'"
            }
        ]
    }

    Analyze the following text and provide corrections and error analysis:
    """

    @staticmethod
    @log_execution_time
    async def pronunciation_assessment(actual_text: str) -> Json:
        actual_ipa = ipa.convert(actual_text)
        response = await client.chat.completions.create(
            model="accounts/fireworks/models/deepseek-v3",
            response_format={
                "type": "json_object",
                "schema": PronunciationAnalysisResponse.model_json_schema(),
            },
            messages=[
                {
                    "role": "system",
                    "content": PronunciationEvaluationService.UNIFIED_PROMPT,
                },
                {"role": "user", "content": f"Text: {actual_text}\nIPA: {actual_ipa}"},
            ],
            temperature=0,
        )
        analysis = json.loads(response.choices[0].message.content)
        analysis["phonemeErrors"] = update_transcription_error_indices(
            actual_text, analysis["phonemeErrors"]
        )
        result = {
            "actualPhoneticTranscription": actual_ipa,
            "expectedPhoneticTranscription": ipa.convert(analysis["expectedText"]),
            "phonemeErrorDetails": analysis["phonemeErrors"],
        }

        return result


if __name__ == "__main__":
    actual_text = "I rarely like reading English. I find it youthful."

    async def main():
        result = await PronunciationEvaluationService.pronunciation_assessment(
            actual_text
        )
        print(json.dumps(result, indent=4, ensure_ascii=False))

    import asyncio

    asyncio.run(main())
