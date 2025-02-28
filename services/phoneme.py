from typing import Iterable, Literal, Optional
from utils.logging import log_execution_time as an_yeu_lananh
import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
import json
import eng_to_ipa as ipa
from config.settings import OPENAI_API_KEY
from utils.phoneme import update_transcription_error_indices

class PhonemeErrorDetail(BaseModel):
    transcribedWord: str
    expectedWord: str
    expectedPronunciation: str
    actualPronunciation: str
    errorType: Literal["substitution", "omission"]
    errorStartIndexWord: int
    errorStartIndexTranscription: Optional[int]
    errorEndIndexTranscription: Optional[int]
    errorEndIndexWord: int
    substituted: str
    errorDescription: str = Field(..., max_length=150)
    improvementAdvice: str = Field(..., max_length=150)

class PronunciationAnalysisResponse(BaseModel):
    actualPhoneticTranscription: str
    expectedPhoneticTranscription: str
    phonemeErrorDetails: list[PhonemeErrorDetail]

class PronunciationEvaluationService:
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    instructor_client = instructor.from_openai(client, mode=instructor.Mode.JSON)
    def __init__(self):
        pass    
    PREDICTION_PROMPT = (
        "You are an expert in phonetic transcription and pronunciation analysis. "
        "Your task is to predict the intended words based on the given actual text "
        "and corresponding phonetic transcription, accounting for pronunciation errors, mispronunciations, and contextual meaning. "
        "Your goal is to provide the most accurate intended transcription based on phonetic similarities and logical sentence structure.\n\n"
        "### Guidelines:\n"
        "- **Prioritize vowel shifts** (e.g., /ɛr/ → /ɪə/ in 'rarely' vs. 'really'). \n"
        "- **Correct common consonant substitutions** (e.g., /θ/ → /s/ in 'think' vs. 'sink').\n"
        "- **Identify and correct phonetic omissions** (e.g., /ˈæŋɡɚ/ → 'anger' instead of 'angry').\n"
        "- **Correct phonetic additions** (e.g., /ˈʌv/ → 'of' instead of 'have').\n"
        "- **Correct words that are phonetically close** but mispronounced (e.g., /ˈrɛrli/ → 'really' instead of 'rarely').\n"
        "- If a word in IPA suggests a common speech error, infer the correct intended word.\n"
        "- Ensure the sentence remains grammatically and contextually meaningful.\n"
        "- Return ONLY the corrected sentence without explanations or additional text.\n\n"
        "- Remember to consider the logical context of the sentence.\n\n"
        "### Examples:\n"
        "**Input:**\n"
        "Actual Text: 'I rarely like reading English. I find it youthful.'\n"
        "Actual IPA: aɪ ˈrɛrli laɪk ˈrɛdɪŋ ˈɪŋlɪʃ. aɪ faɪnd ɪt ˈjuθfəl.\n"
        "**Output:** I really like reading English. I find it useful.\n\n"
        "**Input:**\n"
        "Actual Text: 'I am very anger.'\n"
        "Actual IPA: aɪ æm ˈvɛri ˈæŋɡɚ.\n"
        "**Output:** I am very angry.\n\n"
        "**Input:**\n"
        "Actual Text: 'She bought a three pair of shoes.'\n"
        "Actual IPA: ʃi bɔt ə θri pɛr ʌv ʃuz.\n"
        "**Output:** She bought three pairs of shoes.\n"
    )

    COMPARISON_PROMPT = """
        You are a phonetic analysis expert. Compare the actual and expected IPA pronunciations to identify specific pronunciation errors.
        Input Format:
        - actual_word: The word/phrase being analyzed
        - expected_word: The expected word/phrase
        - actual_pronunciation: IPA string of actual pronunciation
        - expected_pronunciation: IPA string of expected pronunciation

        Required Output Schema:
        [{
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
        },{
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

        Guidelines:
        1. Focus on the most significant error if multiple exist
        2. error_start and error_end must be precise word positions
        3. **Provide only ONE sentence for improvement_advice**. Do NOT write multiple sentences or explanations.
        4. **DO NOT include additional context** beyond the structured response format.
        5. Description should use IPA notation with forward slashes
        Analyze the following pronunciations, focusing on the most important error:

        EXAMPLE:
        ```
        [{
            "transcribedWord": "rarely",
            "expectedWord": "really",
            "expectedPronunciation": "/ˈrɪli/",
            "actualPronunciation": "/ˈrɛrli/",
            "errorType": "substitution",
            "errorStartIndex": 1,
            "errorEndIndex": 2,
            "substituted": "ɛr",
            "errorDescription": "The /ɪ/ sound was substituted with /ɛr/.",
            "improvementAdvice": "Practice the /ɪ/ sound by relaxing the tongue and slightly raising it towards the front of the mouth."
        },
        {
            "transcribedWord": "youthful",
            "expectedWord": "useful",
            "expectedPronunciation": "/ˈjusfəl/",
            "actualPronunciation": "/ˈjuθfəl/",
            "errorType": "substitution",
            "errorStartIndex": 3,
            "errorEndIndex": 4,
            "substituted": "θ",
            "errorDescription": "The /s/ sound was substituted with /θ/.",
            "improvementAdvice": "Focus on placing the tip of the tongue close to the roof of the mouth just behind the teeth to create the /s/ sound."
        }]
        ```
    """

    @an_yeu_lananh
    @staticmethod
    async def predict_intended_word(actual_text: str, actual_ipa: str) -> str:
        """Predicts the intended words based on phonetic transcription and logical context."""
        response = await PronunciationEvaluationService.instructor_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": PronunciationEvaluationService.PREDICTION_PROMPT,
                },
                {
                    "role": "user",
                    "content": (
                        f"Actual Text: {actual_text}\n"
                        f"Actual IPA: {actual_ipa}\n"
                        "Predict the most likely intended sentence based on phonetics and context."
                        "REMEMBER TO BASED ON LOGICAL CONTEXT."
                    ),
                },
            ],
            response_model=str,
            temperature=0,
        )   
        return response
    
    
    @staticmethod
    def generate_comparison_prompt(
        actual_word: str, expect_word: str, actual_ipa: str, expected_ipa: str
    ) -> str:
        """Generate a structured prompt for the comparison."""
        return f"""Input:
        actual_word: {actual_word}
        expected_word: {expect_word}
        actual_pronunciation: {actual_ipa}
        expected_pronunciation: {expected_ipa}

        IDENTIFY ALL PRONUNCIATION ERROR AND PROVIDE ANALYSIS FOLLOWING THE SCHEMA EXACTLY.
        """

    @an_yeu_lananh
    @staticmethod
    async def compare_phonemes(actual_word: str, expected_word: str, actual_ipa: str, expected_ipa: str) -> list[PhonemeErrorDetail]:
        prompt = PronunciationEvaluationService.generate_comparison_prompt(
            actual_word, expected_word, actual_ipa, expected_ipa
        )
        result = {}
        result["actualPhoneticTranscription"] = actual_ipa
        result["expectedPhoneticTranscription"] = expected_ipa
        
        response = await PronunciationEvaluationService.instructor_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": PronunciationEvaluationService.COMPARISON_PROMPT,
                },
                {"role": "user", "content": prompt},
            ],
            response_model= Iterable[PhonemeErrorDetail],
            temperature=0,
        )
        return response


    @an_yeu_lananh
    @staticmethod
    async def pronunciation_assessment(actual_text: str) -> PronunciationAnalysisResponse:
        """Analyzes pronunciation by detecting errors in phonetic transcription."""
        actualPhoneticTranscription = ipa.convert(actual_text)
        expected_text = await PronunciationEvaluationService.predict_intended_word(actual_text, actualPhoneticTranscription)
        expectedPhoneticTranscription = ipa.convert(expected_text)

        phoneme_errors = await PronunciationEvaluationService.compare_phonemes(
            actual_text, expected_text, actualPhoneticTranscription, expectedPhoneticTranscription
        )
        phoneme_errors = update_transcription_error_indices(actual_text, phoneme_errors)

        return PronunciationAnalysisResponse(
            actualPhoneticTranscription=actualPhoneticTranscription,
            expectedPhoneticTranscription=expectedPhoneticTranscription,
            phonemeErrorDetails=phoneme_errors
        )


if __name__ == "__main__":
    actual_text = "I rarely like reading English. I find it youthful."
    async def main():
        result = await PronunciationEvaluationService.pronunciation_assessment(actual_text)
        print(json.dumps(result.model_dump(), indent = 4, ensure_ascii= True))
    import asyncio
    asyncio.run(main())