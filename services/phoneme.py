"""
TLDR:
1. Transcribe Audio: Use a model like Whisper to convert speech to text. Mispronounced words will be transcribed incorrectly.
2. Correct Words: Use a model like GPT-4 to predict the intended words.
3. Convert to IPA: Turn both the transcribed and intended words into IPA.
4. Detect Errors: Compare the IPA strings to spot pronunciation mistakes.
5. Enhance Output: Refine the response with extra data (e.g., Language Confidence API).
"""

from pydantic import Json
from config.client import openai_client as client
import json
import eng_to_ipa as ipa


class PronunciationEvaluationService:
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

    @staticmethod
    def predict_intended_word(actual_text: str, actual_ipa: str) -> str:
        """Predicts the intended words based on phonetic transcription and logical context."""
        response = client.chat.completions.create(
            model="gpt-4o",
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
            temperature=0,
        )
        return response.choices[0].message.content.strip()

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

    @staticmethod
    def compare_phonemes(
        actual_word: str, expected_word: str, actual_ipa: str, expected_ipa: str
    ) -> Json:
        """
        Enhanced comparison of IPA strings with better prompting and validation.

        Args:
            actual_word: The word or phrase being analyzed
            actual_ipa: The IPA string of the actual pronunciation
            expected_ipa: The IPA string of the expected pronunciation

        Returns:
            Dictionary containing error analysis in the specified schema
        """
        prompt = PronunciationEvaluationService.generate_comparison_prompt(
            actual_word, expected_word, actual_ipa, expected_ipa
        )
        result = {}
        result["actualPhoneticTranscription"] = actual_ipa
        result["expectedPhoneticTranscription"] = expected_ipa
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": PronunciationEvaluationService.COMPARISON_PROMPT,
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        phonemeErrorDetails = json.loads(response.choices[0].message.content)
        result["phonemeErrorDetails"] = phonemeErrorDetails
        return result

    @staticmethod
    async def pronunciation_assessment(actual_text: str) -> Json:
        actualPhoneticTranscription = ipa.convert(actual_text)
        expected_text = PronunciationEvaluationService.predict_intended_word(
            actual_text, actualPhoneticTranscription
        )
        expectedPhoneticTranscription = ipa.convert(expected_text)
        return PronunciationEvaluationService.compare_phonemes(
            actual_text,
            expected_text,
            actualPhoneticTranscription,
            expectedPhoneticTranscription,
        )
