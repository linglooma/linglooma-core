import asyncio
import json
import base64
import instructor
from pydantic import Json
import logging
from openai import AsyncOpenAI, BaseModel, OpenAI
from config.client import groq_client
from pydantic import Json
from utils.logging import log_execution_time

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Grading(BaseModel):
    overall: float
    fluencyCoherence: float
    lexicalResource: float
    grammaticalRangeAccuracy: float
    pronunciation: float


class IELTSGradingService:
    IELTS_GRADING_PROMPT = """
            You are an expert IELTS examiner specializing in evaluating IELTS speaking responses. 
            Your task is to assess a given IELTS speaking response based on key criteria and provide a band score. 
            You will be given a JSON object containing the transcribed response, 
            pronunciation assessment details, and other error-related information. 
            Your response should follow the IELTS Speaking band score descriptors, providing an accurate and justified band score for each criterion.
            Scoring Criteria

            Evaluate the response based on the following four official IELTS Speaking criteria:
                Fluency & Coherence (FC) – How smoothly the speaker talks and whether their ideas are logically connected.
                Lexical Resource (LR) – The variety, appropriateness, and accuracy of vocabulary used.
                Grammatical Range & Accuracy (GRA) – The complexity and correctness of sentence structures.
                Pronunciation (P) – The clarity of pronunciation, stress, and intonation.            
            Example Output:
            "overall": 5.5, // ROUND TO THE NEAREST 0.5
            "fluencyCoherence": 4.5,
            "lexicalResource": 5.0,
            "grammaticalRangeAccuracy": 6.0,
            "pronunciation": 6.0,
            """

    @staticmethod
    @log_execution_time
    async def grading(assessment: Json) -> Grading:
        client = instructor.from_groq(groq_client, mode=instructor.Mode.JSON)
        grading = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": IELTSGradingService.IELTS_GRADING_PROMPT},
                {"role": "user", "content": json.dumps(assessment)},
            ],
            temperature=0.5,
            response_model=Grading,
        )
        return grading


async def main():
    assessment = {
        "speechTranscription": "I rarely like reading English. I find it youthful.",
        "pronunciationAssessment": {
            "actualPhoneticTranscription": "aɪ ˈrɛrli laɪk ˈrɛdɪŋ ˈɪŋlɪʃ. aɪ faɪnd ɪt ˈjuθfəl.",
            "expectedPhoneticTranscription": "aɪ ˈrɪli laɪk ˈrɛdɪŋ ˈɪŋlɪʃ. aɪ faɪnd ɪt ˈjusfəl.",
            "phonemeErrorDetails": [
                {
                    "transcribedWord": "rarely",
                    "expectedWord": "really",
                    "expectedPronunciation": "/ˈrɪli/",
                    "actualPronunciation": "/ˈrɛrli/",
                    "errorType": "substitution",
                    "errorStartIndex": 1,
                    "errorEndIndex": 2,
                    "substituted": "ɛr",
                    "errorDescription": "The /ɪ/ sound was substituted with /ɛr/.",
                    "improvementAdvice": "Practice the /ɪ/ sound by relaxing the tongue and slightly raising it towards the front of the mouth.",
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
                    "improvementAdvice": "Focus on placing the tip of the tongue close to the roof of the mouth just behind the teeth to create the /s/ sound.",
                },
            ],
            "wordStressErrorDetails": [
                {
                    "word": "I",
                    "syllableBreakdown": ["I"],
                    "errorType": "None",
                    "actualStressedSyllableIndex": 0,
                    "expectedStressedSyllableIndex": 0,
                    "errorDescription": "",
                    "improvementAdvice": "Good Job, Keep Practicing!",
                    "errorStartIndex": -1,
                    "errorEndIndex": -1,
                },
                {
                    "word": "rarely",
                    "syllableBreakdown": ["rare", "ly"],
                    "errorType": "None",
                    "actualStressedSyllableIndex": 0,
                    "expectedStressedSyllableIndex": 0,
                    "errorDescription": "",
                    "improvementAdvice": "Good Job, Keep Practicing!",
                    "errorStartIndex": -1,
                    "errorEndIndex": -1,
                },
                {
                    "word": "like",
                    "syllableBreakdown": ["like"],
                    "errorType": "None",
                    "actualStressedSyllableIndex": 0,
                    "expectedStressedSyllableIndex": 0,
                    "errorDescription": "",
                    "improvementAdvice": "Good Job, Keep Practicing!",
                    "errorStartIndex": -1,
                    "errorEndIndex": -1,
                },
                {
                    "word": "reading",
                    "syllableBreakdown": ["read", "ing"],
                    "errorType": "None",
                    "actualStressedSyllableIndex": 0,
                    "expectedStressedSyllableIndex": 0,
                    "errorDescription": "",
                    "improvementAdvice": "Good Job, Keep Practicing!",
                    "errorStartIndex": -1,
                    "errorEndIndex": -1,
                },
                {
                    "word": "English",
                    "syllableBreakdown": ["Eng", "lish"],
                    "errorType": "None",
                    "actualStressedSyllableIndex": 0,
                    "expectedStressedSyllableIndex": 0,
                    "errorDescription": "",
                    "improvementAdvice": "Good Job, Keep Practicing!",
                    "errorStartIndex": -1,
                    "errorEndIndex": -1,
                },
                {
                    "word": "I",
                    "syllableBreakdown": ["I"],
                    "errorType": "None",
                    "actualStressedSyllableIndex": 0,
                    "expectedStressedSyllableIndex": 0,
                    "errorDescription": "",
                    "improvementAdvice": "Good Job, Keep Practicing!",
                    "errorStartIndex": -1,
                    "errorEndIndex": -1,
                },
                {
                    "word": "find",
                    "syllableBreakdown": ["find"],
                    "errorType": "None",
                    "actualStressedSyllableIndex": 0,
                    "expectedStressedSyllableIndex": 0,
                    "errorDescription": "",
                    "improvementAdvice": "Good Job, Keep Practicing!",
                    "errorStartIndex": -1,
                    "errorEndIndex": -1,
                },
                {
                    "word": "it",
                    "syllableBreakdown": ["it"],
                    "errorType": "None",
                    "actualStressedSyllableIndex": 0,
                    "expectedStressedSyllableIndex": 0,
                    "errorDescription": "",
                    "improvementAdvice": "Good Job, Keep Practicing!",
                    "errorStartIndex": -1,
                    "errorEndIndex": -1,
                },
                {
                    "word": "youthful",
                    "syllableBreakdown": ["youth", "ful"],
                    "errorType": "None",
                    "actualStressedSyllableIndex": 0,
                    "expectedStressedSyllableIndex": 0,
                    "errorDescription": "",
                    "improvementAdvice": "Good Job, Keep Practicing!",
                    "errorStartIndex": -1,
                    "errorEndIndex": -1,
                },
            ],
            "intonationErrorDetails": {
                "clauseText": "I rarely like reading English. I find it youthful.",
                "actualIntonationType": "Unclear",
                "expectedIntonationType": "Falling",
                "errorDescription": "The statement 'I rarely like reading English' is declarative but uses an unclear intonation, which can confuse the listener about the speaker's certainty. The detected mean pitch and pitch slope suggest a lack of definitive falling intonation.",
                "improvementAdvice": "Focus on lowering your pitch towards the end of the sentence to convey certainty, especially on the word 'English'.",
                "errorStartIndex": 0,
                "errorEndIndex": 30,
            },
        },
        "score": {
            "overall": 3.0,
            "fluencyCoherence": 3.0,
            "lexicalResource": 3.0,
            "grammaticalRangeAccuracy": 3.0,
            "pronunciation": 3.0,
        },
        "overallAdvices": [
            "Practice the 'i' sound in 'really' by relaxing your tongue and slightly raising it towards the front of your mouth.",
            "Focus on making the 's' sound in 'useful' by placing the tip of your tongue close to the roof of your mouth just behind your teeth.",
            "Try to lower your pitch at the end of sentences like 'I rarely like reading English' to convey certainty.",
        ],
    }
    grading = await IELTSGradingService.grading(assessment)
    print(grading.model_dump())


if __name__ == "__main__":
    asyncio.run(main())
