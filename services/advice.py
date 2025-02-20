import asyncio
from typing import List
from config.client import openai_client as client
import json

class AdviceSummarizerService:
    @staticmethod
    async def summarize(text: dict) -> List[str]:
        prompt = """
            You are given a JSON object with pronunciation assessment details.
            Your task is to generate a CONCISE LIST OF UP TO 3 PIECES OF ADVICE
            to help the speaker improve their pronunciation.
            Focus on phoneme errors, word stress issues, and intonation mistakes.
            Extract and summarize the most important improvement points in a
            concise format using natural and encouraging language.            
            Example output:
            [
                "Work on pronouncing the 'th' sound in 'this' and 'that'.",
                "Remember to stress the first syllable in 'banana'.",
                "Try to make your voice go up at the end of questions."
            ]
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(text)},
            ],
            temperature=0.5,
        )
        try:
            parsed_response = json.loads(response.choices[0].message.content)
            if isinstance(parsed_response, list):  # Ensure it's a list
                return parsed_response
            else:
                raise ValueError("Response is not a valid list.")
        except json.JSONDecodeError:
            raise ValueError("Failed to parse JSON response.")

async def main():
    text = {
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
                "expectedIntonationType": "Rising",
                "errorDescription": "The speaker should use a rising intonation pattern",
                "improvementAdvice": "Try to emphasize the key points with a rising tone",
                "errorStartIndex": 0,
                "errorEndIndex": 50,
            },
        },
        "score": {
            "overall": 3.0,
            "fluencyCoherence": 3.0,
            "lexicalResource": 3.0,
            "grammaticalRangeAccuracy": 3.0,
            "pronunciation": 3.0,
        },
    }
    result = await AdviceSummarizerService.summarize(text)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())