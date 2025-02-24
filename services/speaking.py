import asyncio
import logging
import json
from copy import deepcopy
from services.advice import AdviceSummarizerService
from services.phoneme import PronunciationEvaluationService
from services.transcribe import AudioProcessor
from services.grading import IELTSGradingService
from services.wordstress import (
    WordstressEvaluationService,
)
from services.innotation import InnotationEvaluationService
from typing import Any, Dict

from utils.logging import log_execution_time


import asyncio
import logging
import json
from copy import deepcopy
from services.advice import AdviceSummarizerService
from services.phoneme import PronunciationEvaluationService
from services.transcribe import AudioProcessor
from services.grading import IELTSGradingService
from services.wordstress import (
    WordstressEvaluationService,
)
from services.innotation import InnotationEvaluationService
from typing import Any, Dict

from utils.logging import log_execution_time


class SpeakingEvaluationService:
    def __init__(self):
        pass

    @staticmethod
    @log_execution_time
    async def evaluate(audio_path: str) -> Dict[str, Any]:
        return {
            "speechTranscription": "I rarely like reading English. I find it youthful.",
            "pronunciationAssessment": {
                "actualPhoneticTranscription": "aɪ ˈrɛrli laɪk ˈrɛdɪŋ ˈɪŋlɪʃ. aɪ faɪnd ɪt ˈjuθfəl.",
                "expectedPhoneticTranscription": "aɪ ˈrɪli laɪk ˈrɛdɪŋ ˈɪŋlɪʃ. aɪ faɪnd ɪt ˈjusfəl.",
                "phonemeErrorDetails": [
                    {
                        "transcribedWord": "rarely",
                        "expectedWord": "really",
                        "expectedPronunciation": "ˈrɪli",
                        "actualPronunciation": "ˈrɛrli",
                        "errorType": "substitution",
                        "errorStartIndexWord": 2,
                        "errorEndIndexWord": 3,
                        "substituted": "ɛr",
                        "errorDescription": "/ˈrɛrli/ was pronounced instead of /ˈrɪli/",
                        "improvementAdvice": "Replace /ɛr/ with /ɪ/ in 'really'",
                        "errorStartIndexTranscription": 2,
                        "errorEndIndexTranscription": 7,
                    },
                    {
                        "transcribedWord": "youthful",
                        "expectedWord": "useful",
                        "expectedPronunciation": "ˈjusfəl",
                        "actualPronunciation": "ˈjuθfəl",
                        "errorType": "substitution",
                        "errorStartIndexWord": 2,
                        "errorEndIndexWord": 3,
                        "substituted": "juθ",
                        "errorDescription": "/ˈjuθfəl/ was pronounced instead of /ˈjusfəl/",
                        "improvementAdvice": "Replace /θ/ with /s/ in 'useful'",
                        "errorStartIndexTranscription": 41,
                        "errorEndIndexTranscription": 48,
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

        logging.basicConfig(level=logging.INFO)
        logging.info("Starting speech evaluation")

        speaking_evaluation = {}

        try:
            actual_text = await asyncio.gather(
                AudioProcessor.transcribe(audio_path),
            )
            actual_text = actual_text[0]

            speaking_evaluation["speechTranscription"] = actual_text

            (
                pronunciationAssessment,
                wordstressAssessment,
                intonationAssessment,
                score,
            ) = await asyncio.gather(
                PronunciationEvaluationService.pronunciation_assessment(actual_text),
                WordstressEvaluationService.analyze_pronunciation(
                    actual_text, audio_path
                ),
                InnotationEvaluationService.process_audio(actual_text, audio_path),
                IELTSGradingService.grade_ielts_response(audio_path),
            )
            print(pronunciationAssessment)
            speaking_evaluation["pronunciationAssessment"] = pronunciationAssessment
            speaking_evaluation["pronunciationAssessment"]["wordStressErrorDetails"] = (
                wordstressAssessment
            )
            speaking_evaluation["pronunciationAssessment"]["intonationErrorDetails"] = (
                intonationAssessment
            )
            speaking_evaluation["score"] = score
            speaking_evaluation[
                "overallAdvices"
            ] = await AdviceSummarizerService.summarize(deepcopy(speaking_evaluation))
            logging.info("Evaluation completed successfully")
            return speaking_evaluation

        except Exception as e:
            logging.error(f"Error in speech evaluation: {e}")
            return speaking_evaluation


if __name__ == "__main__":
    audio_path = (
        "/home/xuananle/Documents/Linglooma/Linglooma-core/resources/audio/part2-2.mp3"
    )

    async def main():
        result = await SpeakingEvaluationService.evaluate(audio_path)
        print(json.dumps(result, indent=4, ensure_ascii=False))

    asyncio.run(main())
