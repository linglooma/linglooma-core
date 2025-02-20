import asyncio
import logging
from pydantic import Json
import json
from copy import deepcopy
from services.advice import AdviceSummarizerService
from services.phoneme import PronunciationEvaluationService
from services.transcribe import AudioTranscriber
from services.grading import grade_ielts_response
from services.wordstress import (
    WordstressEvaluationService,
)
from services.innotation import InnotationEvaluationService


class SpeakingEvaluationService:
    def __init__(self):
        pass

    @staticmethod
    async def evaluate(audio_path: str) -> Json[any]:
        logging.basicConfig(level=logging.INFO)
        logging.info("Starting speech evaluation")
        
        speaking_evaluation = {}
        try:
            actual_text, score = await asyncio.gather(
                AudioTranscriber.transcribe(audio_path), grade_ielts_response(audio_path)
            )
            logging.info("Transcription and scoring completed")
            logging.info(f"Transcription: {actual_text}")
            logging.info(f"Score: {score}")
            speaking_evaluation["speechTranscription"] = actual_text
            
            (
                pronunciationAssessment,
                wordstressAssessment,
                InnotationAssessment,
            ) = await asyncio.gather(
                PronunciationEvaluationService.pronunciation_assessment(actual_text),
                WordstressEvaluationService.analyze_pronunciation(actual_text, audio_path),
                InnotationEvaluationService.process_audio(actual_text, audio_path),
            )
            
            logging.info("Pronunciation, word stress, and intonation assessments completed")
            logging.info(f"Pronunciation: {pronunciationAssessment}")
            logging.info(f"Word stress: {wordstressAssessment}")
            logging.info(f"Intonation: {InnotationAssessment}")
            
            speaking_evaluation["pronunciationAssessment"] = pronunciationAssessment
            speaking_evaluation["pronunciationAssessment"]["wordStressErrorDetails"] = (
                wordstressAssessment
            )
            speaking_evaluation["pronunciationAssessment"]["intonationErrorDetails"] = (
                InnotationAssessment
            )
            speaking_evaluation["score"] = score
            
            speaking_evaluation["overallAdvices"] = await AdviceSummarizerService.summarize(
                deepcopy(speaking_evaluation)
            )
            logging.info("Evaluation completed successfully")
            return speaking_evaluation
        except Exception as e:
            logging.error(f"Error in speech evaluation: {e}")
            return {}

if __name__ == "__main__":
    audio_path = (
        "/home/xuananle/Documents/Linglooma/Linglooma-core/resources/audio/part2-2.mp3"
    )

    async def main():
        result = await SpeakingEvaluationService.evaluate(audio_path)
        print(json.dumps(result, indent=4, ensure_ascii=False))

    asyncio.run(main())
