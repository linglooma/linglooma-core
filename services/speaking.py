import asyncio
import logging
import json
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
        logging.basicConfig(level=logging.INFO)
        logging.info("Starting speech evaluation")

        speaking_evaluation = {}

        try:
            audio_transcription = await AudioProcessor.transcribe(audio_path)
            logging.info("Transcription completed successfully")
            logging.info(audio_transcription.transcription)
            speaking_evaluation["speechTranscription"] = (
                audio_transcription.transcription
            )
            (
                pronunciationAssessment,
                wordstressAssessment,
                intonationAssessment,
            ) = await asyncio.gather(
                PronunciationEvaluationService.pronunciation_assessment(
                    audio_transcription.transcription
                ),
                WordstressEvaluationService.evaluate_stress(
                    audio_transcription, audio_path
                ),
                InnotationEvaluationService.process_audio(
                    audio_transcription.transcription, audio_path
                ),
            )
            speaking_evaluation["pronunciationAssessment"] = pronunciationAssessment.model_dump()
            speaking_evaluation["pronunciationAssessment"]["wordStressErrorDetails"] = (
                wordstressAssessment
            )
            speaking_evaluation["pronunciationAssessment"]["intonationErrorDetails"] = (
                intonationAssessment
            )
            (score, overallAdvices) = await asyncio.gather(
                IELTSGradingService.grading(speaking_evaluation.copy()),
                AdviceSummarizerService.summarize(speaking_evaluation.copy()),
            )

            speaking_evaluation["overallAdvices"] = overallAdvices
            speaking_evaluation["score"] = score.model_dump()

            logging.info("Evaluation completed successfully")
            print(json.dumps(speaking_evaluation, indent=4, ensure_ascii=False))
            return speaking_evaluation

        except Exception as e:
            logging.error(f"Error in speech evaluation: {e}")
            return speaking_evaluation


if __name__ == "__main__":
    audio_path = "/home/xuananle/Documents/Linglooma/Linglooma-core/resources/audio/recorded_audio.mp3"

    async def main():
        result = await SpeakingEvaluationService.evaluate(audio_path)
        print(json.dumps(result, indent=4, ensure_ascii=False))

    asyncio.run(main())
