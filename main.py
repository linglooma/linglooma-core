import asyncio
from concurrent import futures
import grpc
from google.protobuf.json_format import ParseDict
import uuid
from grpc_service.speaking_pb2 import SpeakingAssessment, SpeakingAssessmentRequest
from grpc_service.speaking_pb2_grpc import (
    SpeakingAssessmentServiceServicer,
    add_SpeakingAssessmentServiceServicer_to_server,
)
import logging
from services.speaking import SpeakingEvaluationService

logging.basicConfig(
    level=logging.INFO,  # Log level (INFO, DEBUG, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("/tmp/linglooma/app.log"), logging.StreamHandler()],
)


class SpeakingAssessmentServiceImpl(SpeakingAssessmentServiceServicer):
    async def AssessSpeaking(self, request: SpeakingAssessmentRequest, context):
        audio_path = f"/tmp/{str(uuid.uuid4())}.mp3"
        with open(audio_path, "wb") as f:
            f.write(request.audio)
        evaluation_result = await SpeakingEvaluationService.evaluate(audio_path)
        evaluation_result = ParseDict(evaluation_result, SpeakingAssessment())
        return evaluation_result


async def serve():
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    add_SpeakingAssessmentServiceServicer_to_server(
        SpeakingAssessmentServiceImpl(), server
    )
    server.add_insecure_port("[::]:50051")
    print("gRPC Server running on port 50051")
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
