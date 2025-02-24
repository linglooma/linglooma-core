import grpc
import json
from google.protobuf.json_format import MessageToDict
from grpc_service.speaking_pb2 import SpeakingAssessmentRequest
from grpc_service.speaking_pb2_grpc import SpeakingAssessmentServiceStub


def run():
    channel = grpc.insecure_channel("localhost:50051")
    stub = SpeakingAssessmentServiceStub(channel)
    with open(
        "/home/xuananle/Documents/Linglooma/Linglooma-core/resources/audio/part2-1.mp3",
        "rb",
    ) as f:
        audio_data = f.read()
    request = SpeakingAssessmentRequest(audio=audio_data)
    response = stub.AssessSpeaking(request)
    print(
        json.dumps(
            MessageToDict(response, preserving_proto_field_name=True),
            indent=4,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    run()
