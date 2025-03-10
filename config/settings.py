import os
from dotenv import load_dotenv

load_dotenv()

GRPC_HOST = os.getenv("GRPC_HOST", "localhost")
GRPC_PORT = int(os.getenv("GRPC_PORT", 50051))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
