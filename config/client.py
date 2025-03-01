from groq import AsyncGroq
from openai import AsyncOpenAI
from config import settings

openai_api_key = settings.OPENAI_API_KEY

# Trick lord as fuck
# openai_client = AsyncOpenAI(
#     base_url="https://api.fireworks.ai/inference/v1", api_key=settings.FIREWORKS_API_KEY
# )
openai_client = AsyncOpenAI(
    base_url="https://api.fireworks.ai/inference/v1", api_key=settings.FIREWORKS_API_KEY
)
groq_client = AsyncGroq(api_key=settings.GROQ_API_KEY)
