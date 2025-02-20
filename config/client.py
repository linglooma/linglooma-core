from openai import OpenAI
from config import settings

api_key = settings.OPENAI_API_KEY

openai_client = OpenAI(api_key=api_key)
