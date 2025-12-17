# app/llm.py
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",  # or "gemini-2.0-flash-001" if available
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.2,
    convert_system_message_to_human=True,
)