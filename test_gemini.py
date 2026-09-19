import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=api_key,
    temperature=0.1
)

response = llm.invoke(
    "Explain the ABCD rule for melanoma in 4 short points."
)

print("\nGemini response:\n")
print(response.content)