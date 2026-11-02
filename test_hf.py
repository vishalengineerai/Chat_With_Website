import os
import sys
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key or not groq_api_key.startswith("gsk_"):
    print("\n❌ GROQ_API_KEY is missing or invalid in .env!")
    sys.exit(1)

print(f"✅ Groq API Key detected: {groq_api_key[:6]}...{groq_api_key[-4:]}")
print("Connecting to Groq (openai/gpt-oss-20b)...")

try:
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        groq_api_key=groq_api_key,
        temperature=0.2,
    )

    test_prompt = "In one concise sentence, explain why open-source models are valuable."
    response = llm.invoke(test_prompt)
    print("🎉 SUCCESS! Groq responded:")
    print(f"\"{response.content.strip()}\"\n")
    print("=" * 60)
    print("✅ Groq connection verified! High speed & active model ready.")
    print("=" * 60)

except Exception as e:
    print(f"\n❌ Error invoking Groq: {e}")
    sys.exit(1)
