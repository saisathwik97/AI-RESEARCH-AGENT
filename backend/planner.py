import os
import json

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def create_plan(question: str, conversation_history: list):

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",

        messages=[
            {
                "role": "system",
                "content": (
                        "You are a research planning agent. "
                        "Break the user's question into at most 3 clear, independent research tasks. "
                        "Keep each task concise. "
                        "Return ONLY a JSON array of strings."
                    )
            },
           {
    "role": "user",
    "content": (
        f"Previous conversation:\n{conversation_history}\n\n"
        f"Current question:\n{question}"
    )
}
        ],

        temperature=0
    )

    content = response.choices[0].message.content

    print("🧠 PLAN GENERATED:")
    print(content)

    try:
        content = content.strip()

        if content.startswith("```"):
            content = content.replace("```json", "")
            content = content.replace("```", "")
            content = content.strip()

        plan = json.loads(content)

        return plan

    except json.JSONDecodeError:
        print("❌ Planner returned invalid JSON")
        return [question]
