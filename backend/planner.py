import os
import json

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def create_plan(question: str, conversation_history):

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",

        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research planning agent. "

                    "This is an AI Research Agent project. "

                    "Break the user's question into at most "
                    "3 clear and independent research tasks. "

                    "Keep each task concise. "
                    "IMPORTANT: This project is an AI Research Agent. "
                    "Therefore, whenever the user mentions RAG, "
                    "interpret RAG ONLY as Retrieval-Augmented Generation. "
                    "Never create a task about Red-Amber-Green, "
                    "risk status, project management, or other meanings of RAG "
                    "unless the user explicitly asks for that meaning. "
                    
                    "If the question has only one concept, "
                    "return ONE task instead of creating multiple tasks "
                    "for alternative interpretations. "
                    
                    "Do not explore alternative meanings of acronyms "
                    "unless the user explicitly asks for them. "
                    "IMPORTANT CONTEXT RULE: "
        
                    "When the question is about this AI project, "
                    "RAG always means Retrieval-Augmented Generation. "
                      
                    "Do NOT interpret RAG as Red-Amber-Green, "
                    "risk status, healthcare triage, finance, "
                    "or any other meaning unless the user explicitly "
                    "asks about those meanings. "

                    "Use the previous conversation to resolve "
                    "ambiguous references such as it, this, that, "
                    "or the acronym RAG. "

                    "Do not create unnecessary tasks. "

                    "Return ONLY a JSON array of strings."
                )
            },

            {
                "role": "user",
                "content": (
                    f"Previous conversation:\n"
                    f"{conversation_history}\n\n"

                    f"Current question:\n"
                    f"{question}"
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

            content = content.replace(
                "```json",
                ""
            )

            content = content.replace(
                "```",
                ""
            )

            content = content.strip()

        plan = json.loads(content)

        return plan

    except json.JSONDecodeError:

        print(
            "❌ Planner returned invalid JSON"
        )

        return [question]