import os
import json

from dotenv import load_dotenv
from groq import Groq
from planner import create_plan

from tools import calculator, get_time, web_search


load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)
with open("memory.json", "r") as f:
    conversation_history = json.load(f)

# -------------------------
# Tool definitions
# -------------------------

time_tool = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get the current date and time.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
}
web_search_tool = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the internet using a natural-language query. "
            "This is the only tool available for web research."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language search query."
                }
            },
            "required": ["query"]
        }
    }
}

calculator_tool = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Performs basic arithmetic operations: addition, subtraction, multiplication, and division.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {
                    "type": "number",
                    "description": "The first number for the arithmetic operation."
                },
                "b": {
                    "type": "number",
                    "description": "The second number for the arithmetic operation."
                },
                "operation": {
                    "type": "string",
                    "enum": [
                        "add",
                        "subtract",
                        "multiply",
                        "divide"
                    ],
                    "description": "The arithmetic operation."
                }
            },
            "required": [
                "a",
                "b",
                "operation"
            ]
        }
    }
}


# -------------------------
# Tool registry
# -------------------------

tool_functions = {
    "calculator": calculator,
    "get_time": get_time,
    "web_search": web_search
}


# -------------------------
# Agent Loop
# -------------------------
searched_queries = set()
def ask_gemini(question: str):

    # STEP 1: Create a plan
    global conversation_history
    plan = create_plan(question,conversation_history)

    print("\n🧠 EXECUTION PLAN")

    for i, task in enumerate(plan, start=1):
        print(f"{i}. {task}")

    searched_queries = set()

    all_results = []

    # STEP 2: Execute each task
    for i, task in enumerate(plan, start=1):

        print(f"\n🚀 EXECUTING TASK {i}: {task}")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI research agent. "
                    "Complete the user's task using the available tools. "

                    "IMPORTANT TOOL RULES: "
                    "The ONLY available tools are calculator, get_time, and web_search. "
                    "NEVER call any other tool. "
                    "NEVER call open_browser, browser, open_url, fetch, or any other tool. "
                    "IMPORTANT: For arithmetic calculations, always use the calculator tool "
                    "instead of calculating the result yourself. "
                    "Use web_search for current or external information."
                    "The web_search tool accepts exactly one argument named query. "
                    "query must be a natural-language search query. "
                    "NEVER pass cursor, id, URL, source, source_id, or result_id to web_search. "

                    "Use web_search when current or external information is needed."
                )
            },
            {
                "role": "user",
                "content": task
            }
        ]

        # Agent loop for this task
        search_count=0
        max_searches=3
        while True:

            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=messages,
                tools=[
                    calculator_tool,
                    time_tool,
                    web_search_tool
                ],
                tool_choice="auto"
            )

            message = response.choices[0].message

            messages.append(message)

            if not message.tool_calls:

                result = message.content

                print("✅ TASK RESULT:")
                print(result)

                all_results.append(
                    f"Task {i}: {task}\nResult: {result}"
                )

                break

            for tool_call in message.tool_calls:

                tool_name = tool_call.function.name

                arguments = json.loads(
                    tool_call.function.arguments
                )

                print("🔧 TOOL REQUESTED")
                print("Tool:", tool_name)
                print("Arguments:", arguments)

                tool = tool_functions.get(tool_name)

                if tool is None:

                    result = "Unknown tool"

                elif tool_name == "calculator":

                    result = tool(
                        arguments["a"],
                        arguments["b"],
                        arguments["operation"]
                    )

                elif tool_name == "web_search":

                    if search_count >= max_searches:
                        result = (
                            "Search limit reached. "
                            "Use the information already collected "
                            "and provide the best answer."
                        )

                    else:
                        query = arguments["query"]

                        if query in searched_queries:
                            result = (
                                "This query has already been searched. "
                                "Use the existing results."
                            )

                        else:
                            searched_queries.add(query)
                            search_count += 1

                            result = tool(query)

                else:

                    result = tool()

                print("🔧 TOOL RESULT:")
                print(result)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result)
                    }
                )

    # STEP 3: Synthesize all results
    print("\n🧠 SYNTHESIZING FINAL ANSWER")

    final_messages = [
         {
            "role": "system",
            "content": (
                "You are a research synthesis agent. "
                "Use the research results provided to produce a clear, "
                "accurate final answer. Do not invent information. "
                "You may use the previous conversation to understand "
                "the context of the user's current question."
            )
        },
        {
            "role": "user",
            "content": (
                f"Original question:\n{question}\n\n"
                f"Research results:\n"
                f"{chr(10).join(all_results)}"
            )
        }
    ]
    final_messages.extend(conversation_history)
    final_messages.append(
        {
            "role":"user",
            "content": (
            f"Original question:\n{question}\n\n"
            f"Research results:\n"
            f"{chr(10).join(all_results)}"
        )})
    final_response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=final_messages,
        temperature=0
    )

    final_answer = final_response.choices[0].message.content

    print("\n🤖 FINAL ANSWER:")
    print(final_answer)
    conversation_history.append(
        {
            "role": "user",
            "content": question
        }
    )

    conversation_history.append(
        {
            "role": "assistant",
            "content": final_answer
        }
    )
    conversation_history = conversation_history[-6:]
    with open("memory.json", "w") as f:
        json.dump(conversation_history, f, indent=2)

    print("\n🧠 CONVERSATION MEMORY:")
    print(conversation_history)

    return final_answer