import os
import json

from dotenv import load_dotenv
from groq import Groq

from planner import create_plan
from semantic_retriever import retrieve
from tools import calculator, get_time, web_search


load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# -------------------------
# Load conversation memory
# -------------------------

with open("memory.json", "r") as f:
    conversation_history = json.load(f)


# -------------------------
# Tool definitions
# -------------------------
rag_tool = {
    "type": "function",
    "function": {
        "name": "retrieve_documents",
        "description": (
            "Search the project's internal documents using semantic similarity. "
            "Use this tool when the question may be answered from the project's "
            "stored documentation or internal knowledge."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The question or topic to search for in the internal documents."
                }
            },
            "required": ["query"]
        }
    }
}
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
        "description": (
            "Performs basic arithmetic operations: "
            "addition, subtraction, multiplication, and division."
        ),
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
    "web_search": web_search,
    "retrieve_documents": retrieve
}


# -------------------------
# Agent
# -------------------------

def ask_gemini(question: str):

    global conversation_history

    # -------------------------
    # STEP 1: RAG retrieval
    # -------------------------

    # retrieved_results = retrieve(question)

    # print("\n📚 RETRIEVED CONTEXT")

    # for result in retrieved_results:

    #     print("\nSimilarity:", result["similarity"])
    #     print(result["text"])


    # -------------------------
    # STEP 2: Create plan
    # -------------------------

    plan = create_plan(
        question,
        conversation_history
    )

    print("\n🧠 EXECUTION PLAN")

    for i, task in enumerate(plan, start=1):

        print(f"{i}. {task}")


    # -------------------------
    # Search tracking
    # -------------------------

    searched_queries = set()

    all_results = []


    # -------------------------
    # STEP 3: Execute tasks
    # -------------------------

    for i, task in enumerate(plan, start=1):

        print(
            f"\n🚀 EXECUTING TASK {i}: {task}"
        )


        messages = [

                {
            "role": "system",
            "content": (
                "You are an AI research agent. "
                "Complete the user's task using the available tools. "

                "IMPORTANT TOOL RULES: "

                "The ONLY available tools are "
                "calculator, get_time, web_search, and retrieve_documents. "

                "NEVER call any other tool. "

                "NEVER call open_browser, browser, open_url, "
                "fetch, or any other tool. "

                "For arithmetic calculations, always use the calculator tool "
                "instead of calculating the result yourself. "

                "Use retrieve_documents when the information may be available "
                "in the project's internal documentation. "

                "Use web_search when current, external, or internet information "
                "is required. "

                "Prefer retrieve_documents over web_search for questions about "
                "the project's concepts, architecture, implementation, RAG, "
                "memory, tools, planning, or other topics covered by the "
                "internal documentation. "

                "The web_search tool accepts exactly one argument named query. "

                "query must be a natural-language search query. "

                "NEVER pass cursor, id, URL, source, "
                "source_id, or result_id to web_search."
            )
        },

            {
                "role": "user",
                "content": task
            }

        ]


        # -------------------------
        # Agent loop
        # -------------------------

        search_count = 0
        max_searches = 3


        while True:

            response = client.chat.completions.create(

                model="openai/gpt-oss-20b",

                messages=messages,

                tools=[
                    calculator_tool,
                    time_tool,
                    web_search_tool,
                    rag_tool
                ],

                tool_choice="auto"
            )


            message = response.choices[0].message


            messages.append(message)


            # -------------------------
            # No tool call
            # -------------------------

            if not message.tool_calls:

                result = message.content

                print("\n✅ TASK RESULT:")
                print(result)


                all_results.append(

                    f"Task {i}: {task}\n"
                    f"Result: {result}"

                )

                break


            # -------------------------
            # Handle tool calls
            # -------------------------

            for tool_call in message.tool_calls:

                tool_name = tool_call.function.name


                arguments = json.loads(
                    tool_call.function.arguments
                )


                print("\n🔧 TOOL REQUESTED")

                print(
                    "Tool:",
                    tool_name
                )

                print(
                    "Arguments:",
                    arguments
                )


                tool = tool_functions.get(
                    tool_name
                )


                # -------------------------
                # Unknown tool
                # -------------------------

                if tool is None:

                    result = "Unknown tool"


                # -------------------------
                # Calculator
                # -------------------------

                elif tool_name == "calculator":

                    result = tool(

                        arguments["a"],

                        arguments["b"],

                        arguments["operation"]

                    )


                # -------------------------
                # Web search
                # -------------------------

                elif tool_name == "web_search":

                    if search_count >= max_searches:

                        result = (
                            "Search limit reached. "
                            "Use the information already "
                            "collected and provide the best answer."
                        )

                    else:

                        query = arguments["query"]


                        if query in searched_queries:

                            result = (
                                "This query has already been searched. "
                                "Use the existing results."
                            )

                        else:

                            searched_queries.add(
                                query
                            )

                            search_count += 1

                            result = tool(
                                query
                            )
                elif tool_name == "retrieve_documents":

                    query = arguments["query"]

                    retrieved_results = tool(query)

                    formatted_results = []

                    for result in retrieved_results:

                        formatted_results.append(
                           f"Source: {result['source']}\n"
                           f"Similarity: {result['similarity']}\n"
                           f"{result['text']}"
                         )

                    result = "\n\n".join(formatted_results)

                # -------------------------
                # Other tools
                # -------------------------

                else:

                    result = tool()


                print("\n🔧 TOOL RESULT:")

                print(result)


                # -------------------------
                # Send tool result to LLM
                # -------------------------

                messages.append(

                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result)
                    }

                )


    # -------------------------
    # STEP 4: Final synthesis
    # -------------------------

    print(
        "\n🧠 SYNTHESIZING FINAL ANSWER"
    )


    final_messages = [

        {
            "role": "system",
            "content": (
            "You are a research synthesis agent. "
            "Use the research results provided to produce a clear and accurate final answer. "
            "Do not invent information that is not supported by the research results. "
            "Use Markdown formatting. "
            "Prefer headings, short paragraphs, numbered steps, and bullet points. "
            "Avoid large tables unless they are genuinely necessary. "
            "Keep the answer concise and easy to read. "
            "Explain technical concepts in a structured way. "
            "You may use the previous conversation to understand the context of the user's current question."
            )
        }

    ]


    # Add previous conversation

    final_messages.extend(
        conversation_history
    )


    # Add research results ONCE

    final_messages.append(

        {
            "role": "user",
            "content": (

                f"Original question:\n"
                f"{question}\n\n"

                f"Research results:\n"
                f"{chr(10).join(all_results)}"

            )
        }

    )


    # -------------------------
    # Debug request size
    # -------------------------

    print("\n📦 FINAL MESSAGE SIZE:")

    total_chars = sum(
        len(str(message.get("content", "")))
        for message in final_messages
    )

    print(
        "Characters:",
        total_chars
    )


    # -------------------------
    # Final LLM call
    # -------------------------

    final_response = client.chat.completions.create(

        model="openai/gpt-oss-20b",

        messages=final_messages,

        temperature=0

    )


    final_answer = (
        final_response
        .choices[0]
        .message
        .content
    )


    # -------------------------
    # Print final answer
    # -------------------------

    print("\n🤖 FINAL ANSWER:")

    print(final_answer)


    # -------------------------
    # Update memory
    # -------------------------

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


    # Keep last 6 messages

    conversation_history = (
        conversation_history[-6:]
    )


    # -------------------------
    # Save memory
    # -------------------------

    with open(
        "memory.json",
        "w"
    ) as f:

        json.dump(
            conversation_history,
            f,
            indent=2
        )


    print(
        "\n🧠 CONVERSATION MEMORY:"
    )

    print(
        conversation_history
    )


    return final_answer