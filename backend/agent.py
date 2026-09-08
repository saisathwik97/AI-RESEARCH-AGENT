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
            "Use ONLY the research results provided in the user's message "
            "to answer the original question. "

            "Treat every retrieved document passage as authoritative source text. "
            "Do not create descriptions for listed items unless those descriptions "
            "are explicitly present in the research results. "

            "When a source provides a list, reproduce the list using the "
            "same terminology and do not add explanations that are not present "
            "in the source. "
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
            "Use retrieve_documents first when the question may be answered "
            "from the project's internal documentation. "
            "If retrieve_documents returns relevant information, use that "
            "information and do not call web_search unless the user explicitly "
            "needs current or external information. "
            "Use web_search when current, external, or internet information "
            "is required, or when the internal documents do not contain "
            "sufficient information to answer the question. "
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
    "You are a research synthesis agent. "

    "Use ONLY the research results provided in the user's message "
    "to answer the original question. "
     "IMPORTANT RAG GROUNDING RULE: "
"When retrieve_documents returns relevant information, "
"the retrieved text is the only factual source. "

"Answer using only claims explicitly stated in the retrieved text. "

"Do NOT create new subcomponents, categories, stages, "
"examples, explanations, interpretations, or conclusions "
"unless they are explicitly stated in the retrieved text. "

"Do NOT rename or reinterpret concepts from the retrieved text. "

"Do NOT combine separate retrieved statements into a new claim "
"unless the relationship between them is explicitly stated. "

"If the retrieved text does not contain enough information "
"to answer the task, say that the retrieved documents do not "
"provide that information. "
    "IMPORTANT GROUNDING RULE: "
    "Do NOT use your pretrained knowledge to add facts, examples, "
    "libraries, frameworks, tools, domains, URLs, or claims. "

    "Do NOT expand, enrich, or infer information beyond the "
    "provided research results. "

    "If the research results contain only a short list, "
    "preserve that list without adding additional details. "

    "If the research results do not contain enough information "
    "to answer a requested detail, explicitly say that the "
    "research results do not provide that detail. "

    "You may use previous conversation ONLY to understand "
    "the context of the user's question. "
    "Do NOT use previous assistant answers as factual evidence. "

    "When information comes from retrieve_documents, "
    "the retrieved document text is the authoritative source. "
    "Use only claims explicitly supported by that text. "
    "Do not combine it with outside knowledge. "

    "Do not invent citations or sources. "

    "Use Markdown formatting. "
    "Prefer headings, short paragraphs, numbered steps, "
    "and bullet points. "
    "Avoid large tables unless genuinely necessary. "
    "Keep the answer concise and easy to read."
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

                    if not retrieved_results:

                        result = (
                            "No sufficiently relevant information "
                            "was found in the internal documents."
                        )

                    else:

                        formatted_results = []

                        for result_item in retrieved_results:

                            formatted_results.append(
                                f"[SOURCE: {result_item['source']}]\n"
                                f"[SIMILARITY: {result_item['similarity']:.4f}]\n"
                                f"{result_item['text']}"
                            )

                        result = "\n\n".join(
                            formatted_results
                        )
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
            "You are a strict research answer generator. "

            "Your ONLY factual source is the text provided under "
            "'Research results'. "

            "You MUST NOT use your pretrained knowledge. "

            "You MUST NOT add any information that is not explicitly "
            "present in the Research results. "

            "You MUST NOT add examples, libraries, frameworks, tools, "
            "use cases, domains, explanations, or claims from outside "
            "the Research results. "

            "Simply rewrite and organize the Research results into "
            "a clear answer to the Original question. "
             " When presenting information from the Research results, "
            "preserve the source information provided with each result. "
            "Do not invent sources or citations. "
            "You may change wording and formatting, but you must "
            "preserve the factual meaning of the Research results. "

            "If the Research results do not contain enough information "
            "to answer something, explicitly say that the research "
            "results do not provide that information. "

            "Use Markdown formatting."
        )
    }
]

    final_messages.append(
        {
            "role": "user",
            "content": (
                f"Original question:\n{question}\n\n"
                f"Research results:\n"
                f"{chr(10).join(all_results)}"
            )
        }
    )

    # Add research results ONCE

    


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