from ddgs import DDGS
def calculator(a:float,b:float,operation:str):
    print("Calculator caller ..!")

    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b != 0:
            return a / b
        else:
            return "Cannot divide by zero."
    else:
        return "Invalid operation. Please choose from 'add', 'subtract', 'multiply', or 'divide'."

from datetime import datetime


def get_time():
    print("🕐 TIME TOOL CALLED")

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def web_search(query: str):
    print("🌐 WEB SEARCH TOOL CALLED")
    print("Query:", query)

    try:
        results = DDGS().text(
            query,
            max_results=1
        )

        if not results:
            return "No search results found for this query."

        formatted_results = []

        for i, result in enumerate(results, start=1):
            formatted_results.append(
                f"""
Source {i}
Title: {result.get('title')}
URL: {result.get('href')}
Summary: {result.get('body')}
"""
            )

        return "\n".join(formatted_results)

    except Exception as e:
        print("❌ WEB SEARCH ERROR:", e)

        return (
            f"Web search failed for query '{query}'. "
            "Try a different search query."
        )