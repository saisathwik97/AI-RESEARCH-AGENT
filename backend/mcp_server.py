from mcp.server import MCPServer

mcp = MCPServer("AI Research Tools")


@mcp.tool()
def calculator(a: float, b: float, operation: str):
    """Perform addition, subtraction, multiplication, or division."""

    if operation == "add":
        return a + b

    elif operation == "subtract":
        return a - b

    elif operation == "multiply":
        return a * b

    elif operation == "divide":
        if b == 0:
            return "Cannot divide by zero"
        return a / b

    else:
        return "Invalid operation"


if __name__ == "__main__":
    mcp.run()