import asyncio

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters


async def main():

    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],
    )

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            # Initialize connection
            await session.initialize()

            # Discover available tools
            tools = await session.list_tools()

            print("\nAvailable MCP tools:")

            for tool in tools.tools:
                print(f"- {tool.name}")

            # Call the add tool
            result = await session.call_tool(
                "calculator",
                arguments={
                    "a": 10,
                    "b": 20,
                    "operation": "multiply"
                }
            )

            print("\nTool result:")
            print(result)


if __name__ == "__main__":
    asyncio.run(main())