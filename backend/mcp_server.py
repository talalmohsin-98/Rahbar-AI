from mcp.server.fastmcp import FastMCP

mcp = FastMCP("rahbar-ai", port=8001)

@mcp.tool()
def ping(message: str) -> str:
    """Echoes back a message — used to verify the MCP server is reachable."""
    return f"pong: {message}"

if __name__ == "__main__":
    mcp.run(transport="sse")