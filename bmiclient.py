import asyncio
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os
import json

# Start the MCP-compatible server (e.g., BMI calculator)
server_params = StdioServerParameters(command="python", args=["bmiserver.py"])


def llm_client(message: str) -> str:
    """
    Send a message to the OpenAI LLM and return the response.
    """
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an intelligent assistant. You will execute tasks as prompted."},
            {"role": "user", "content": message}
        ],
        max_tokens=300,
        temperature=0.2
    )

    return response.choices[0].message.content.strip()


def get_router_prompt(query: str, tools) -> str:
    tool_names = [tool.name for tool in tools]
    return (
        "You are a smart router that decides whether a user's question should call a tool.\n\n"
        f"Available tools: {tool_names}\n\n"
        f"User question: {query}\n\n"
        "If the question requires one of these tools, respond ONLY with this JSON:\n"
        '{ "use_tool": true }\n\n'
        "If no tool is needed, respond ONLY with this JSON:\n"
        '{ "use_tool": false }\n'
    )


def get_tool_selection_prompt(query: str, tools) -> str:
    tools_description = "\n".join([
        f"- {tool.name}: {tool.description}, {tool.inputSchema}" for tool in tools
    ])
    return (
        f"You are a helpful assistant. You can call the following tools:\n\n"
        f"{tools_description}\n\n"
        f"User query: {query}\n\n"
        "IMPORTANT: When you need to use a tool, respond ONLY with the exact JSON object below and nothing else:\n"
        '{\n'
        '  "tool": "tool-name",\n'
        '  "arguments": {\n'
        '    "arg1": "value",\n'
        '    "arg2": "value"\n'
        '  }\n'
        '}'
    )


def is_valid_tool_call(response: str, tools: list) -> bool:
    try:
        parsed = json.loads(response)
        if not isinstance(parsed, dict):
            return False
        if "tool" not in parsed or "arguments" not in parsed:
            return False
        if parsed["tool"] not in [tool.name for tool in tools]:
            return False
        if not isinstance(parsed["arguments"], dict):
            return False
        return True
    except json.JSONDecodeError:
        return False


async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()

            # Greet the user
            greeting = llm_client("Greet the user briefly and let them know they can ask any question.")
            print(greeting)

            while True:
                query = input("\nYour question (or type 'exit' to quit): ").strip()
                if query.lower() in ("exit", "quit"):
                    print("Goodbye!")
                    break

                # Step 1: route decision
                route_prompt = get_router_prompt(query, tools.tools)
                route_response = llm_client(route_prompt)

                try:
                    route_decision = json.loads(route_response)
                except json.JSONDecodeError:
                    print("Invalid router response. Falling back to direct LLM answer.")
                    print(llm_client(query))
                    continue

                if route_decision.get("use_tool") is True:
                    # Step 2: build tool call JSON
                    prompt = get_tool_selection_prompt(query, tools.tools)
                    tool_response = llm_client(prompt)

                    if is_valid_tool_call(tool_response, tools.tools):
                        tool_call = json.loads(tool_response)

                        try:
                            result = await session.call_tool(
                                tool_call["tool"], arguments=tool_call["arguments"]
                            )

                            print(
                                f'\n✅ Tool Result:\n'
                                f'BMI for weight {tool_call["arguments"].get("weight_kg")}kg '
                                f'and height {tool_call["arguments"].get("height_m")}m is:\n'
                                f'{result.content[0].text}'
                            )
                        except Exception as e:
                            print(f"❌ Tool execution failed: {e}")
                    else:
                        print("❌ Invalid tool call response from LLM.")
                else:
                    # Step 3: direct answer from LLM
                    print("\n💬 Direct answer from LLM:")
                    print(llm_client(query))


if __name__ == "__main__":
    asyncio.run(run())
