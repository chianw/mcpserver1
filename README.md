# mcpserver1

## Simple Python-based MCP server and client that uses OpenAI

MCP server provides a tool that takes in weight in kg and height in meters, and returns BMI

MCP client uses STDIO to connect to the MCP server and includes a router to decide whether the tool should be used or if direct LLM response is required

It is asssumed that **OPENAI_API_KEY** is available as environment variable

