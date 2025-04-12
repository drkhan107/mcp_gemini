# client.py
import asyncio
import sys
import os
import json
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack

import google.generativeai as genai
import google.ai.generativelanguage as glm
from google.protobuf.struct_pb2 import Struct

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool as McpTool # Alias McpTool to avoid clash
from dotenv import load_dotenv
import logging

from datetime import datetime


# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - CLIENT - %(levelname)s - %(message)s')

# Load API Key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

genai.configure(api_key=GOOGLE_API_KEY)

class GeminiMCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.mcp_tools: List[McpTool] = []
        self.gemini_tools: List[glm.Tool] = []
        try:
            # Set up the model
            generation_config = {
                "temperature": 0.9,
                "top_p": 1,
                "top_k": 1,
                "max_output_tokens": 2048,
            }
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]
            self.model = genai.GenerativeModel(
                model_name="gemini-2.5-pro-preview-03-25", # Or another function-calling capable model
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            self.chat = self.model.start_chat(enable_automatic_function_calling=False) # Manual function calling
            logging.info("Gemini model initialized.")
        except Exception as e:
            logging.exception("Failed to initialize Gemini model")
            raise

    def _mcp_schema_to_gemini_schema(self, mcp_schema: Dict[str, Any]) -> glm.Schema:
        """Converts MCP JSON Schema dict to Gemini Schema object."""
        # Basic conversion, might need more robust handling for complex schemas
        properties = {}
        required = mcp_schema.get('required', [])
        for name, prop_schema in mcp_schema.get('properties', {}).items():
            # Gemini type mapping (simplified)
            prop_type_str = prop_schema.get('type', 'string').upper()
            try:
                prop_type = glm.Type[prop_type_str]
            except KeyError:
                prop_type = glm.Type.STRING # Default fallback

            properties[name] = glm.Schema(
                type=prop_type,
                description=prop_schema.get('description', '')
            )

        return glm.Schema(
            type=glm.Type.OBJECT,
            properties=properties,
            required=required,
            description=mcp_schema.get('description', '')
        )

    async def connect_to_server(self, server_script_path: str):
        """Connects to the specified MCP server."""
        logging.info(f"Attempting to connect to server: {server_script_path}")
        is_python = server_script_path.endswith('.py')
        if not is_python:
            # Extend this if you support other server types (e.g., node)
            raise ValueError("Server script must be a .py file for this client")

        command = "python" # Assumes python is in PATH
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None # Inherit environment
        )

        try:
            # Enter the stdio_client context
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            read_stream, write_stream = stdio_transport

            # Enter the ClientSession context
            self.session = await self.exit_stack.enter_async_context(ClientSession(read_stream, write_stream))

            await self.session.initialize()
            logging.info("MCP session initialized.")

            # List available MCP tools
            response = await self.session.list_tools()
            self.mcp_tools = response.tools if response else []

            # Convert MCP tools to Gemini tool format
            self.gemini_tools = []
            declarations = []
            for mcp_tool in self.mcp_tools:
                 if isinstance(mcp_tool.inputSchema, dict): # Ensure it's a dict
                    param_schema = self._mcp_schema_to_gemini_schema(mcp_tool.inputSchema)
                    declarations.append(glm.FunctionDeclaration(
                        name=mcp_tool.name,
                        description=mcp_tool.description or '',
                        parameters=param_schema
                    ))
                 else:
                     logging.warning(f"Skipping tool {mcp_tool.name} due to non-dict inputSchema: {type(mcp_tool.inputSchema)}")

            if declarations:
                self.gemini_tools.append(glm.Tool(function_declarations=declarations))

            tool_names = [t.name for t in self.mcp_tools]
            logging.info(f"Connected to server. Available tools: {tool_names}")
            if not tool_names:
                 logging.warning("No tools found on the server.")

        except Exception as e:
            logging.exception(f"Failed to connect or initialize MCP server at {server_script_path}")
            await self.cleanup() # Ensure cleanup on connection failure
            raise

    async def process_query(self, query: str) -> str:
        """Processes a user query using Gemini and MCP tools."""
        if not self.session:
            return "Error: Not connected to MCP server."
        if not self.model:
             return "Error: Gemini model not initialized."

        # --- CUSTOM PROMPT AREA ---
        # Example: Use a template to structure the request
        prompt_template = """
        **Task:** Answer the user's question below.
        **Instructions:**
        1. Analyze the question: '{user_query}'.
        2. Determine if current information from the internet is required.
        3. If yes, use the 'web_search_and_crawl' tool with an appropriate search term derived from the question.
        4. Synthesize the information (from your knowledge or the web search results) into a clear and concise answer.
        5. If you used the search tool, briefly mention that the information is based on recent web results.

        Today is: {today}

        **User Question:**
        {user_query}
        """

        today=datetime.today().strftime('%Y-%m-%d')
        final_prompt = prompt_template.format(user_query=query, today=today)
        # --------------------------

        logging.info(f"Sending final prompt structure to Gemini (showing first 200 chars): '{final_prompt[:200]}...'")
        try:
            logging.debug(f"Attempting to send message with tools: {self.gemini_tools}")
            if not self.gemini_tools:
                 logging.warning("No Gemini tools configured to send!")

            # Send the formatted prompt to Gemini
            response = await self.chat.send_message_async(final_prompt, tools=self.gemini_tools) # Use final_prompt here
            logging.debug(f"Initial Gemini response received: {response.candidates[0].content if response.candidates else 'No candidates'}")


            # Check for function call request
            response_part = response.candidates[0].content.parts[0]
            if response_part.function_call:
                fc = response_part.function_call
                tool_name = fc.name
                tool_args_struct = fc.args # This is a google.protobuf.struct_pb2.Struct

                 # Convert Struct to Python dict
                tool_args = {}
                for key, value in tool_args_struct.items():
                    # Basic conversion, might need refinement for complex types
                    if hasattr(value, 'string_value'):
                        tool_args[key] = value.string_value
                    elif hasattr(value, 'number_value'):
                        tool_args[key] = value.number_value
                    elif hasattr(value, 'bool_value'):
                         tool_args[key] = value.bool_value
                    # Add more type handling if needed (list, struct, null)
                    else:
                         tool_args[key] = str(value) # Fallback

                logging.info(f"Gemini requested tool call: {tool_name}({tool_args})")

                # Find the corresponding MCP tool (optional, for validation)
                mcp_tool = next((t for t in self.mcp_tools if t.name == tool_name), None)
                if not mcp_tool:
                    logging.error(f"Gemini requested unknown tool: {tool_name}")
                    # Send error back to Gemini
                    func_response = glm.FunctionResponse(name=tool_name, response={"error": f"Tool '{tool_name}' not found."})
                    response = await self.chat.send_message_async(
                         glm.Part(function_response=func_response), tools=self.gemini_tools
                    )

                else:
                    # Call the MCP server tool
                    try:
                        logging.info(f"Calling MCP tool '{tool_name}'...")
                        mcp_result = await self.session.call_tool(tool_name, tool_args)
                        logging.info(f"MCP tool '{tool_name}' executed.")

                        # Check if MCP result indicates an error (basic check)
                        mcp_content_str = " ".join(c.text for c in mcp_result.content if hasattr(c, 'text'))
                        if mcp_result.isError or "error occurred" in mcp_content_str.lower():
                             logging.warning(f"MCP tool '{tool_name}' returned an error: {mcp_content_str}")
                             tool_response_content = {"error": mcp_content_str}
                        else:
                             tool_response_content = {"result": mcp_content_str}


                        # Format result for Gemini
                        func_response = glm.FunctionResponse(
                            name=tool_name,
                            response=tool_response_content
                        )

                        # Send tool result back to Gemini
                        logging.info(f"Sending tool result for '{tool_name}' back to Gemini.")
                        response = await self.chat.send_message_async(
                             glm.Part(function_response=func_response), tools=self.gemini_tools
                        )
                        logging.debug("Received final Gemini response after tool call.")

                    except Exception as e:
                        logging.exception(f"Error calling MCP tool '{tool_name}'")
                        # Send error back to Gemini
                        func_response = glm.FunctionResponse(name=tool_name, response={"error": f"Failed to execute tool: {e}"})
                        response = await self.chat.send_message_async(
                             glm.Part(function_response=func_response), tools=self.gemini_tools
                        )

            # --- Extract final text response ---
            final_text = ""
            try:
                 # Handle potential lack of text part if only error occurred
                 if response.candidates and response.candidates[0].content.parts:
                      final_text = response.candidates[0].content.parts[0].text
                 elif response.prompt_feedback and response.prompt_feedback.block_reason:
                      final_text = f"Blocked by safety settings: {response.prompt_feedback.block_reason}"
                      logging.warning(f"Response blocked: {response.prompt_feedback.block_reason_message}")
                 else:
                      final_text = "[No text content in final response]"
                      logging.warning("Gemini's final response did not contain text.")

            except (AttributeError, IndexError, ValueError) as e:
                logging.exception(f"Error extracting final text from Gemini response: {e}. Full response: {response}")
                final_text = "[Error processing Gemini response]"

            return final_text

        except Exception as e:
            logging.exception(f"An error occurred while processing query '{query}'")
            return f"An error occurred: {e}"

    async def chat_loop(self):
        """Runs the interactive chat loop."""
        print("\n--- Gemini MCP Client ---")
        print("Connected to MCP server. Ask me anything!")
        print("Type 'quit' or 'exit' to end.")

        while True:
            try:
                query = input("\nYou: ").strip()
                if query.lower() in ['quit', 'exit']:
                    print("Exiting chat.")
                    break
                if not query:
                    continue

                print("Gemini: Thinking...")
                response_text = await self.process_query(query)
                print(f"\nGemini: {response_text}")

            except KeyboardInterrupt:
                print("\nExiting chat.")
                break
            except Exception as e:
                logging.exception("Error in chat loop")
                print(f"\nAn error occurred: {e}")

    async def cleanup(self):
        """Cleans up resources."""
        logging.info("Cleaning up client resources...")
        await self.exit_stack.aclose()
        logging.info("Client resources cleaned up.")

# --- Main Execution ---
async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server.py>")
        sys.exit(1)

    server_path = sys.argv[1]
    client = GeminiMCPClient()

    try:
        await client.connect_to_server(server_path)
        await client.chat_loop()
    except Exception as e:
        logging.exception("Client failed to start or run.")
        print(f"Client failed: {e}")
    finally:
        await client.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nClient shutdown requested.")