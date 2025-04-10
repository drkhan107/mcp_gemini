import asyncio
from contextlib import AsyncExitStack
from datetime import datetime
import logging
from typing import Any, Dict, Optional
from mcp import ClientSession
from mcp.client.sse import sse_client
import google.generativeai as genai
import google.ai.generativelanguage as glm
from dotenv import load_dotenv
import os


# Load API Key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

genai.configure(api_key=GOOGLE_API_KEY)

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = None

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
        self.chat = self.model.start_chat(enable_automatic_function_calling=True) # Manual function calling
        logging.info("Gemini model initialized.")
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
    async def connect_to_sse_server(self, server_url: str):
        """Connect to an MCP server running with SSE transport"""
        # Store the context managers so they stay alive
        self._streams_context = sse_client(url=server_url)
        streams = await self._streams_context.__aenter__()

        self._session_context = ClientSession(*streams)
        self.session: ClientSession = await self._session_context.__aenter__()

        # Initialize
        await self.session.initialize()

        # List available tools to verify connection
        print("Initialized SSE client...")
        print("Listing tools...")
        response = await self.session.list_tools()
        tools = response.tools
        self.mcp_tools=tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])


        ######## For Gemini
        # Convert MCP tools to Gemini tool format
        self.gemini_tools = []
        declarations = []
        for mcp_tool in tools:
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
            self.gemini_tools=declarations#glm.Tool(function_declarations=declarations)
        #print("G tools", self.gemini_tools[0])
        tool_names = [t.name for t in tools]
        logging.info(f"Connected to server. Available tools: {tool_names}")
        if not tool_names:
                logging.warning("No tools found on the server.")

        ##########

    async def cleanup(self):
        """Properly clean up the session and streams"""
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self._streams_context:
            await self._streams_context.__aexit__(None, None, None)

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()
        available_tools = [{ 
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        # Initial Claude API call
        response = self.anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=messages,
            tools=available_tools
        )

        # Process response and handle tool calls
        tool_results = []
        final_text = []

        for content in response.content:
            if content.type == 'text':
                final_text.append(content.text)
            elif content.type == 'tool_use':
                tool_name = content.name
                tool_args = content.input
                
                # Execute tool call
                result = await self.session.call_tool(tool_name, tool_args)
                tool_results.append({"call": tool_name, "result": result})
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                # Continue conversation with tool results
                if hasattr(content, 'text') and content.text:
                    messages.append({
                    "role": "assistant",
                    "content": content.text
                    })
                messages.append({
                    "role": "user", 
                    "content": result.content
                })

                # Get next response from Claude
                response = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    messages=messages,
                )

                final_text.append(response.content[0].text)

        return "\n".join(final_text)
    
    async def process_query_gemini(self, query: str) -> str:
        
        """Processes a user query using Gemini and MCP tools.
        Added this function to support Gemini llm for MCP tools"""
        if not self.session:
            return "Error: Not connected to MCP server."
        if not self.model:
             return "Error: Gemini model not initialized."

        # --- CUSTOM PROMPT AREA ---
        # Example: Use a template to structure the request
        prompt_template = """
        **Task:** You are NoteTaker, a personal meeting assistant. Your task is to take and store notes and answe the user queries from the past meeting notes.
        **Instructions:**
        1. Analyze the question: '{user_query}'.
        2. Determine if current information from the tool is required.
        3. If yes, use the appropriate tool with an appropriate query term derived from the question.
        4. Synthesize the information into a clear and concise answer.
        5. If you used any tool, briefly mention that the information is based on the information from the tool.
        6. Resolve dates in this format: YYYY-MM-DD.
        7. To insert new notes or MoM's, if date is not given, send Todays, date. 
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
            #print(f"*** -- ***Initial Gemini response received: {response.candidates[0].content if response.candidates else 'No candidates'}")


            # Check for function call request
            response_text = response.candidates[0].content.parts[0]
            
            print("** Part",response_text)
            parts=len(response.candidates[0].content.parts)
            print("Parts Length: ",parts)
            if parts >1 :
                response_part=response.candidates[0].content.parts[1] 
                fc = response_part.function_call
                tool_name = fc.name
                tool_args_struct = fc.args # This is a google.protobuf.struct_pb2.Struct
                print("** Tool Call",tool_name)
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
                mcp_tool = next((t for t in self.gemini_tools if t.name == tool_name), None)
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
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                    
                response = await self.process_query_gemini(query)
                print("\n" + response)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")


async def main():
    if len(sys.argv) < 2:
        print("Usage: uv run client.py <URL of SSE MCP server (i.e. http://localhost:8080/sse)>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_sse_server(server_url=sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    import sys
    asyncio.run(main())