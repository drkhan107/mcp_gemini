import asyncio
import json
import os
import sys
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack, asynccontextmanager

import streamlit as st
from mcp import ClientSession
from mcp.client.sse import sse_client

from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import Tool

# No 'Part' import needed here

# --- Configuration ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found in .env file. Please add it.")
    st.stop()

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)

# Default MCP Server URL
DEFAULT_MCP_SERVER_URL = "http://localhost:8080/sse"

# --- MCP Client Class (remains the same) ---
class MCPClient:
    # ... (Keep the existing MCPClient class code) ...
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self._streams_context = None
        self._session_context = None
        self._available_tools_cache: Optional[List[Dict[str, Any]]] = None

    async def connect_to_sse_server(self, server_url: str):
        """Connect to an MCP server running with SSE transport"""
        st.write(f"Attempting to connect to MCP server at {server_url}...")
        try:
            # Use AsyncExitStack for robust resource management
            async with AsyncExitStack() as stack:
                self._streams_context = sse_client(url=server_url)
                streams = await stack.enter_async_context(self._streams_context)

                self._session_context = ClientSession(*streams)
                self.session: ClientSession = await stack.enter_async_context(self._session_context)

                 # Keep the contexts managed by the stack until cleanup is called elsewhere if needed
                # Or structure connect/cleanup differently if connect is one-off

                await self.session.initialize()

                st.write("MCP Session Initialized. Listing available tools...")
                response = await self.session.list_tools()
                tools = response.tools
                self._available_tools_cache = [{
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                } for tool in tools]

                st.success(f"Connected to MCP server! Tools available: {[tool['name'] for tool in self._available_tools_cache]}")
                self._available_tools_cache=tools
                return True # Indicate success
        except Exception as e:
            st.error(f"Failed to connect to MCP server: {e}")
            # Ensure cleanup happens on connection failure if contexts were partially entered
            await self.cleanup() # Attempt cleanup
            self._available_tools_cache = None
            return False # Indicate failure

    async def cleanup(self):
        """Properly clean up the session and streams"""
        st.write("Cleaning up MCP connection...")
        # Rely on context managers' __aexit__ if using AsyncExitStack properly
        # Manual cleanup if not using AsyncExitStack or for explicit disconnect
        if self._session_context and hasattr(self._session_context, '__aexit__'):
             try:
                  await self._session_context.__aexit__(None, None, None)
             except Exception as e:
                  st.warning(f"Error during session cleanup: {e}")
        if self._streams_context and hasattr(self._streams_context, '__aexit__'):
             try:
                  await self._streams_context.__aexit__(None, None, None)
             except Exception as e:
                  st.warning(f"Error during streams cleanup: {e}")

        self._session_context = None
        self._streams_context = None
        self.session = None
        self._available_tools_cache = None
        st.write("MCP Cleanup complete.")

    async def get_available_tools(self) -> List[Dict[str, Any]]:
        if not self.session:
            st.error("MCP session not available.")
            return []
        if self._available_tools_cache is None:
             st.write("Refreshing tool list...")
             response = await self.session.list_tools()
             tools = response.tools
             self._available_tools_cache = [{
                 "name": tool.name,
                 "description": tool.description,
                 "input_schema": tool.inputSchema
             } for tool in tools]
            
             st.write(f"Tools refreshed: {[tool['name'] for tool in self._available_tools_cache]}")
             ## Testt
             self._available_tools_cache=tools
             
        return self._available_tools_cache

    async def call_tool(self, tool_name: str, args: Dict[str, Any]):
        """Calls a specific tool on the MCP server."""
        if not self.session:
            raise ConnectionError("MCP Client is not connected.")
        st.write(f"Calling MCP tool: {tool_name} with args: {args}")
        result = await self.session.call_tool(tool_name, args)
        st.write(f"Received result from {tool_name}")
        return result


# --- Streamlit App Logic (State, Connection Management remain the same) ---
st.set_page_config(layout="wide")
st.title("📝 Meeting Notes Assistant (MCP + Gemini)")
st.caption("Add notes using natural language or ask questions about past notes.")

# --- State Management ---
if 'mcp_client_instance' not in st.session_state: # Renamed to avoid conflict
    st.session_state.mcp_client_instance = MCPClient()
    st.session_state.mcp_connected = False
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'gemini_chat' not in st.session_state:
    try:
        model = genai.GenerativeModel(
            'gemini-1.5-flash-latest',
             system_instruction="You are a helpful meeting notes assistant..." # Keep your system prompt
        )
        st.session_state.gemini_chat = model.start_chat(history=[])
    except Exception as e:
        st.error(f"Failed to initialize Gemini Model: {e}")
        st.stop()

# --- Connection Management ---
mcp_server_url = st.sidebar.text_input("MCP Server URL", DEFAULT_MCP_SERVER_URL)

async def run_connection():
    client = st.session_state.mcp_client_instance
    connected = await client.connect_to_sse_server(mcp_server_url)
    st.session_state.mcp_connected = connected
    st.rerun()

if not st.session_state.mcp_connected:
    if st.sidebar.button("Connect to MCP Server"):
        try:
            asyncio.run(run_connection())
        except Exception as e:
            st.error(f"Error running connection: {e}")
            st.session_state.mcp_connected = False
else:
    st.sidebar.success(f"Connected to {mcp_server_url}")
    if st.sidebar.button("Disconnect"):
        async def run_cleanup():
            client = st.session_state.mcp_client_instance
            await client.cleanup()
            st.session_state.mcp_connected = False
            st.session_state.messages = []
            # Optionally reset gemini chat history?
            # st.session_state.gemini_chat = model.start_chat(history=[])
            st.rerun()
        try:
            asyncio.run(run_cleanup())
        except Exception as e:
            st.error(f"Error during disconnect: {e}")

# --- Display Chat History (remains the same) ---
st.write("## Chat History")
for msg in st.session_state.messages:
    role = msg["role"]
    if role == "tool":
        with st.chat_message("assistant", avatar="🛠️"):
             st.markdown(f"```tool_code\n{msg['content']}\n```")
             if "result" in msg:
                 st.markdown(f"**Result:**\n```text\n{msg['result']}\n```")
    else:
       with st.chat_message(role):
            st.markdown(msg["content"])

import google.ai.generativelanguage as glm
def _mcp_schema_to_gemini_schema(mcp_schema: Dict[str, Any]) -> glm.Schema:
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
# --- Main Chat Interaction ---
async def process_user_query(user_query: str):
    """Handles user input, interacts with Gemini and MCP tools."""
    if not st.session_state.mcp_connected or not st.session_state.mcp_client_instance:
        st.error("Not connected to MCP server. Please connect first.")
        return

    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    mcp_client: MCPClient = st.session_state.mcp_client_instance
    gemini_chat = st.session_state.gemini_chat
    try:
        available_tools = await mcp_client.get_available_tools()
        if not available_tools:
             st.error("Could not fetch tools from MCP server.")
             st.session_state.messages.append({"role": "model", "content": "Sorry, I cannot access my tools right now."})
             st.rerun()
             return
       # gemini_tools=_mcp_schema_to_gemini_schema(available_tools)


        declarations = []
        for mcp_tool in mcp_client._available_tools_cache:
            print(mcp_tool.name)
            if isinstance(mcp_tool.inputSchema, dict): # Ensure it's a dict
                param_schema = _mcp_schema_to_gemini_schema(mcp_tool.inputSchema)
                declarations.append(glm.FunctionDeclaration(
                    name=mcp_tool.name,
                    description=mcp_tool.description or '',
                    parameters=param_schema
                ))
            else:
                st.warning(f"Skipping tool {mcp_tool.name} due to non-dict inputSchema: {type(mcp_tool.inputSchema)}")

        # if declarations:
        #     self.gemini_tools.append(glm.Tool(function_declarations=declarations))



        gemini_tool_config = Tool(function_declarations=declarations)

        with st.spinner("Thinking..."):
            response = await gemini_chat.send_message_async(
                user_query,
                tools=[gemini_tool_config]
            )

        if not response.parts:
            st.error("Received an empty response from the language model.")
            st.session_state.messages.append({"role": "model", "content": "Sorry, I received an empty response."})
            return

        response_part = response.parts[0]

        # Check for function calls (tool use)
        if response_part.function_call:
            function_call = response_part.function_call
            tool_name = function_call.name
            tool_args = dict(function_call.args)

            tool_call_msg = f"Calling tool: `{tool_name}` with args: `{json.dumps(tool_args)}`"
            st.session_state.messages.append({"role": "tool", "content": tool_call_msg})
            with st.chat_message("assistant", avatar="🛠️"):
                 st.markdown(tool_call_msg)
                 st.spinner(f"Executing {tool_name}...")

            tool_result = await mcp_client.call_tool(tool_name, tool_args)
            tool_result_content = tool_result.content if isinstance(tool_result.content, str) else json.dumps(tool_result.content)
            st.session_state.messages[-1]["result"] = tool_result_content

            # --- MODIFIED SECTION ---
            # Send the tool result back to Gemini
            with st.spinner("Processing tool result..."):
                 # Construct the function response payload as a dictionary
                 # The SDK's send_message should handle formatting this correctly.
                 function_response_payload = {
                    "function_response": {
                        "name": tool_name,
                        "response": {
                            # Pass the actual content returned by the tool
                            "content": tool_result_content
                        }
                    }
                 }

                 # Send this dictionary directly.
                 response = await gemini_chat.send_message_async(
                    function_response_payload # Pass the dict directly
                 )
                 # --- END MODIFIED SECTION ---

                 # Get the final text response after processing the tool result
                 if response.parts and response.parts[0].text:
                     final_text = response.text # Or response.parts[0].text
                 else:
                     final_text = "[Tool executed, but no further text response from model]"


        elif response_part.text:
            # No tool call, just a text response
            final_text = response_part.text
        else:
            st.warning("Received a response part with no text or function call.")
            final_text = "[No actionable response received]"

        # Display final Gemini response and add to history
        st.session_state.messages.append({"role": "model", "content": final_text})
        with st.chat_message("model"):
            st.markdown(final_text)

    except Exception as e:
        st.error(f"An error occurred during processing: {e}", icon="🚨")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        st.session_state.messages.append({"role": "model", "content": f"Sorry, an error occurred: {e}"})


# --- Chat Input (remains the same) ---
if prompt := st.chat_input("What notes should I record or find?", disabled=not st.session_state.mcp_connected):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
             # If in an environment like Jupyter/Streamlit where a loop is running
             asyncio.ensure_future(process_user_query(prompt))
        else:
             loop.run_until_complete(process_user_query(prompt))
    except RuntimeError as ex:
         if "There is no current event loop in thread" in str(ex):
              loop = asyncio.new_event_loop()
              asyncio.set_event_loop(loop)
              loop.run_until_complete(process_user_query(prompt))
              # loop.close() # Consider closing if appropriate for the thread context
         else:
              st.error(f"Runtime error running chat processing: {ex}")
    except Exception as e:
         st.error(f"Error processing chat input: {e}")

    # Rerun might be needed depending on how ensure_future updates state
    # If updates aren't appearing, uncomment the next line:
    # st.rerun()
elif not st.session_state.mcp_connected:
     st.info("Please connect to the MCP server using the sidebar to start.")