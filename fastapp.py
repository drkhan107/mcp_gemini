import asyncio
from contextlib import AsyncExitStack
from datetime import datetime
import logging
from typing import Any, Dict, List, Optional
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from mcp import ClientSession
from mcp.client.sse import sse_client
import google.generativeai as genai
import google.ai.generativelanguage as glm
from dotenv import load_dotenv
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Load API Key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

genai.configure(api_key=GOOGLE_API_KEY)

# Define request/response models
class ConnectionRequest(BaseModel):
    server_url: str

class QueryRequest(BaseModel):
    query: str

class ToolInfo(BaseModel):
    name: str
    description: Optional[str] = None

class ConnectionResponse(BaseModel):
    status: str
    connected: bool
    available_tools: List[ToolInfo] = []
    message: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    
class StatusResponse(BaseModel):
    status: str
    connected: bool
    available_tools: List[ToolInfo] = []

# Create FastAPI app
app = FastAPI(
    title="MCP Client API",
    description="API for interacting with MCP (Machine Common Protocol) through Gemini",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.connected = False
        self.mcp_tools = []
        self.gemini_tools = []

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
            model_name="gemini-2.5-pro-preview-03-25",
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        self.chat = self.model.start_chat(enable_automatic_function_calling=True)
        logging.info("Gemini model initialized.")

    def _mcp_schema_to_gemini_schema(self, mcp_schema: Dict[str, Any]) -> glm.Schema:
        """Converts MCP JSON Schema dict to Gemini Schema object."""
        properties = {}
        required = mcp_schema.get('required', [])
        for name, prop_schema in mcp_schema.get('properties', {}).items():
            prop_type_str = prop_schema.get('type', 'string').upper()
            try:
                prop_type = glm.Type[prop_type_str]
            except KeyError:
                prop_type = glm.Type.STRING  # Default fallback

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

    async def connect_to_sse_server(self, server_url: str) -> ConnectionResponse:
        """Connect to an MCP server running with SSE transport"""
        # Store the context managers so they stay alive
        try:
            self._streams_context = sse_client(url=server_url)
            streams = await self._streams_context.__aenter__()

            self._session_context = ClientSession(*streams)
            self.session: ClientSession = await self._session_context.__aenter__()
            print("Connecting: ", server_url)
            # Initialize
            await self.session.initialize()

            # List available tools to verify connection
            logging.info("Initialized SSE client...")
            logging.info("Listing tools...")
            response = await self.session.list_tools()
            tools = response.tools
            self.mcp_tools = tools
            tool_names = [tool.name for tool in tools]
            logging.info(f"Connected to server. Available tools: {tool_names}")

            # Convert MCP tools to Gemini tool format
            self.gemini_tools = []
            declarations = []
            for mcp_tool in tools:
                if isinstance(mcp_tool.inputSchema, dict):
                    param_schema = self._mcp_schema_to_gemini_schema(mcp_tool.inputSchema)
                    declarations.append(glm.FunctionDeclaration(
                        name=mcp_tool.name,
                        description=mcp_tool.description or '',
                        parameters=param_schema
                    ))
                else:
                    logging.warning(f"Skipping tool {mcp_tool.name} due to non-dict inputSchema: {type(mcp_tool.inputSchema)}")

            if declarations:
                self.gemini_tools = declarations

            self.connected = True
            
            # Create response with tools information
            available_tools = [
                ToolInfo(name=tool.name, description=tool.description) 
                for tool in tools
            ]
            
            return ConnectionResponse(
                status="success",
                connected=True,
                available_tools=available_tools,
                message="Successfully connected to MCP server"
            )
            
        except Exception as e:
            logging.error(f"Failed during connection or tool processing: {e}", exc_info=True)
            await self.cleanup()
            self.connected = False
            return ConnectionResponse(
                status="error",
                connected=False,
                message=f"Connection error: {str(e)}"
            )

    async def cleanup(self):
        """Properly clean up the session and streams"""
        logging.info("Attempting to clean up MCP client resources...")
        if self.exit_stack:
            try:
                await self.exit_stack.aclose()
                logging.info("AsyncExitStack successfully closed contexts.")
            except RuntimeError as re:
                if "Attempted to exit cancel scope" in str(re):
                    logging.warning(f"Known issue: Caught RuntimeError during cleanup: {re}")
                else:
                    logging.error(f"Caught unexpected RuntimeError during cleanup: {re}", exc_info=True)
            except Exception as e:
                logging.error(f"Caught general Exception during AsyncExitStack cleanup: {e}", exc_info=True)
            finally:
                self.session = None
                self._streams_context = None
                self._session_context = None
                self.connected = False
                self.mcp_tools = []
                self.gemini_tools = []
                self.exit_stack = AsyncExitStack()
                logging.info("Client state reset after cleanup attempt.")
        else:
            logging.warning("Cleanup called but exit_stack was None or already closed.")
            self.session = None
            self._streams_context = None
            self._session_context = None
            self.connected = False
            self.mcp_tools = []
            self.gemini_tools = []
            self.exit_stack = AsyncExitStack()

    async def process_query_gemini(self, query: str) -> str:
        """Processes a user query using Gemini and MCP tools."""
        if not self.session:
            return "Error: Not connected to MCP server."
        if not self.model:
            return "Error: Gemini model not initialized."

        # Custom prompt template
        prompt_template = """
        **Task:** You are NoteTaker, a personal meeting assistant. Your task is to take and store notes and answer the user queries from the past meeting notes.
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

        today = datetime.today().strftime('%Y-%m-%d')
        final_prompt = prompt_template.format(user_query=query, today=today)

        logging.info(f"Sending final prompt structure to Gemini (showing first 200 chars): '{final_prompt[:200]}...'")
        try:
            logging.debug(f"Attempting to send message with tools: {self.gemini_tools}")
            if not self.gemini_tools:
                logging.warning("No Gemini tools configured to send!")

            # Send the formatted prompt to Gemini
            response = await self.chat.send_message_async(final_prompt, tools=self.gemini_tools)
            logging.debug(f"Initial Gemini response received: {response.candidates[0].content if response.candidates else 'No candidates'}")

            # Check for function call request
            response_text = response.candidates[0].content.parts[0]
            parts = len(response.candidates[0].content.parts)
            
            if parts > 1:
                response_part = response.candidates[0].content.parts[1]
                fc = response_part.function_call
                tool_name = fc.name
                
                # Convert Struct to Python dict
                tool_args = {}
                for key, value in fc.args.items():
                    if hasattr(value, 'string_value'):
                        tool_args[key] = value.string_value
                    elif hasattr(value, 'number_value'):
                        tool_args[key] = value.number_value
                    elif hasattr(value, 'bool_value'):
                        tool_args[key] = value.bool_value
                    else:
                        tool_args[key] = str(value)  # Fallback

                logging.info(f"Gemini requested tool call: {tool_name}({tool_args})")

                # Find the corresponding MCP tool
                mcp_tool = next((t for t in self.gemini_tools if t.name == tool_name), None)
                if not mcp_tool:
                    logging.error(f"Gemini requested unknown tool: {tool_name}")
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

                        # Check if MCP result indicates an error
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

            # Extract final text response
            final_text = ""
            try:
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

# Create a client instance
mcp_client = MCPClient()

# FastAPI dependency for client state check
async def get_client():
    if not mcp_client.connected:
        raise HTTPException(status_code=400, detail="Not connected to MCP server. Connect first.")
    return mcp_client

# Define API routes
@app.get("/", response_model=StatusResponse)
async def get_status():
    """Get the current status of the MCP client connection"""
    tools = []
    if mcp_client.connected and mcp_client.mcp_tools:
        tools = [
            ToolInfo(name=tool.name, description=tool.description) 
            for tool in mcp_client.mcp_tools
        ]
    
    return StatusResponse(
        status="connected" if mcp_client.connected else "disconnected",
        connected=mcp_client.connected,
        available_tools=tools
    )

@app.post("/connect", response_model=ConnectionResponse)
async def connect(connection_req: ConnectionRequest, background_tasks: BackgroundTasks):
    """Connect to an MCP server at the specified URL"""
    if mcp_client.connected:
        # If already connected, clean up first
        background_tasks.add_task(mcp_client.cleanup)
        return ConnectionResponse(
            status="info", 
            connected=False,
            message="Disconnecting from current server and connecting to new one..."
        )
    
    result = await mcp_client.connect_to_sse_server(connection_req.server_url)
    return result

@app.post("/disconnect", response_model=StatusResponse)
async def disconnect():
    """Disconnect from the current MCP server"""
    if mcp_client.connected:
        await mcp_client.cleanup()
    
    return StatusResponse(
        status="disconnected",
        connected=False,
        available_tools=[]
    )

@app.post("/query", response_model=QueryResponse)
async def query(query_req: QueryRequest, client: MCPClient = Depends(get_client)):
    """Process a query using Gemini and available MCP tools"""
    response = await client.process_query_gemini(query_req.query)
    return QueryResponse(response=response)

# Add startup and shutdown events
@app.on_event("startup")
async def startup_event():
    logging.info("FastAPI server starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    logging.info("FastAPI server shutting down...")
    if mcp_client.connected:
        await mcp_client.cleanup()

if __name__ == "__main__":
    uvicorn.run("fastapp:app", host="0.0.0.0", port=8000, reload=True)