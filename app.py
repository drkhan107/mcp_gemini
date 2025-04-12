import streamlit as st
import requests
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

API_BASE_URL = "http://localhost:8000"  # Default FastAPI server URL

# Set page config
st.set_page_config(
    page_title="MCP Client UI",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "connected" not in st.session_state:
    st.session_state.connected = False
if "available_tools" not in st.session_state:
    st.session_state.available_tools = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "connection_url" not in st.session_state:
    st.session_state.connection_url = ""


def check_server_status():
    """Check the status of the FastAPI server and MCP connection"""
    try:
        response = requests.get(f"{API_BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            st.session_state.connected = data["connected"]
            st.session_state.available_tools = data["available_tools"]
            return True
        return False
    except requests.exceptions.RequestException:
        return False


def connect_to_mcp(server_url: str):
    """Connect to the MCP server"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/connect",
            json={"server_url": server_url}
        )
        data = response.json()
        
        if response.status_code == 200:
            st.session_state.connected = data["connected"]
            st.session_state.available_tools = data["available_tools"]
            st.session_state.connection_url = server_url
            
            if data["connected"]:
                st.success(f"Connected to MCP server: {server_url}")
                return True
            else:
                st.warning(f"Connection in progress: {data.get('message', 'No details provided')}")
                return False
        else:
            st.error(f"Failed to connect: {data.get('message', 'Unknown error')}")
            return False
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API: {str(e)}")
        return False


def disconnect_from_mcp():
    """Disconnect from the MCP server"""
    try:
        response = requests.post(f"{API_BASE_URL}/disconnect")
        if response.status_code == 200:
            st.session_state.connected = False
            st.session_state.available_tools = []
            st.session_state.connection_url = ""
            st.success("Disconnected from MCP server")
            return True
        return False
    except requests.exceptions.RequestException as e:
        st.error(f"Error disconnecting: {str(e)}")
        return False


def send_query(query: str):
    """Send a query to the MCP server via Gemini"""
    if not st.session_state.connected:
        st.warning("Not connected to MCP server. Connect first.")
        return None
    
    try:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": query, "timestamp": datetime.now().strftime("%H:%M:%S")})
        
        with st.spinner("Generating response..."):
            response = requests.post(
                f"{API_BASE_URL}/query",
                json={"query": query}
            )
        
        if response.status_code == 200:
            data = response.json()
            assistant_response = data["response"]
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": assistant_response, 
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            
            return assistant_response
        else:
            error_msg = f"Error: {response.status_code} - {response.text}"
            st.session_state.chat_history.append({
                "role": "system", 
                "content": error_msg, 
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            return None
    except requests.exceptions.RequestException as e:
        error_msg = f"Connection error: {str(e)}"
        st.session_state.chat_history.append({
            "role": "system", 
            "content": error_msg, 
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        return None


# Check if FastAPI server is running on startup
server_running = check_server_status()

# App title and description
st.title("📝 MCP NoteTaker Client")
st.markdown("""
This interface connects to a Machine Common Protocol (MCP) server
to take meeting notes and retrieve information using Google's Gemini AI.
""")

# Display server status
if server_running:
    st.success("✅ Connected to FastAPI server")
else:
    st.error("❌ Cannot connect to FastAPI server. Make sure it's running at " + API_BASE_URL)

# Sidebar for connection controls
with st.sidebar:
    st.header("Connection Settings")
    
    # Show current connection status
    if st.session_state.connected:
        st.info(f"Connected to: {st.session_state.connection_url}")
        if st.button("Disconnect"):
            disconnect_from_mcp()
    else:
        st.warning("Not connected to MCP server")
        
        # Connection form
        with st.form("connect_form"):
            server_url = st.text_input(
                "MCP Server URL", 
                value="http://localhost:8080/sse",
                help="The MCP server URL with SSE endpoint"
            )
            
            submit_button = st.form_submit_button("Connect")
            if submit_button:
                connect_to_mcp(server_url)
    
    # Display available tools when connected
    if st.session_state.connected and st.session_state.available_tools:
        st.header("Available Tools")
        for tool in st.session_state.available_tools:
            with st.expander(tool["name"]):
                st.write(tool.get("description", "No description available"))

# Main chat interface
st.header("NoteTaker Chat")

# Display chat history
chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"""
            <div style="display: flex; justify-content: flex-end;">
                <div style="background-color: #e6f7ff; padding: 10px; border-radius: 10px; max-width: 80%;">
                    <p style="margin: 0; color: #333;"><strong>You</strong> <span style="color: #888; font-size: 0.8em;">{message["timestamp"]}</span></p>
                    <p style="margin: 0;">{message["content"]}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif message["role"] == "assistant":
            st.markdown(f"""
            <div style="display: flex; justify-content: flex-start;">
                <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; max-width: 80%;">
                    <p style="margin: 0; color: #333;"><strong>NoteTaker</strong> <span style="color: #888; font-size: 0.8em;">{message["timestamp"]}</span></p>
                    <p style="margin: 0;">{message["content"]}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:  # system messages
            st.markdown(f"""
            <div style="display: flex; justify-content: center;">
                <div style="background-color: #ffe6e6; padding: 5px; border-radius: 10px; max-width: 80%;">
                    <p style="margin: 0; color: #cc0000; font-size: 0.9em;">{message["content"]}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Add some spacing
    st.write("")
    st.write("")

# Input for new query
if st.session_state.connected:
    query = st.chat_input("Ask NoteTaker something...")
    if query:
        send_query(query)
        st.rerun()  # Rerun to update the UI with new messages
else:
    st.info("Connect to an MCP server first to start chatting.")

# Example queries section
with st.expander("Example Queries"):
    examples = [
        "Take notes from today's meeting: The team discussed the Q2 roadmap and decided to prioritize the mobile app redesign.",
        "What was discussed in yesterday's meeting?",
        "Store this MoM: Marketing team agreed on launching the campaign on May 15th.",
        "Find all notes related to product launch discussions."
    ]
    
    for example in examples:
        if st.button(example, key=f"example_{hash(example)}"):
            send_query(example)
            st.rerun()

# Footer
st.markdown("---")
st.caption("MCP NoteTaker Client - Powered by Gemini AI")