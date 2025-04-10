import os
import uuid
import datetime
import pickle
import asyncio
from typing import Any, List, Optional, Dict, Tuple
import numpy as np
import faiss # type: ignore
import httpx
import google.generativeai as genai # Import the library
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from mcp.server.sse import SseServerTransport
from starlette.requests import Request
from starlette.routing import Mount, Route
from mcp.server import Server
import uvicorn
from dotenv import load_dotenv
from dateutil import parser as date_parser
from dateutil.tz import gettz
from loguru import logger

# Load environment variables
load_dotenv()

# --- Configuration ---
FAISS_INDEX_PATH = "./faiss_notes.index"
METADATA_PATH = "./notes_metadata.pkl"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

# Configure the Google Generative AI Client
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    logger.info("Google Generative AI client configured successfully.")
except Exception as e:
    logger.error(f"Failed to configure Google Generative AI client: {e}", exc_info=True)
    raise

# Choose the Embedding Model
# Use a recent model. 'models/text-embedding-004' or 'models/embedding-001' are good choices.
# Older 'gecko' names usually refer to PaLM API models.
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
logger.info(f"Using embedding model: {EMBEDDING_MODEL_NAME}")

# Constants
SEARCH_K_MULTIPLIER = 5

# --- Initialize FAISS and Metadata Storage ---
logger.info("Initializing FAISS and Metadata storage...")

# Global variables for FAISS index and metadata
faiss_index: Optional[faiss.Index] = None
metadata_store: Dict[int, Dict[str, Any]] = {} # Maps FAISS index ID to { "document": str, "metadata": dict }
embedding_dimension: Optional[int] = None
data_lock = asyncio.Lock() # Lock for thread-safe access to index/metadata


async def generate_embedding(text: str, task_type: str) -> Optional[List[float]]:
    """Generates embedding using the configured Google model."""
    try:
        # task_type should be "RETRIEVAL_DOCUMENT" or "RETRIEVAL_QUERY"
        result = await genai.embed_content_async(
            model=EMBEDDING_MODEL_NAME,
            content=text,
            task_type=task_type)
        return result['embedding']
    except Exception as e:
        logger.error(f"Failed to generate embedding for task '{task_type}': {e}", exc_info=True)
        return None

async def get_embedding_dimension() -> int:
    """Gets the embedding dimension by embedding a dummy text."""
    logger.info("Determining embedding dimension...")
    try:
        # Use RETRIEVAL_DOCUMENT as a default task type for dimension check
        dummy_embedding = await generate_embedding("test", task_type="RETRIEVAL_DOCUMENT")
        if dummy_embedding:
            dim = len(dummy_embedding)
            logger.info(f"Detected embedding dimension: {dim}")
            return dim
        else:
            raise ValueError("Failed to generate dummy embedding.")
    except Exception as e:
        logger.error(f"Failed to determine embedding dimension: {e}. Defaulting to 768.", exc_info=True)
        # Fallback dimension for text-embedding-004 is 768
        return 768

async def load_data():
    """Loads FAISS index and metadata from disk."""
    global faiss_index, metadata_store, embedding_dimension
    async with data_lock:
        if embedding_dimension is None:
             # Determine dimension asynchronously
             embedding_dimension = await get_embedding_dimension()

        if os.path.exists(FAISS_INDEX_PATH):
            try:
                logger.info(f"Loading FAISS index from {FAISS_INDEX_PATH}")
                print(f"Loading FAISS index from {FAISS_INDEX_PATH}")


                faiss_index = faiss.read_index(FAISS_INDEX_PATH)
                # Verify dimension compatibility
                if faiss_index.d != embedding_dimension:
                     logger.warning(f"Loaded index dimension ({faiss_index.d}) differs from expected ({embedding_dimension}). Re-initializing.")
                     print(f"Loaded index dimension ({faiss_index.d}) differs from expected ({embedding_dimension}). Re-initializing.")
                     faiss_index = None # Force re-initialization
                else:
                     logger.info(f"FAISS index loaded with {faiss_index.ntotal} vectors.")
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}. Starting fresh.", exc_info=True)
                faiss_index = None
        else:
            logger.info("FAISS index file not found. Initializing new index.")
            faiss_index = None

        if os.path.exists(METADATA_PATH):
            try:
                logger.info(f"Loading metadata from {METADATA_PATH}")
                print(f"Loading metadata from {METADATA_PATH}")

                with open(METADATA_PATH, "rb") as f:
                    metadata_store = pickle.load(f)
                logger.info(f"Metadata loaded for {len(metadata_store)} entries.")
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}. Starting fresh.", exc_info=True)
                metadata_store = {}
        else:
             logger.info("Metadata file not found. Initializing empty store.")
             metadata_store = {}

        # Initialize FAISS index if it wasn't loaded or dimension mismatch
        if faiss_index is None:
             logger.info(f"Creating new FAISS index (IndexFlatL2) with dimension {embedding_dimension}.")
             faiss_index = faiss.IndexFlatL2(embedding_dimension)


async def save_data():
    """Saves FAISS index and metadata to disk."""
    global faiss_index, metadata_store
    async with data_lock:
        if faiss_index is None:
            logger.warning("Attempted to save data, but FAISS index is not initialized.")
            return

        try:
            logger.info(f"Saving FAISS index to {FAISS_INDEX_PATH} ({faiss_index.ntotal} vectors)")
            faiss.write_index(faiss_index, FAISS_INDEX_PATH)
            logger.info("FAISS index saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}", exc_info=True)

        try:
            logger.info(f"Saving metadata to {METADATA_PATH} ({len(metadata_store)} entries)")
            with open(METADATA_PATH, "wb") as f:
                pickle.dump(metadata_store, f)
            logger.info("Metadata saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}", exc_info=True)

# --- Initialize FastMCP server ---
mcp = FastMCP("note_taker_faiss_gemini_embed") # Changed name slightly

# --- Helper Functions (Date parsing remains the same) ---
def parse_date_string(date_str: Optional[str]) -> datetime.date:
    """Parses a date string (like 'today', 'yesterday', '2023-10-26') into a date object."""
    if not date_str:
        return datetime.date.today()
    try:
        dt = date_parser.parse(date_str, fuzzy=True, default=datetime.datetime.now(gettz()))
        return dt.date()
    except (ValueError, OverflowError):
        logger.warning(f"Could not parse date string: '{date_str}'. Defaulting to today.")
        return datetime.date.today()

def format_date_for_storage(date_obj: datetime.date) -> str:
    """Formats date object into a consistent string format for metadata."""
    return date_obj.isoformat() # YYYY-MM-DD

def parse_stored_date(date_str: str) -> Optional[datetime.date]:
    """Parses the stored date string back into a date object."""
    try:
        return datetime.date.fromisoformat(date_str)
    except ValueError:
        return None

# --- MCP Tools ---

@mcp.tool()
async def add_note(content: str, meeting_date: str = None) -> str:
    """
    Adds a new minutes of meetings (MoM) to the knowledge base using FAISS and Google Embeddings.

    Args:
        content: The text content of the Mom;s or notes.
        date: Meeting date in '2024-07-15' format. If no date is mentioned then send Today's date.
    """
    global faiss_index, metadata_store
    if faiss_index is None or embedding_dimension is None:
         logger.error("FAISS index not initialized. Cannot add note.")
         return "Error: Knowledge base index is not ready."

    try:
        resolved_date = parse_date_string(meeting_date)
        iso_date_str = format_date_for_storage(resolved_date)

        logger.info(f"Adding note for date: {iso_date_str} (parsed from '{meeting_date}')")

        content=f'Meeting Date: {meeting_date} \n {content}'
        # 1. Generate Embedding using Google API
        logger.debug(f"Generating embedding for document: '{content[:50]}...'")
        embedding_list = await generate_embedding(content, task_type="RETRIEVAL_DOCUMENT")
        if not embedding_list:
             logger.error("Failed to generate embedding for the note content.")
             return "Error: Could not generate embedding for the note."

        vector = np.array(embedding_list).astype('float32').reshape(1, -1)
        logger.debug(f"Embedding generated with shape: {vector.shape}")
        if vector.shape[1] != embedding_dimension:
             logger.error(f"Generated embedding dimension ({vector.shape[1]}) does not match index dimension ({embedding_dimension}).")
             return "Error: Embedding dimension mismatch."

        # 2. Add to FAISS index and Metadata store (under lock)
        async with data_lock:
            if faiss_index is None:
                 logger.error("FAISS index became None unexpectedly.")
                 return "Error: Knowledge base index is unavailable."

            faiss_id = faiss_index.ntotal
            faiss_index.add(vector)
            logger.info(f"Vector added to FAISS index with ID: {faiss_id}. New total: {faiss_index.ntotal}")

            note_metadata = {
                "date": meeting_date,
                "original_date_mention": meeting_date if meeting_date else "defaulted_today",
                "insertion_timestamp": datetime.datetime.now().isoformat()
            }
            metadata_store[faiss_id] = {
                "document": content,
                "metadata": note_metadata
            }
            logger.info(f"Metadata stored for FAISS ID: {faiss_id}")

        # 3. Persist changes
        await save_data()

        return f"Successfully added note for {resolved_date.strftime('%B %d, %Y')} (Internal ID: {faiss_id})."

    except Exception as e:
        logger.error(f"Error adding note: {e}", exc_info=True)
        return f"Error adding note: {str(e)}"


@mcp.tool()
async def query_notes(query: str, date_filter: Optional[str] = None, num_results: int = 5) -> str:
    """
    Searches the meeting notes knowledge base (FAISS + Google Embeddings) for relevant information.
    Optionally filters results by a specific date or relative date.

    Args:
        query: The question or topic to search for in the notes.
        date_filter: Optional date string (e.g., 'today', '2024-07-15') to filter notes by.
        num_results: The maximum number of relevant note snippets to return.
    """
    global faiss_index, metadata_store
    if faiss_index is None or embedding_dimension is None or faiss_index.ntotal == 0:
        logger.info("Query attempted but FAISS index is empty or not initialized.")
        return "No notes found in the knowledge base yet."

    try:
        logger.info(f"Querying notes with query: '{query}', date_filter: '{date_filter}', num_results: {num_results}")

        # 1. Generate Query Embedding using Google API
        logger.debug("Generating embedding for query...")
        query_embedding_list = await generate_embedding(query, task_type="RETRIEVAL_QUERY")
        if not query_embedding_list:
            logger.error("Failed to generate embedding for the query.")
            return "Error: Could not process the query."

        query_vector = np.array(query_embedding_list).astype('float32').reshape(1, -1)
        logger.debug(f"Query embedding generated with shape: {query_vector.shape}")
        if query_vector.shape[1] != embedding_dimension:
             logger.error(f"Query embedding dimension ({query_vector.shape[1]}) does not match index dimension ({embedding_dimension}).")
             return "Error: Query embedding dimension mismatch."

        # 2. Perform FAISS Search
        k_to_search = max(num_results * SEARCH_K_MULTIPLIER, 10)
        logger.debug(f"Searching FAISS index for {k_to_search} nearest neighbors...")

        distances, indices = faiss_index.search(query_vector, k_to_search)

        if indices.size == 0 or indices[0][0] == -1:
            logger.info("FAISS search returned no results.")
            return "No relevant notes found for your query."

        logger.debug(f"FAISS search returned {len(indices[0])} potential candidates.")

        # 3. Filter and Format Results
        filtered_results: List[Tuple[float, int, Dict[str, Any]]] = []
        target_iso_date_str = None
        if date_filter:
            resolved_date = parse_date_string(date_filter)
            target_iso_date_str = format_date_for_storage(resolved_date)
            logger.info(f"Applying date filter for: {target_iso_date_str}")

        async with data_lock: # Lock ensures metadata_store isn't modified while iterating
            for i, faiss_id in enumerate(indices[0]):
                if faiss_id == -1:
                    continue
                if faiss_id in metadata_store:
                    note_data = metadata_store[faiss_id]
                    metadata = note_data["metadata"]
                    distance = distances[0][i] # L2 distance

                    if target_iso_date_str:
                        if metadata.get("date") == target_iso_date_str:
                            filtered_results.append((distance, faiss_id, note_data))
                    else:
                        filtered_results.append((distance, faiss_id, note_data))
                else:
                     logger.warning(f"FAISS ID {faiss_id} found in index but not in metadata store. Skipping.")

        filtered_results.sort(key=lambda x: x[0])
        final_results = filtered_results[:num_results]

        if not final_results:
             logger.info("No notes found matching the query and date filter criteria.")
             return "No relevant notes found matching your criteria."

        # Format the final results
        formatted_output = []
        for dist, faiss_id, note_data in final_results:
            doc = note_data["document"]
            meta = note_data["metadata"]
            note_date = parse_stored_date(meta.get("date", ""))
            date_str = note_date.strftime('%B %d, %Y') if note_date else "Unknown Date"
            formatted_output.append(
                f"Note from {date_str} (FAISS ID: {faiss_id}, Distance: {dist:.4f}):\n"
                f"{doc}\n"
                f"---"
            )

        logger.info(f"Returning {len(formatted_output)} formatted results.")
        return "\n".join(formatted_output)

    except Exception as e:
        logger.error(f"Error querying notes: {e}", exc_info=True)
        return f"Error querying notes: {str(e)}"


# --- Starlette App Setup (Identical structure, includes load_data) ---
def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """Create a Starlette application that can server the provied mcp server with SSE."""
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> None:
        client_host = request.client.host if request.client else "unknown"
        logger.info(f"SSE connection attempt from {client_host}")
        try:
            async with sse.connect_sse(
                    request.scope,
                    request.receive,
                    request._send,  # noqa: SLF001
            ) as (read_stream, write_stream):
                logger.info(f"SSE connection established with {client_host}")
                await mcp_server.run(
                    read_stream,
                    write_stream,
                    mcp_server.create_initialization_options(),
                )
            logger.info(f"SSE connection closed for {client_host}")
        except Exception as e:
            logger.error(f"Error during SSE handling for {client_host}: {e}", exc_info=True)

    async def handle_post(request: Request) -> None:
        client_host = request.client.host if request.client else "unknown"
        logger.info(f"POST request to /messages/ from {client_host}")
        await sse.handle_post_message(request)

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            # Route("/messages/", endpoint=handle_post, methods=["POST"]),
            Mount("/messages/", app=sse.handle_post_message),
        ],
        on_startup=[load_data], # Load data when the server starts
    )

# --- Main Execution ---
if __name__ == "__main__":
    mcp_server = mcp._mcp_server  # noqa: WPS437

    import argparse

    parser = argparse.ArgumentParser(description='Run Note Taker MCP SSE server (FAISS + Google Embeddings)')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080, help='Port to listen on')
    args = parser.parse_args()

    logger.add("mcp_server_faiss_gemini_notes.log", rotation="10 MB") # Log to file
    logger.info(f"Starting Note Taker MCP Server (FAISS + Google Embeddings) on {args.host}:{args.port}")

    starlette_app = create_starlette_app(mcp_server, debug=False)

    uvicorn.run(starlette_app, host=args.host, port=args.port, log_level="info")