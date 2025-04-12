# server.py
import asyncio
import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from mcp.server.fastmcp import FastMCP
import mcp.types as types
import logging
from typing import List, Optional
import ssl # <-- Import ssl

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - SERVER - %(levelname)s - %(message)s')

# Initialize FastMCP server
mcp = FastMCP("web_search_crawler")

# --- Helper Function for Crawling ---
async def fetch_and_extract_text(url: str, client: httpx.AsyncClient, timeout: float = 10.0) -> Optional[str]:
    """Fetches content from a URL and extracts text using BeautifulSoup."""
    try:
        logging.info(f"Attempting to crawl: {url}")
        # Note: httpx client passed in already has verify=False set
        response = await client.get(url, timeout=timeout, follow_redirects=True)
        response.raise_for_status() # Raise an exception for bad status codes

        # ... (rest of the function remains the same) ...

    except httpx.RequestError as e:
        # Check specifically for SSL errors if possible (though verify=False should prevent them)
        if isinstance(e.__cause__, ssl.SSLError):
             logging.error(f"SSL Error fetching {url} (even with verification off?): {e}")
        else:
             logging.error(f"HTTP Request error fetching {url}: {e}")
        return None
    # ... (rest of the error handling remains the same) ...


# --- MCP Tool Implementation ---
@mcp.tool()
async def web_search_and_crawl(query: str) -> str:
    """
    USE THIS TOOL to find information on the internet.
    Performs a web search using DuckDuckGo for the given query, retrieves the content
    of the top 3 relevant web pages, and returns their combined text.
    Essential for current events, real-time data, recent developments,
    or any topic requiring up-to-date information from the web.

    Args:
        query: The search query string (e.g., "latest tesla stock price", "summary of WWDC 2024").
    """
    logging.info(f"Received search query: {query}")
    search_results = []
    snipets=[]
    combined_text = ""
    max_results = 3

    try:
        # Use DDGS context manager for potential cleanup
        with DDGS(verify=False) as ddgs:
            # Perform the search
            search_results = list(ddgs.text(query, max_results=max_results))

        if not search_results:
            logging.warning(f"No search results found for query: {query}")
            return "No search results found."

        logging.info(f"Found {len(search_results)} results. Attempting to crawl top {max_results}.")

        # snipets=[result['snippet'] for result in search_results[:max_results]]
        # Prepare URLs to crawl
        urls_to_crawl = [result['href'] for result in search_results[:max_results]]
        return search_results

        # Crawl concurrently
        # !!--- MODIFICATION HERE ---!!
        # Disable SSL verification in the AsyncClient
        logging.warning("SSL certificate verification is DISABLED for HTTPX client.")
        # async with httpx.AsyncClient(
        #     headers={'User-Agent': 'MCP-Crawler-Bot/1.0'},
        #     verify=False # <--- ADD THIS LINE TO DISABLE VERIFICATION
        # ) as client:
        #     tasks = [fetch_and_extract_text(url, client) for url in urls_to_crawl]
        #     crawl_results = await asyncio.gather(*tasks)

        # ... (rest of the function remains the same) ...

    except Exception as e:
        logging.exception(f"An error occurred during search or crawl for query '{query}': {e}")
        return f"An error occurred: {e}"

# --- Run the Server ---
if __name__ == "__main__":
    logging.info("Starting MCP Server...")
    mcp.run(transport='stdio')
    logging.info("MCP Server stopped.")