# Model Context Protocol (MCP)

A working demo of MCP integrated with Google's Gemini.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```
### 2. Set up environment variables
Create a .env file in the root directory and add your Google API key:

GOOGLE_API_KEY="your_api_key_here"

### 3. 📦 Install Dependencies
Install all required packages from requirements.txt:

```bash
pip install -r requirements.txt
```

### 4. 🖥️ Run the MCP Server
Start the MCP server:

```bash
python sse_server.py
```
✅ This will start the MCP server at the configured port (default is http://localhost:8080/sse).

### 5. 🧠 Start the MCP Client
Once the server is running, start the SSE client with the server URL:

```bash

python ssc_client.py http://localhost:8080/sse
```

✅ Done!
You now have a working demo of the Model Context Protocol with Gemini.

