# RAG WebSocket Server with LLM Integration

A FastAPI-based WebSocket server that provides RAG (Retrieval-Augmented Generation) capabilities with LLM integration, allowing for intelligent chat-based interaction with a vector database.

## Features

- WebSocket interface for real-time communication
- RAG system that retrieves relevant information from a vector database
- LLM integration that uses RAG as a tool for generating responses
- Conversation history management
- Easy deployment with ngrok

## Setup

1. Install dependencies:
   ```
   pip install -r ../requirements.txt
   ```

2. Make sure your `.env` file contains the required environment variables:
   ```
   AZURE_OPENAI_API_KEY=your_openai_api_key
   AZURE_OPENAI_ENDPOINT=your_openai_endpoint
   ```

## Running the Server

1. Start the WebSocket server:
   ```
   python app.py
   ```
   This will start the server on http://localhost:5000

## Deploying with Ngrok

1. Install Ngrok if you haven't already: https://ngrok.com/download

2. Run Ngrok to expose the WebSocket server:
   ```
   ngrok http 5000
   ```

3. Ngrok will provide a public URL that you can use to connect to your WebSocket server from anywhere.

## Connecting from Frontend

You can connect to this WebSocket server using standard WebSocket libraries in your frontend.

JavaScript example:
```javascript
// Connect to the WebSocket server
const socket = new WebSocket("ws://localhost:5000/ws"); // or your ngrok URL with ws:// protocol

// Connection opened
socket.addEventListener("open", (event) => {
  console.log("Connected to server");
});

// Listen for messages
socket.addEventListener("message", (event) => {
  const data = JSON.parse(event.data);
  console.log("Received response:", data);
});

// Listen for errors
socket.addEventListener("error", (event) => {
  console.error("WebSocket error:", event);
});

// Send a query
function sendQuery(question) {
  socket.send(JSON.stringify({
    query: question
  }));
}

// Example usage
sendQuery("What is Kubeflow?");
```

### Example using Socket.IO Client (will require additional server setup)

If you prefer Socket.IO, you'll need to add Socket.IO compatibility to the FastAPI server using socket.io-client libraries.

## Message Format

### Sending Questions

Send a JSON message with the query:
```json
{
  "query": "What is Kubeflow?"
}
```

### Receiving Responses

Responses will be in the following format:
```json
{
  "type": "response",
  "query": "What is Kubeflow?",
  "response": "Kubeflow is an open-source machine learning platform...",
  "tool_calls": [
    {
      "query": "Kubeflow",
      "result": "From documentation.txt:\nKubeflow is an open-source platform..."
    }
  ]
}
```

## How It Works

1. The server accepts WebSocket connections from clients
2. Clients send queries to the server
3. The server processes the query:
   - The LLM decides whether to use the RAG tool to retrieve information
   - If needed, the RAG tool is called to get relevant context from the vector database
   - The LLM generates a response using the retrieved context
4. The server sends the response back to the client
5. Conversation history is maintained for context 