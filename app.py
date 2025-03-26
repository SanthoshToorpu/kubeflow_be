from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pymilvus import MilvusClient
from groq import Groq
import os
import logging
import json
import uvicorn
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Milvus client
milvus_client = MilvusClient(
    uri=os.getenv("MILVUS_URI"),
    token=os.getenv("MILVUS_TOKEN")
)

# Initialize Groq client
groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)

# Constants
COLLECTION_NAME = "content_embeddings"
EMBEDDING_DIM = 1536
LLM_MODEL = "llama-3.3-70b-versatile"

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connection manager for WebSockets
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected: {id(websocket)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"Client disconnected: {id(websocket)}")

    async def send_message(self, websocket: WebSocket, message: Dict):
        await websocket.send_json(message)

manager = ConnectionManager()

# RAG functions
def generate_query_embedding(query: str) -> list[float]:
    """Generate embedding for a query"""
    logger.info(f"Generating embedding for query: {query[:30]}...")
    
    try:
        # Use Groq for embeddings
        response = groq_client.embeddings.create(
            model="text-embedding-3-small",
            input=[query]
        )
        
        embedding = response.data[0].embedding
        
        if len(embedding) != EMBEDDING_DIM:
            logger.warning(f"Embedding dimension mismatch. Expected {EMBEDDING_DIM}, got {len(embedding)}")
        
        logger.info(f"Successfully generated embedding with dimension {len(embedding)}")
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        # Fallback to simpler embedding if needed
        # This is just a placeholder - in production you'd want a proper fallback
        import random
        logger.warning("Using fallback random embedding - for testing only!")
        return [random.uniform(-1, 1) for _ in range(EMBEDDING_DIM)]

def search_vector_store(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """Search the vector store for similar content"""
    logger.info(f"Searching vector store for top {top_k} results...")
    
    # Perform the search
    results = milvus_client.search(
        collection_name=COLLECTION_NAME,
        data=[query_embedding],
        anns_field="embedding",
        limit=top_k,
        output_fields=["content", "source_file", "chunk_index"]
    )
    
    # Format results
    formatted_results = [
        {
            "content": hit["entity"]["content"],
            "source_file": hit["entity"]["source_file"],
            "chunk_index": hit["entity"]["chunk_index"],
            "score": hit["distance"]
        }
        for hit in results[0]
    ]
    
    logger.info(f"Found {len(formatted_results)} results")
    return formatted_results

# RAG tool for LLM to call
def rag_tool(query: str) -> str:
    """
    Retrieve relevant information from the knowledge base based on the query.
    
    Args:
        query: The search query to find relevant information
        
    Returns:
        String containing retrieved context from the knowledge base
    """
    try:
        # Generate query embedding
        query_embedding = generate_query_embedding(query)
        
        # Search vector store
        results = search_vector_store(query_embedding)
        
        # Extract context for response
        context = "\n\n".join([f"From {r['source_file']}:\n{r['content']}" for r in results])
        
        return context
    except Exception as e:
        logger.error(f"Error in RAG tool: {str(e)}")
        return f"Error retrieving information: {str(e)}"

# Function to extract tool calls from text response
def extract_function_calls(text: str) -> List[Dict]:
    """Extract function calls from text response if Groq returns them as text"""
    pattern = r'<function=(\w+)\s+({.*?})\s*</function>'
    matches = re.findall(pattern, text)
    
    function_calls = []
    for function_name, args_str in matches:
        try:
            args = json.loads(args_str)
            function_calls.append({
                "function_name": function_name,
                "arguments": args
            })
        except json.JSONDecodeError:
            logger.error(f"Could not parse function arguments: {args_str}")
    
    return function_calls

# System prompt for the LLM
SYSTEM_PROMPT = """You are a helpful AI assistant with access to a RAG (Retrieval-Augmented Generation) system.
You can retrieve information from a knowledge base using the retrieve_information function.
When a user asks a question, first decide if you need to search for information.
If the question requires specific knowledge that you're not completely sure about, use the retrieve_information function to get relevant context.
Base your answers on the retrieved information and cite sources when appropriate.
If no relevant information is retrieved, or if the question doesn't require searching, answer based on your general knowledge.
Keep your responses concise, accurate, and helpful.

IMPORTANT: DO NOT output the function call format in your response to the user. The system will handle function calls automatically.
Always provide a natural, conversational response to the user based on the information retrieved.
"""

async def generate_llm_response(query: str, conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Generate a response from the LLM using RAG as a tool when needed.
    
    Args:
        query: User's question
        conversation_history: Previous messages in the conversation
        
    Returns:
        Dictionary with the response and any additional metadata
    """
    logger.info(f"Generating LLM response for: {query[:50]}...")
    
    if conversation_history is None:
        conversation_history = []
    
    # Prepare the messages for the LLM
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    
    # Add conversation history
    for msg in conversation_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add the user's current query
    messages.append({"role": "user", "content": query})
    
    try:
        # First attempt: Try with function calling
        tool_calls_results = []
        
        # Send initial request to LLM
        response = groq_client.chat.completions.create(
            messages=messages,
            model=LLM_MODEL,
            temperature=0.2,  # Lower temperature for more deterministic responses
            max_tokens=1000
        )
        
        result_content = response.choices[0].message.content
        
        # Check if the response contains function call syntax
        function_calls = extract_function_calls(result_content)
        
        if function_calls:
            logger.info(f"Detected function calls in text: {function_calls}")
            
            # Process each function call
            for func_call in function_calls:
                if func_call["function_name"] == "retrieve_information":
                    search_query = func_call["arguments"].get("query", query)
                    
                    # Call the RAG tool
                    retrieval_result = rag_tool(search_query)
                    tool_calls_results.append({"query": search_query, "result": retrieval_result})
                    
                    # Add the additional context to messages
                    messages.append({
                        "role": "user", 
                        "content": f"Here's additional information I found about '{search_query}':\n\n{retrieval_result}\n\nPlease use this information to provide a comprehensive and accurate answer to my original question: {query}"
                    })
            
            # Get the final response with the retrieved information
            final_response = groq_client.chat.completions.create(
                messages=messages,
                model=LLM_MODEL,
                temperature=0.3,
                max_tokens=1000
            )
            
            # Clean up any remaining function call syntax
            response_text = re.sub(r'<function=.*?</function>', '', final_response.choices[0].message.content)
            response_text = response_text.strip()
            
            return {
                "response": response_text,
                "tool_calls": tool_calls_results
            }
        
        # If no function calls detected, check if we should retrieve information anyway
        # This handles cases where the model might need information but didn't use function syntax
        if any(keyword in query.lower() for keyword in ["what is", "how does", "explain", "describe", "define"]):
            # For questions that likely need factual information
            if not any(keyword in result_content.lower() for keyword in ["i don't know", "cannot provide", "don't have information"]):
                # If the model seems confident, just return its response
                return {
                    "response": result_content,
                    "tool_calls": []
                }
            
            # If model seems uncertain, use RAG
            retrieval_result = rag_tool(query)
            tool_calls_results.append({"query": query, "result": retrieval_result})
            
            # Add the additional context to messages
            messages.append({
                "role": "user", 
                "content": f"Here's additional information I found:\n\n{retrieval_result}\n\nPlease use this information to provide a comprehensive answer to my original question: {query}"
            })
            
            # Get the final response with the retrieved information
            final_response = groq_client.chat.completions.create(
                messages=messages,
                model=LLM_MODEL,
                temperature=0.3,
                max_tokens=1000
            )
            
            return {
                "response": final_response.choices[0].message.content,
                "tool_calls": tool_calls_results
            }
        
        # If no special handling needed, return the original response
        return {
            "response": result_content,
            "tool_calls": []
        }
            
    except Exception as e:
        logger.error(f"Error generating LLM response: {str(e)}")
        return {
            "response": f"I'm sorry, I encountered an error: {str(e)}",
            "error": str(e)
        }

@app.get("/")
async def root():
    return {"message": "RAG WebSocket Server running with Groq LLM!"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    # Store conversation history
    conversation_history = []
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            logger.info(f"Received message from client {id(websocket)}")
            
            try:
                # Parse the message
                message_data = json.loads(data)
                query = message_data.get("query", "")
                
                if not query:
                    await manager.send_message(websocket, {
                        "type": "error",
                        "message": "No query provided"
                    })
                    continue
                
                # Process the query with LLM using RAG
                llm_response = await generate_llm_response(query, conversation_history)
                
                # Add to conversation history
                conversation_history.append({"role": "user", "content": query})
                conversation_history.append({"role": "assistant", "content": llm_response["response"]})
                
                # Keep conversation history to a reasonable size
                if len(conversation_history) > 10:
                    conversation_history = conversation_history[-10:]
                
                # Send response back to client
                await manager.send_message(websocket, {
                    "type": "response",
                    "query": query,
                    "response": llm_response["response"],
                    "tool_calls": llm_response.get("tool_calls", [])
                })
                
            except json.JSONDecodeError:
                # If not JSON, treat as plain text query
                query = data
                
                # Process the query with LLM using RAG
                llm_response = await generate_llm_response(query, conversation_history)
                
                # Add to conversation history
                conversation_history.append({"role": "user", "content": query})
                conversation_history.append({"role": "assistant", "content": llm_response["response"]})
                
                # Send response back to client
                await manager.send_message(websocket, {
                    "type": "response",
                    "query": query,
                    "response": llm_response["response"],
                    "tool_calls": llm_response.get("tool_calls", [])
                })
                
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                await manager.send_message(websocket, {
                    "type": "error",
                    "message": f"Error processing message: {str(e)}"
                })
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    logger.info("Starting RAG WebSocket server with Groq...")
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", 5000)))
