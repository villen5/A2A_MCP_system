#!/usr/bin/env python
"""
BrightData MCP Agent Example

A Python implementation of an A2A agent with MCP tools using BrightData services.
This agent combines the power of GPT with BrightData tools for web scraping,
search engine results, and structured data extraction.

To run:
    export OPENAI_API_KEY=
    export API_TOKEN=
    python brightdata_mcp_agent.py

Requirements:
    pip install "python-a2a[openai,mcp,server]" requests
"""

import os
import sys
import time
import socket
import argparse
import multiprocessing
from datetime import datetime
import json

# Check for OpenAI API key
if not os.environ.get("OPENAI_API_KEY"):
    print("❌ OPENAI_API_KEY environment variable not set!")
    print("    export OPENAI_API_KEY=your_api_key")
    sys.exit(1)

# Check for BrightData API token
if not os.environ.get("API_TOKEN"):
    print("❌ API_TOKEN environment variable not set!")
    print("    export API_TOKEN=your_brightdata_api_token")
    sys.exit(1)

# Quick dependency check
try:
    import python_a2a
    import openai
    import flask
    import fastapi
    import uvicorn
    import requests
except ImportError as e:
    print(f"❌ Missing dependency: {e.name}")
    print("    pip install \"python-a2a[openai,mcp,server]\" requests")
    sys.exit(1)

from python_a2a import OpenAIA2AServer, A2AServer, run_server, TaskStatus, TaskState
from python_a2a import AgentCard, AgentSkill, A2AClient
from python_a2a.mcp import FastMCP, text_response, create_fastapi_app
from typing import Optional, List

# Find available port
def find_available_port(start_port=5002, max_tries=10):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_tries):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('localhost', port))
            sock.close()
            return port
        except OSError:
            continue
    return start_port + 100  # Return something far away as fallback

# Parse arguments
parser = argparse.ArgumentParser(description="BrightData MCP Agent Example")
parser.add_argument("--port", type=int, default=None, help="Agent port (default: auto)")
parser.add_argument("--mcp-port", type=int, default=None, help="MCP port (default: auto)")
parser.add_argument("--no-auto-mcp", action="store_true", help="Don't auto-start MCP server")
parser.add_argument("--no-test", action="store_true", help="Don't run test queries")
parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use")
args = parser.parse_args()

# Auto-select ports if not specified
if args.port is None:
    args.port = find_available_port()
    print(f"🔍 Auto-selected agent port: {args.port}")
else:
    print(f"🔍 Using specified agent port: {args.port}")

if args.mcp_port is None:
    args.mcp_port = find_available_port(args.port + 1)
    print(f"🔍 Auto-selected MCP port: {args.mcp_port}")
else:
    print(f"🔍 Using specified MCP port: {args.mcp_port}")

# Global variables
API_TOKEN = os.environ.get("API_TOKEN")
WEB_UNLOCKER_ZONE = os.environ.get("WEB_UNLOCKER_ZONE", "mcp_unlocker")


TOOL_PATTERNS = {
    "search_engine": {
        "patterns": [
            r"QUERY:\s*search\s+for\s+(.+)",
            r"search\s+for\s+(.+)",
            r"find\s+information\s+about\s+(.+)",
            r"look\s+up\s+(.+)"
        ],
        "priority": 10
    },
    "scrape_as_markdown": {
        "patterns": [
            r"QUERY:\s*scrape\s+(https?://\S+)",
            r"scrape\s+(https?://\S+)",
            r"extract\s+from\s+(https?://\S+)",
            r"get\s+content\s+from\s+(https?://\S+)"
        ],
        "priority": 20
    },
    "web_data_amazon_product": {
        "patterns": [
            r"QUERY:.*amazon.*(product|price).*?(https?://\S+amazon\S+)",
            r"amazon.*(product|price).*?(https?://\S+amazon\S+)",
            r"check.*(price|details).*amazon.*?(https?://\S+amazon\S+)"
        ],
        "priority": 30,
        "url_group": 2  # Which regex group contains the URL
    },
    "web_data_linkedin_person_profile": {
        "patterns": [
            r"QUERY:.*linkedin.*(person|profile).*?(https?://\S+linkedin.com\S+)",
            r"linkedin.*(person|profile).*?(https?://\S+linkedin.com\S+)"
        ],
        "priority": 30,
        "url_group": 2
    },
    "web_data_linkedin_company_profile": {
        "patterns": [
            r"QUERY:.*linkedin.*(company|organization).*?(https?://\S+linkedin.com\S+)",
            r"linkedin.*(company|organization).*?(https?://\S+linkedin.com\S+)"
        ],
        "priority": 30,
        "url_group": 2
    },
    "web_data_instagram_profiles": {
        "patterns": [
            r"QUERY:.*instagram.*(profile|account).*?(https?://\S+instagram.com\S+)",
            r"instagram.*(profile|account).*?(https?://\S+instagram.com\S+)"
        ],
        "priority": 30,
        "url_group": 2
    }
}





def get_api_headers():
    """Get standard headers for BrightData API calls"""
    return {
        'user-agent': 'brightdata_mcp_agent/1.0.0',
        'authorization': f'Bearer {API_TOKEN}'
    }

def ensure_required_zones():
    """Ensure the required zones exist in BrightData"""
    try:
        print('Checking for required zones...')
        response = requests.get(
            'https://api.brightdata.com/zone/get_active_zones',
            headers=get_api_headers()
        )
        
        zones = response.json() or []
        has_unlocker_zone = any(zone['name'] == WEB_UNLOCKER_ZONE for zone in zones)
        
        if not has_unlocker_zone:
            print(f'Required zone "{WEB_UNLOCKER_ZONE}" not found, creating it...')
            creation_response = requests.post(
                'https://api.brightdata.com/zone',
                headers={
                    **get_api_headers(),
                    'Content-Type': 'application/json'
                },
                json={
                    'zone': {
                        'name': WEB_UNLOCKER_ZONE,
                        'type': 'unblocker'
                    },
                    'plan': {
                        'type': 'unblocker'
                    }
                }
            )
            print(f'Zone "{WEB_UNLOCKER_ZONE}" created successfully')
        else:
            print(f'Required zone "{WEB_UNLOCKER_ZONE}" already exists')
    except Exception as e:
        error_info = getattr(e, 'response', {})
        error_data = error_info.json() if hasattr(error_info, 'json') else None
        print(f'Error checking/creating zones: {error_data or str(e)}')

def search_url(engine, query):
    """Generate search engine URL based on engine and query"""
    q = requests.utils.quote(query)
    if engine == 'yandex':
        return f'https://yandex.com/search/?text={q}'
    elif engine == 'bing':
        return f'https://www.bing.com/search?q={q}'
    return f'https://www.google.com/search?q={q}'

def start_mcp_server(port):
    """Start MCP server with BrightData tools"""
    # Create MCP server
    tools = FastMCP(name="BrightData Tools", description="Web scraping and data extraction tools", version="1.0.0")
    debug_stats = {'tool_calls': {}}
    
    # Ensure BrightData zones exist
    ensure_required_zones()
    
    # Tool function wrapper for logging and error handling
    def tool_fn(name, fn):
        async def wrapper(data):
            debug_stats['tool_calls'][name] = debug_stats['tool_calls'].get(name, 0) + 1
            ts = time.time()
            print(f"[{name}] executing {json.dumps(data)}")
            try:
                return await fn(data)
            except Exception as e:
                if hasattr(e, 'response'):
                    print(f"[{name}] error {e.response.status_code} {e.response.reason}: {e.response.text}")
                    message = e.response.text
                    if message:
                        raise Exception(f"HTTP {e.response.status_code}: {message}")
                else:
                    print(f"[{name}] error {str(e)}")
                raise e
            finally:
                dur = int((time.time() - ts) * 1000)
                print(f"[{name}] tool finished in {dur}ms")
        return wrapper
    
    # Add search engine tool
    @tools.tool(
        name="search_engine",
        description="Scrape search results from Google, Bing or Yandex. Returns SERP results in markdown (URL, title, description)"
    )
    async def search_engine(query: str, engine: str = "google"):
        """Search the web using specified search engine and query"""
        response = requests.post(
            'https://api.brightdata.com/request',
            json={
                'url': search_url(engine, query),
                'zone': WEB_UNLOCKER_ZONE,
                'format': 'raw',
                'data_format': 'markdown',
            },
            headers=get_api_headers()
        )
        return text_response(response.text)
    
    # Add scrape as markdown tool
    @tools.tool(
        name="scrape_as_markdown",
        description="Scrape a single webpage URL with advanced options for content extraction and get back the results in MarkDown language. This tool can unlock any webpage even if it uses bot detection or CAPTCHA."
    )
    async def scrape_as_markdown(url: str):
        """Scrape a webpage and return content as markdown"""
        response = requests.post(
            'https://api.brightdata.com/request',
            json={
                'url': url,
                'zone': WEB_UNLOCKER_ZONE,
                'format': 'raw',
                'data_format': 'markdown',
            },
            headers=get_api_headers()
        )
        return text_response(response.text)
    
    # Add scrape as HTML tool
    @tools.tool(
        name="scrape_as_html",
        description="Scrape a single webpage URL with advanced options for content extraction and get back the results in HTML. This tool can unlock any webpage even if it uses bot detection or CAPTCHA."
    )
    async def scrape_as_html(url: str):
        """Scrape a webpage and return content as HTML"""
        response = requests.post(
            'https://api.brightdata.com/request',
            json={
                'url': url,
                'zone': WEB_UNLOCKER_ZONE,
                'format': 'raw',
            },
            headers=get_api_headers()
        )
        return text_response(response.text)
    
    # Add session stats tool
    @tools.tool(
        name="session_stats",
        description="Tell the user about the tool usage during this session"
    )
    async def session_stats():
        """Get statistics about tool usage in the current session"""
        used_tools = list(debug_stats['tool_calls'].items())
        lines = ['Tool calls this session:']
        for name, calls in used_tools:
            lines.append(f"- {name} tool: called {calls} times")
        return text_response('\n'.join(lines))
    
    # Define datasets and add web_data tools for each
    datasets = [
        {
            'id': 'amazon_product',
            'dataset_id': 'gd_l7q7dkf244hwjntr0',
            'description': (
                'Quickly read structured amazon product data.\n'
                'Requires a valid product URL with /dp/ in it.\n'
                'This can be a cache lookup, so it can be more reliable than scraping'
            ),
            'inputs': ['url'],
        },
        {
            'id': 'amazon_product_reviews',
            'dataset_id': 'gd_le8e811kzy4ggddlq',
            'description': (
                'Quickly read structured amazon product review data.\n'
                'Requires a valid product URL with /dp/ in it.\n'
                'This can be a cache lookup, so it can be more reliable than scraping'
            ),
            'inputs': ['url'],
        },
        {
            'id': 'linkedin_person_profile',
            'dataset_id': 'gd_l1viktl72bvl7bjuj0',
            'description': (
                'Quickly read structured linkedin people profile data.\n'
                'This can be a cache lookup, so it can be more reliable than scraping'
            ),
            'inputs': ['url'],
        },
        {
            'id': 'linkedin_company_profile',
            'dataset_id': 'gd_l1vikfnt1wgvvqz95w',
            'description': (
                'Quickly read structured linkedin company profile data\n'
                'This can be a cache lookup, so it can be more reliable than scraping'
            ),
            'inputs': ['url'],
        },
        {
            'id': 'zoominfo_company_profile',
            'dataset_id': 'gd_m0ci4a4ivx3j5l6nx',
            'description': (
                'Quickly read structured ZoomInfo company profile data.\n'
                'Requires a valid ZoomInfo company URL.\n'
                'This can be a cache lookup, so it can be more reliable than scraping'
            ),
            'inputs': ['url'],
        },
        {
            'id': 'instagram_profiles',
            'dataset_id': 'gd_l1vikfch901nx3by4',
            'description': (
                'Quickly read structured Instagram profile data.\n'
                'Requires a valid Instagram URL.\n'
                'This can be a cache lookup, so it can be more reliable than scraping'
            ),
            'inputs': ['url'],
        }
    ]
    
    # Function to create dataset tool
    def create_dataset_tool(dataset_id, id, description, inputs):
        # Create parameter dictionary for the tool
        param_info = {}
        required_params = []
        for input_name in inputs:
            param_info[input_name] = (str, ...)  # This is FastAPI's way to mark required fields
            required_params.append(input_name)
        
        # Create tool function
        async def web_data_fn(**data):
            for param in required_params:
                if param not in data:
                    raise Exception(f"Missing required parameter: {param}")
            
            # Trigger BrightData dataset collection
            trigger_response = requests.post(
                'https://api.brightdata.com/datasets/v3/trigger',
                params={
                    'dataset_id': dataset_id,
                    'include_errors': True
                },
                json=[data],
                headers=get_api_headers()
            )
            
            response_data = trigger_response.json()
            if not response_data or 'snapshot_id' not in response_data:
                raise Exception('No snapshot ID returned from trigger request')
            
            snapshot_id = response_data['snapshot_id']
            print(f"[web_data_{id}] triggered collection with snapshot ID: {snapshot_id}")
            
            # Poll for the result
            max_attempts = 600
            attempts = 0
            
            while attempts < max_attempts:
                try:
                    snapshot_response = requests.get(
                        f'https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}',
                        params={'format': 'json'},
                        headers=get_api_headers()
                    )
                    
                    snapshot_data = snapshot_response.json()
                    if snapshot_data.get('status') == 'running':
                        print(f"[web_data_{id}] snapshot not ready, polling again (attempt {attempts + 1}/{max_attempts})")
                        attempts += 1
                        time.sleep(3)
                        continue
                    
                    print(f"[web_data_{id}] snapshot data received after {attempts + 1} attempts")
                    return text_response(json.dumps(snapshot_data))
                    
                except Exception as poll_error:
                    print(f"[web_data_{id}] polling error: {str(poll_error)}")
                    attempts += 1
                    time.sleep(3)
            
            raise Exception(f"Timeout after {max_attempts} seconds waiting for data")
        
        # Register the tool with FastMCP
        tools.tool(
            name=f"web_data_{id}",
            description=description
        )(web_data_fn)
    
    # Add all dataset tools
    for dataset in datasets:
        create_dataset_tool(
            dataset_id=dataset['dataset_id'],
            id=dataset['id'],
            description=dataset['description'],
            inputs=dataset['inputs']
        )
    
    # Start the server
    app = create_fastapi_app(tools)
    uvicorn.run(app, host="localhost", port=port)

# Create a combined OpenAI+MCP agent
class BrightDataMCPAgent(A2AServer):
    """OpenAI-powered agent with BrightData tools access"""
    
    def __init__(self, agent_card, openai_model, mcp_url=None):
        super().__init__(agent_card=agent_card)
        self.mcp_url = mcp_url
        
        # Setup OpenAI client but we'll call it directly
        self.openai_client = OpenAIA2AServer(
            api_key=os.environ["OPENAI_API_KEY"],
            model=openai_model,
            temperature=0,
            system_prompt=(
                "You are a helpful AI assistant that specializes in providing accurate data using BrightData web tools.\n\n"
                "IMPORTANT: When you receive a query, first identify which tool is most appropriate to answer it effectively:\n\n"
                "1. For search queries (search for X, find information about Y), use the search_engine tool.\n"
                "2. For specific webpage content extraction, use scrape_as_markdown.\n"
                "3. For Amazon product data, use web_data_amazon_product with the Amazon URL.\n"
                "4. For LinkedIn profiles, use web_data_linkedin_person_profile or web_data_linkedin_company_profile.\n"
                "5. For Instagram profiles, use web_data_instagram_profiles.\n\n"
                "After using a tool, provide a CONCISE, DIRECT response that:\n"
                "- Summarizes the key information in 2-3 sentences\n"
                "- Extracts only the most relevant data points\n"
                "- Organizes information in a structured way when appropriate\n"
                "- Avoids unnecessary details or explanations\n\n"
                "DO NOT explain how you got the information or which tools you used unless specifically asked.\n"
                "Focus on delivering accurate, to-the-point answers that directly address the query.\n"

                "If you recieve a prompt that require multiple searches or scrapings - Identify the needed action, then use the neccessery tools to find the data - When generating the response to these prompt - organize the data in the form of a list"
            )
        )
    
    def handle_task(self, task):
        """Handle incoming tasks, routing to OpenAI and tools as needed"""
        try:
            # Extract message text
            message_data = task.message or {}
            content = message_data.get("content", {})
            text = content.get("text", "") if isinstance(content, dict) else ""
            
            # Default text to send to OpenAI
            prompt_text = text
            tool_result = None
            
            # Check if we should use a tool based on simple keyword matching
            text_lower = text.lower()
            
            # Handle search engine requests
            if any(kw in text_lower for kw in ["search", "find information", "look up", "today", "find", "what", "where", "when", "how"]):
                search_query = text
                if "search for" in text_lower:
                    parts = text.split("search for", 1)
                    if len(parts) > 1:
                        search_query = parts[1].strip()
                elif "find information about" in text_lower:
                    parts = text.split("find information about", 1)
                    if len(parts) > 1:
                        search_query = parts[1].strip()
                elif "look up" in text_lower:
                    parts = text.split("look up", 1)
                    if len(parts) > 1:
                        search_query = parts[1].strip()
                
                try:
                    if self.mcp_url:
                        # Call the search_engine tool
                        tool_result = self.call_tool("search_engine", {"query": search_query})
                        prompt_text = f"{text}\n\nI'll search for information using a search engine.\nSearch results: {tool_result}"
                except Exception as e:
                    print(f"Error calling search_engine tool: {e}")
            
            # Handle webpage scraping requests
            elif any(kw in text_lower for kw in ["scrape", "extract from", "get content from", "price"]) and "http" in text_lower:
                # Extract the URL
                import re
                url_match = re.search(r'https?://\S+', text)
                if url_match:
                    url = url_match.group(0)
                    try:
                        if self.mcp_url:
                            # Call the scrape_as_markdown tool
                            tool_result = self.call_tool("scrape_as_markdown", {"url": url})
                            prompt_text = f"{text}\n\nI'll scrape the content from {url}.\nScraping result: {tool_result}"
                    except Exception as e:
                        print(f"Error calling scrape_as_markdown tool: {e}")
            
            # Get response from OpenAI
            from python_a2a import Message, TextContent, MessageRole
            
            # Create message for OpenAI
            message = Message(
                content=TextContent(text=prompt_text),
                role=MessageRole.USER
            )
            
            # Get response from OpenAI
            response = self.openai_client.handle_message(message)
            
            # Create response artifact
            response_text = response.content.text
            
            # Update with tool result if we didn't include it in prompt
            if tool_result and "Tool result" not in response_text:
                response_text = f"{response_text}\n\n(Used tool: {tool_result[:500]}...)"
            
            # Create response artifact
            task.artifacts = [{
                "parts": [{"type": "text", "text": response_text}]
            }]
            task.status = TaskStatus(state=TaskState.COMPLETED)
            
            return task
            
        except Exception as e:
            print(f"Error handling task: {e}")
            error_message = f"Sorry, I encountered an error: {str(e)}"
            task.artifacts = [{
                "parts": [{"type": "text", "text": error_message}]
            }]
            task.status = TaskStatus(state=TaskState.FAILED)
            return task
    
    def call_tool(self, tool_name, parameters):
        """Call an MCP tool directly using HTTP requests"""
        if not self.mcp_url:
            raise ValueError("No MCP URL configured")
        
        # Build the URL for the tool
        tool_url = f"{self.mcp_url}/tools/{tool_name}"
        
        # Make the request
        response = requests.post(
            tool_url, 
            json=parameters,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        
        # Parse and extract text content
        result = response.json()
        if "content" in result and len(result["content"]) > 0:
            if "text" in result["content"][0]:
                return result["content"][0]["text"]
        
        return str(result)

def test_agent(port):
    """Run a series of test queries against the agent"""
    time.sleep(3)  # Wait for server to fully start
    
    print("\n🧪 Testing the agent with sample queries...")
    client = A2AClient(f"http://localhost:{port}")
    
    test_queries = [
        "Search for recent news about artificial intelligence",
        "Can you scrape the content from https://news.ycombinator.com/ and summarize it?",
        "Tell me about space exploration in one paragraph"
    ]
    
    for query in test_queries:
        try:
            print(f"\n💬 Query: {query}")
            response = client.ask(query)
            print(f"🤖 Response: {response}")
            time.sleep(1)  # Pause between queries
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n✅ Test completed!")
    print(f"🔗 Agent running at http://localhost:{port}")
    print("🛑 Press Ctrl+C in the server terminal to stop")

def main():
    # Start MCP server if not explicitly disabled
    mcp_server_process = None
    mcp_url = f"http://localhost:{args.mcp_port}"
    
    if not args.no_auto_mcp:
        print(f"🔧 Starting MCP server on port {args.mcp_port}...")
        mcp_server_process = multiprocessing.Process(
            target=start_mcp_server,
            args=(args.mcp_port,)
        )
        mcp_server_process.start()
        print(f"✅ MCP server process started on port {args.mcp_port}")
        time.sleep(3)  # Give the server time to start
    
    try:
        # Create agent card
        agent_card = AgentCard(
            name="BrightData Web Agent",
            description=f"GPT-powered agent with BrightData web tools",
            url=f"http://localhost:{args.port}",
            version="1.0.0",
            skills=[
                AgentSkill(
                    name="GPT-Powered Responses",
                    description="Answer questions using OpenAI's GPT",
                    examples=["Tell me about quantum physics"]
                ),
                AgentSkill(
                    name="Web Search",
                    description="Search the web using search_engine function",
                    examples=["Search for recent AI advancements", "Find information about climate change"]
                ),
                AgentSkill(
                    name="Web Scraping",
                    description="Extract content from websites using your scrape_as_markdown or scrape_as_html tools",
                    examples=["Scrape https://news.ycombinator.com/", "Get content from https://www.example.com", "here is some data from https://www.example.com"]
                ),
                AgentSkill(
                    name="Structured Data Extraction",
                    description="Extract structured data from specific platformsor from your data sets",
                    examples=["Get data from Amazon product", "Extract LinkedIn profile information", "get data amazon_product"]
                ),
            ]
        )
        
        # Create the agent
        print(f"🤖 Starting BrightData+OpenAI agent with model {args.model}...")
        agent = BrightDataMCPAgent(
            agent_card=agent_card,
            openai_model=args.model,
            mcp_url=mcp_url if not args.no_auto_mcp else None
        )
        
        # Start test client process if testing is not disabled
        client_process = None
        if not args.no_test:
            client_process = multiprocessing.Process(
                target=test_agent,
                args=(args.port,)
            )
            client_process.start()
        
        # Run the server
        print(f"🚀 Server running at http://localhost:{args.port}")
        print("Example queries: 'Search for AI news', 'Scrape https://news.ycombinator.com/', 'Get data from a LinkedIn profile'")
        print("Press Ctrl+C to stop")
        
        run_server(agent, host="0.0.0.0", port=args.port)
        
    except KeyboardInterrupt:
        print("\n✅ Server stopped")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    finally:
        # Clean up processes
        if 'client_process' in locals() and client_process:
            client_process.terminate()
            client_process.join()
            
        if mcp_server_process:
            print("Stopping MCP server...")
            mcp_server_process.terminate()
            mcp_server_process.join()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())


            client_process.terminate()
            client_process.join()
            
        if mcp_server_process:
            print("Stopping MCP server...")
            mcp_server_process.terminate()
            mcp_server_process.join()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
