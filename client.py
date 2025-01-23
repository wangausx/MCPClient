import asyncio
from typing import Optional
from contextlib import AsyncExitStack
import tkinter as tk
from tkinter import ttk, scrolledtext

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from concurrent.futures import ThreadPoolExecutor

from anthropic import Anthropic
from dotenv import load_dotenv
import os
from os import getenv

import threading
import sys
import time


load_dotenv()  # load environment variables from .env

class SessionManager:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.stdio = None
        self.write = None
        self.exit_stack = None
        self._lock = threading.Lock()
        self._loop = None
        self.stop_event = threading.Event()         

    async def initialize_session(self, server_params):
        self.exit_stack = AsyncExitStack()
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        self._loop = asyncio.get_running_loop()
        await self.session.initialize()

    async def run_in_session(self, coroutine):
        with self._lock:
            if self._loop != asyncio.get_running_loop():
                # If we're in a different loop, use run_coroutine_threadsafe
                future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
                try:
                    # Wait for the result synchronously, with a timeout to prevent deadlocks
                    return future.result(timeout=5)  # Adjust timeout as needed
                except Exception as e:
                    future.cancel()
                    raise e
            else:
                # If we're already in the correct loop, just await the coroutine
                return await coroutine
                
    def get_session(self):
        with self._lock:
            return self.session, self._loop
    async def close(self):
        if self.exit_stack:
            await self.exit_stack.aclose()

class MCPClient:
    def __init__(self, worker_loop):
        self.loop = worker_loop
        self.root = tk.Tk()
        self.root.title("My API App")
        self.root.geometry("800x600")
        self.setup_gui()

        # Add debugging prints
        print("Current working directory:", os.getcwd())
        print("Environment variables:", {k: v for k, v in os.environ.items() if 'ANTHROPIC' in k})
        
        load_dotenv(verbose=True)  # Add verbose=True to see if .env is being loaded
        
        # Get API key and print status (mask the actual key for security)
        api_key = getenv("ANTHROPIC_API_KEY")
        print("API key loaded:", "Yes" if api_key else "No")
        
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        self.anthropic = Anthropic(api_key=api_key)
    
    def setup_gui(self):
        # Main container
        self.main_frame = ttk.Frame(self.root, padding="5")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Chat display area
        self.chat_display = scrolledtext.ScrolledText(self.main_frame, wrap=tk.WORD, height=20)
        self.chat_display.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.chat_display.config(state=tk.DISABLED)

        # Input area
        self.input_field = ttk.Entry(self.main_frame)
        self.input_field.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.input_field.bind("<Return>", self.send_message_button_handler)

        # Send button
        send_button = ttk.Button(self.main_frame, text="Send", command=self.send_message_button_handler)
        send_button.grid(row=3, column=1)

        # Status bar
        self.status_var = tk.StringVar(value="Disconnected")
        status_label = ttk.Label(self.main_frame, textvariable=self.status_var)
        status_label.grid(row=2, column=0, columnspan=2)

        # Configure grid weights
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(0, weight=1)

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
            
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        future = asyncio.run_coroutine_threadsafe(
            session_manager.initialize_session(server_params),
            self.loop,  # self.loop refers to the worker_loop
        )
        # Wait for the session initialization to complete
        await asyncio.wrap_future(future)        
        self.status_var.set("Connected")

        
    async def process_query(self, query: str) -> str:
        session, loop = session_manager.get_session()
        """Process a query using Claude and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]
        print(f"Session state: {session}")
        try:
            response = await session_manager.run_in_session(session.list_tools())
        except Exception as e:
            print(f"Error processing query: {e}")
            raise e

        available_tools = [{ 
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        # Initial Claude API call
        print("\nInitial Claude API call...")
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
                result = await session_manager.run_in_session(session.call_tool(tool_name, tool_args))
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
                print("\nFurther response from Claude API received!")
                final_text.append(response.content[0].text)

        return "\n".join(final_text)
   
    def send_message_button_handler(self, event=None):
        print("Button pressed and delegate the handling to worker_loop")
        self.loop.call_soon_threadsafe(asyncio.create_task, self.send_message())

    async def send_message(self):
        print("\nSend button pressed!")
        message = self.input_field.get()
        if not message:
            print("No message to send")
            return
         # Simulate async work (e.g., sending a message over the network)
        await asyncio.sleep(0.1)
        print(f"Sending message: {message}")
 
       # Use after_idle to modify tkinter widgets from the main thread
        self.root.after_idle(lambda: self.input_field.delete(0, tk.END))
        print("\ninput_field deleted!")

        """Process a query using Claude and available tools"""    
        try:
            response = await self.process_query(message)
            print("\nprocess_query completed!")
            # Schedule GUI updates in the main thread
            self.root.after_idle(lambda: self.display_message(response))
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}\n"
            self.root.after_idle(lambda: self.display_message(error_msg))

    def display_message(self, message):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, message)
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)

    def on_closing(self, worker_thread):

        # Destroy the GUI window
        self.root.destroy()

        # Safely close the session and stop the worker loop
        future = asyncio.run_coroutine_threadsafe(session_manager.close(), self.loop)
        try:
            future.result()  # Wait for session cleanup to complete
        except Exception as e:
            print(f"Error during session cleanup: {e}")

        # Stop the worker loop
        self.loop.call_soon_threadsafe(self.loop.stop)

        # Wait for the worker thread to finish
        worker_thread.join()

def run_worker_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

async def graceful_shutdown(loop):
    """Gracefully stop the worker loop."""
    tasks = asyncio.all_tasks(loop=loop)
    [task.cancel() for task in tasks]
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        pass
    finally:
        loop.stop()

session_manager = SessionManager()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    # Start a separate thread for the asyncio event loop
    worker_loop = asyncio.new_event_loop()
    worker_thread = threading.Thread(target=run_worker_loop, args=(worker_loop, ))
    worker_thread.start()

    client = MCPClient(worker_loop)

    await client.connect_to_server(sys.argv[1])

    client.root.protocol("WM_DELETE_WINDOW", lambda: client.on_closing(worker_thread))
    try:
        client.root.mainloop()
    except KeyboardInterrupt:
        client.on_closing(worker_thread)


if __name__ == "__main__":
    asyncio.run(main())

