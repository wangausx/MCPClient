import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
from queue import Queue
import json
import websocket
import rel
import sys
from anthropic import Anthropic

class MCPClientGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MCP Client")
        self.root.geometry("800x600")
        
        # Message queue for thread-safe communication
        self.message_queue = Queue()
        
        # WebSocket connection
        self.ws = None
        self.connected = False
        
        # Anthropic client
        self.anthropic = Anthropic()
        
        self.setup_gui()
        self.setup_websocket()
        
        # Start message processing thread
        self.message_thread = threading.Thread(target=self.process_messages, daemon=True)
        self.message_thread.start()

    def setup_gui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Chat display area
        self.chat_display = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, height=20)
        self.chat_display.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.chat_display.config(state=tk.DISABLED)

        # Input area
        self.input_field = ttk.Entry(main_frame)
        self.input_field.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.input_field.bind("<Return>", self.send_message)

        # Send button
        send_button = ttk.Button(main_frame, text="Send", command=self.send_message)
        send_button.grid(row=1, column=1)

        # Status bar
        self.status_var = tk.StringVar(value="Disconnected")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.grid(row=2, column=0, columnspan=2)

        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)

    def setup_websocket(self):
        def on_message(ws, message):
            self.message_queue.put(("received", message))

        def on_error(ws, error):
            self.message_queue.put(("error", str(error)))

        def on_close(ws, close_status_code, close_msg):
            self.connected = False
            self.message_queue.put(("status", "Disconnected"))

        def on_open(ws):
            self.connected = True
            self.message_queue.put(("status", "Connected"))

        # WebSocket setup
        self.ws = websocket.WebSocketApp("ws://your-mcp-server:port",
                                       on_message=on_message,
                                       on_error=on_error,
                                       on_close=on_close,
                                       on_open=on_open)
        
        # Start WebSocket connection in a separate thread
        ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
        ws_thread.start()

    def send_message(self, event=None):
        if not self.connected:
            self.display_message("System: Not connected to server\n")
            return

        message = self.input_field.get()
        if not message:
            return

        # Clear input field
        self.input_field.delete(0, tk.END)

        # Send message to server
        try:
            # Format message according to your MCP protocol
            mcp_message = {
                "type": "message",
                "content": message
            }
            self.ws.send(json.dumps(mcp_message))
            
            # Display sent message
            self.display_message(f"You: {message}\n")
            
            # Process with Anthropic API
            self.process_with_anthropic(message)
            
        except Exception as e:
            self.display_message(f"Error sending message: {str(e)}\n")

    def process_with_anthropic(self, message):
        try:
            # Create message for Anthropic API
            response = self.anthropic.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": message
                }]
            )
            
            # Process and display response
            if response.content:
                self.display_message(f"Claude: {response.content[0].text}\n")
                
                # Send response back to MCP server if needed
                mcp_response = {
                    "type": "assistant_response",
                    "content": response.content[0].text
                }
                self.ws.send(json.dumps(mcp_response))
                
        except Exception as e:
            self.display_message(f"Error processing with Anthropic API: {str(e)}\n")

    def display_message(self, message):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, message)
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)

    def process_messages(self):
        while True:
            msg_type, content = self.message_queue.get()
            
            if msg_type == "received":
                try:
                    data = json.loads(content)
                    # Process according to your MCP protocol
                    if data.get("type") == "message":
                        self.display_message(f"Server: {data['content']}\n")
                except json.JSONDecodeError:
                    self.display_message(f"Received: {content}\n")
                    
            elif msg_type == "error":
                self.display_message(f"Error: {content}\n")
                
            elif msg_type == "status":
                self.status_var.set(content)

    def on_closing(self):
        if self.ws:
            self.ws.close()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = MCPClientGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()