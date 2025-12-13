"""
WebSocket client for Guardrails Arena

This module provides a client to connect to the arena server.
Teams use this to join the game and participate over the network.
"""

import asyncio
import json
import websockets
from typing import Callable, Optional
from dataclasses import dataclass


@dataclass
class ArenaClient:
    """Client for connecting to the Guardrails Arena server."""
    
    server_url: str = "ws://localhost:8765"
    player_name: str = "Player"
    team: str = "red"  # "red", "blue", or "spectator"
    
    def __init__(self, server_url: str = "ws://localhost:8765", player_name: str = "Player", team: str = "red"):
        self.server_url = server_url
        self.player_name = player_name
        self.team = team
        self.websocket = None
        self.player_id = None
        self.callbacks = {
            "player_joined": [],
            "player_left": [],
            "attack_result": [],
            "attack_blocked": [],
            "guardrails_updated": [],
            "game_started": [],
            "attack_phase_started": [],
            "game_finished": [],
            "game_state": [],
            "chat": [],
            "error": [],
        }
    
    async def connect(self):
        """Connect to the arena server."""
        try:
            self.websocket = await websockets.connect(self.server_url)
            
            # Send join message
            await self.send({
                "type": "join",
                "name": self.player_name,
                "team": self.team,
            })
            
            # Wait for welcome message
            welcome = await self.websocket.recv()
            welcome_data = json.loads(welcome)
            
            if welcome_data.get("type") == "welcome":
                print(f"✅ Connected to arena: {welcome_data.get('message')}")
                return True
            
            return False
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    async def send(self, message: dict):
        """Send a message to the server."""
        if self.websocket:
            await self.websocket.send(json.dumps(message))
    
    async def receive_loop(self):
        """Listen for messages from the server."""
        if not self.websocket:
            return
        
        try:
            async for message in self.websocket:
                data = json.loads(message)
                msg_type = data.get("type")
                
                # Call registered callbacks
                if msg_type in self.callbacks:
                    for callback in self.callbacks[msg_type]:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(data)
                            else:
                                callback(data)
                        except Exception as e:
                            print(f"Error in callback: {e}")
        except websockets.exceptions.ConnectionClosed:
            print("⚠️ Connection to server closed")
    
    def on(self, event_type: str, callback: Callable):
        """Register a callback for an event type."""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
    
    async def attack(self, prompt: str):
        """Submit an attack (Red team)."""
        await self.send({
            "type": "attack",
            "prompt": prompt,
        })
    
    async def update_guardrails(self, **kwargs):
        """Update guardrails (Blue team)."""
        await self.send({
            "type": "update_guardrails",
            **kwargs,
        })
    
    async def game_control(self, action: str):
        """Send game control command."""
        await self.send({
            "type": "game_control",
            "action": action,
        })
    
    async def send_chat(self, message: str):
        """Send a team chat message."""
        await self.send({
            "type": "chat",
            "message": message,
        })
    
    async def get_state(self):
        """Request current game state."""
        await self.send({
            "type": "get_state",
        })
    
    async def disconnect(self):
        """Disconnect from the server."""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None


# Convenience function for notebook use
def create_client(server_url: str = "ws://localhost:8765", player_name: str = "Player", team: str = "red") -> ArenaClient:
    """Create an arena client instance."""
    return ArenaClient(server_url=server_url, player_name=player_name, team=team)


# Async helper for Jupyter
async def run_client_interactive(client: ArenaClient):
    """Run client with interactive loop (for notebook)."""
    connected = await client.connect()
    if not connected:
        return
    
    # Start receive loop
    await client.receive_loop()


# For notebook cell execution
def start_client_background(client: ArenaClient):
    """Start client in background (for Jupyter notebook)."""
    import threading
    
    def run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_client_interactive(client))
    
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread
