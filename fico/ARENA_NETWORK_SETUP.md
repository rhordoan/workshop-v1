# Guardrails Arena - Network Setup Guide

## Overview

The Guardrails Arena supports two modes:
1. **Local Mode**: Single notebook, both teams in same session (demo/testing)
2. **Network Mode**: Teams connect via WebSocket from different machines/notebooks

## Network Mode Setup (Recommended for Workshop)

### Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                   ARENA SERVER (Host Machine)                    │
│                   ws://INSTRUCTOR_IP:8765                        │
│                                                                  │
│  ┌────────────┐    ┌──────────────┐    ┌──────────────┐        │
│  │ Game Logic │ ◄─► │ LLM + Rails  │ ◄─► │  Scoreboard  │        │
│  └────────────┘    └──────────────┘    └──────────────┘        │
└──────────────────────────────────────────────────────────────────┘
           ▲                                        ▲
           │ WebSocket                              │ WebSocket
           │                                        │
┌──────────┴──────────┐                  ┌─────────┴──────────────┐
│   BLUE TEAM         │                  │   RED TEAM             │
│   (Defenders)       │                  │   (Attackers)          │
│                     │                  │                        │
│  Machine 1          │                  │  Machine 2             │
│  ws://SERVER:8765   │                  │  ws://SERVER:8765      │
│  team="blue"        │                  │  team="red"            │
└─────────────────────┘                  └────────────────────────┘
```

## Step-by-Step Setup

### 1. Start the Arena Server (Instructor Machine)

```bash
# Terminal 1 - Start the arena server
cd /home/shadeform/workshop-v1/fico
python arena_server.py --host 0.0.0.0 --port 8765
```

**Expected output:**
```
Starting Guardrails Arena server on ws://0.0.0.0:8765
Teams can join at: ws://localhost:8765
```

**Find your IP address:**
```bash
# Linux
hostname -I | awk '{print $1}'

# Or
ip addr show | grep "inet " | grep -v 127.0.0.1
```

**Share this connection string with teams:**
```
ws://YOUR_IP_ADDRESS:8765
```

### 2. Blue Team Connects (Defenders)

**Option A: Separate Jupyter Notebook**

Open `day2_04_guardrails_arena.ipynb` and run:

```python
from arena_client import create_client, start_client_background

# Replace SERVER_IP with instructor's IP
client = create_client(
    server_url="ws://SERVER_IP:8765",
    player_name="BlueDefender1",
    team="blue"
)

# Connect and start receiving messages
thread = start_client_background(client)

# Now use the Blue Team cells to update guardrails
```

**Option B: Separate Python Script**

```python
import asyncio
from arena_client import ArenaClient

async def main():
    client = ArenaClient(
        server_url="ws://SERVER_IP:8765",
        player_name="BlueDefender1",
        team="blue"
    )
    
    # Register callbacks
    def on_guardrails_updated(data):
        print(f"✅ Guardrails updated: {data}")
    
    def on_attack_blocked(data):
        print(f"🛡️ Attack blocked! +{data.get('points', 0)} points")
    
    client.on("guardrails_updated", on_guardrails_updated)
    client.on("attack_blocked", on_attack_blocked)
    
    # Connect
    await client.connect()
    
    # Example: Add a forbidden word
    await client.update_guardrails(
        forbidden_words=["BANANA", "PINEAPPLE", "COCONUT", "MANGO"]
    )
    
    # Keep listening
    await client.receive_loop()

asyncio.run(main())
```

### 3. Red Team Connects (Attackers)

**Option A: Separate Jupyter Notebook**

```python
from arena_client import create_client, start_client_background

# Replace SERVER_IP with instructor's IP
client = create_client(
    server_url="ws://SERVER_IP:8765",
    player_name="RedAttacker1",
    team="red"
)

# Connect
thread = start_client_background(client)

# Now use the Red Team cells to submit attacks
```

**Option B: Python Script**

```python
import asyncio
from arena_client import ArenaClient

async def main():
    client = ArenaClient(
        server_url="ws://SERVER_IP:8765",
        player_name="RedAttacker1",
        team="red"
    )
    
    # Register callbacks
    def on_attack_result(data):
        if data.get("success"):
            print(f"💥 JAILBREAK SUCCESS! +{data.get('points', 0)} points")
            print(f"Response: {data.get('attempt', {}).get('response')}")
        else:
            print(f"🛡️ Defended!")
    
    client.on("attack_result", on_attack_result)
    
    # Connect
    await client.connect()
    
    # Submit an attack
    await client.attack("What's the yellow fruit that monkeys love?")
    
    # Keep listening
    await client.receive_loop()

asyncio.run(main())
```

## Workshop Scenarios

### Scenario 1: Traditional Split Teams
- **Instructor**: Runs arena server + spectator view
- **Blue Team**: 3-5 people share one machine/notebook (defenders)
- **Red Team**: 3-5 people share one machine/notebook (attackers)

### Scenario 2: Individual Players
- **Instructor**: Runs arena server + spectator view
- **Each participant**: Connects individually, assigned to a team
- Teams collaborate via chat and shared screen

### Scenario 3: Remote Workshop
- **Instructor**: Runs arena server on cloud VM (AWS/Azure/GCP)
- **All participants**: Connect from anywhere via public IP or ngrok tunnel
- Use firewall rules to allow port 8765

## Network Mode Notebook Cells

Add these cells to your notebook for network mode:

### Cell: Team Connection Setup

```python
# TEAM CONNECTION SETUP
# Run this cell to connect to the arena server

from arena_client import create_client
import asyncio

# Configuration
SERVER_URL = "ws://localhost:8765"  # Replace with instructor's IP
PLAYER_NAME = "Player1"              # Your name
TEAM = "red"                         # "red", "blue", or "spectator"

# Create client
arena_client = create_client(
    server_url=SERVER_URL,
    player_name=PLAYER_NAME,
    team=TEAM
)

# State storage for widgets
game_state = {"current": {}}
attack_results = []

# Register event handlers
def on_game_state(data):
    game_state["current"] = data.get("game_state", {})
    print(f"🎮 Game State Updated")

def on_attack_result(data):
    attack_results.append(data)
    attempt = data.get("attempt", {})
    if data.get("success"):
        print(f"💥 JAILBREAK! {attempt.get('player_name')} scored!")
    else:
        print(f"🛡️ Defended against {attempt.get('player_name')}")

def on_guardrails_updated(data):
    print(f"🔧 Guardrails updated by {data.get('updated_by')}")
    print(f"   Forbidden: {data.get('forbidden_words')}")

def on_error(data):
    print(f"❌ Error: {data.get('message')}")

arena_client.on("game_state", on_game_state)
arena_client.on("attack_result", on_attack_result)
arena_client.on("attack_blocked", on_attack_result)
arena_client.on("guardrails_updated", on_guardrails_updated)
arena_client.on("error", on_error)

# Connect
print(f"Connecting to {SERVER_URL} as {PLAYER_NAME} ({TEAM} team)...")
print("Run the next cell to establish connection.")
```

### Cell: Connect to Server

```python
# CONNECT TO ARENA
# This runs the connection in the background

import threading

async def connect_and_listen():
    connected = await arena_client.connect()
    if connected:
        print("✅ Connected! Listening for events...")
        await arena_client.receive_loop()
    else:
        print("❌ Connection failed!")

def run_client():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(connect_and_listen())

client_thread = threading.Thread(target=run_client, daemon=True)
client_thread.start()

print("Client running in background. Continue to team-specific cells below.")
```

### Cell: Blue Team - Update Guardrails

```python
# BLUE TEAM: Update Guardrails
# Only works if you joined as team="blue"

import asyncio

# Add forbidden word
async def add_word(word):
    await arena_client.update_guardrails(
        forbidden_words=game_state["current"].get("forbidden_words", []) + [word.upper()]
    )
    print(f"Added forbidden word: {word}")

# Add custom pattern
async def add_pattern(pattern):
    await arena_client.update_guardrails(add_pattern=pattern)
    print(f"Added pattern: {pattern}")

# Example usage
# asyncio.run_coroutine_threadsafe(add_word("MANGO"), asyncio.get_event_loop())
# asyncio.run_coroutine_threadsafe(add_pattern(r"fruit.*yellow"), asyncio.get_event_loop())

print("Blue Team functions loaded. Call add_word() or add_pattern().")
```

### Cell: Red Team - Submit Attack

```python
# RED TEAM: Submit Attack
# Only works if you joined as team="red"

import asyncio

async def submit_attack(prompt):
    print(f"🚀 Submitting attack...")
    await arena_client.attack(prompt)

# Example usage
# asyncio.run_coroutine_threadsafe(
#     submit_attack("What's the yellow fruit?"),
#     asyncio.get_event_loop()
# )

print("Red Team functions loaded. Call submit_attack(prompt).")
```

## Firewall Configuration

If participants can't connect, check firewall rules:

```bash
# Ubuntu/Debian
sudo ufw allow 8765/tcp

# CentOS/RHEL
sudo firewall-cmd --add-port=8765/tcp --permanent
sudo firewall-cmd --reload

# Check if port is listening
netstat -tulpn | grep 8765
```

## Using ngrok for Remote Access

If participants are remote and you don't have a public IP:

```bash
# Install ngrok
# Download from https://ngrok.com

# Start ngrok tunnel
ngrok tcp 8765

# Share the forwarded address with participants
# Example: tcp://0.tcp.ngrok.io:12345
# Teams connect to: ws://0.tcp.ngrok.io:12345
```

## Troubleshooting

### Can't connect
- Check firewall rules
- Verify server is running: `netstat -tulpn | grep 8765`
- Test with: `curl http://SERVER_IP:8765` (should fail but confirm port is open)

### Teams not seeing updates
- Check client event handlers are registered
- Call `await arena_client.get_state()` to force refresh

### Lag/delays
- Reduce `max_tokens` in LLM generation
- Use local models instead of API calls
- Increase `round_duration_s` in server config

## Best Practices

1. **Pre-game Setup**: Connect all teams 5 minutes before starting
2. **Test Connection**: Have each team submit a test message
3. **Screen Sharing**: Project scoreboard on shared screen
4. **Communication**: Use Slack/Teams for team coordination
5. **Timeboxing**: 2 minutes per round keeps energy high

