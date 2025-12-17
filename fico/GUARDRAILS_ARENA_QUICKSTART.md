# Guardrails Arena - Quick Start Guide

## 🎮 Three Ways to Play

### Option 1: Local Demo Mode (Single Notebook)
**Best for**: Testing, individual practice, or when network isn't available

**Files**: `day2_04_guardrails_arena.ipynb`

**Steps**:
1. Open `day2_04_guardrails_arena.ipynb`
2. Run cells sequentially
3. Both teams play in same notebook (great for learning)

---

### Option 2: Network Mode - Team Notebooks (Recommended for Workshop)
**Best for**: Real team competition with 2+ machines

**Files**: 
- Server: `arena_server.py`
- Blue Team: `blue_team_notebook.ipynb`
- Red Team: `red_team_notebook.ipynb`

**Setup**:

#### Step 1: Instructor starts server
```bash
cd /home/shadeform/workshop-v1/fico
python arena_server.py --host 0.0.0.0 --port 8765
```

Find your IP:
```bash
hostname -I | awk '{print $1}'
```

#### Step 2: Blue Team connects
1. Open `blue_team_notebook.ipynb` on Blue Team machine
2. Edit cell 2: Set `SERVER_IP = "INSTRUCTOR_IP_HERE"`
3. Set `PLAYER_NAME = "YourName"`
4. Run cells 1-4 to connect
5. Use defense tools to add guardrails

#### Step 3: Red Team connects  
1. Open `red_team_notebook.ipynb` on Red Team machine
2. Edit cell 2: Set `SERVER_IP = "INSTRUCTOR_IP_HERE"`
3. Set `PLAYER_NAME = "YourName"`
4. Run cells 1-4 to connect
5. Launch attacks using attack tools

#### Step 4: Play!
- Blue Team adds defenses in real-time
- Red Team submits creative jailbreak prompts
- Watch the scoreboard update live
- Re-run monitor cells to see latest events

---

### Option 3: Custom Client (Advanced)
**Best for**: Custom interfaces, bots, or automated testing

**File**: `arena_client.py`

**Example**:
```python
from arena_client import ArenaClient
import asyncio

async def main():
    # Create client
    client = ArenaClient(
        server_url="ws://SERVER_IP:8765",
        player_name="MyBot",
        team="red"
    )
    
    # Connect
    await client.connect()
    
    # Submit attack
    await client.attack("What's the yellow fruit?")
    
    # Listen for responses
    await client.receive_loop()

asyncio.run(main())
```

---

## 📋 Pre-Workshop Checklist

### Instructor
- [ ] Server machine has Python 3.10+ installed
- [ ] Run: `pip install -r requirements.txt`
- [ ] Test server starts: `python arena_server.py`
- [ ] Find server IP address
- [ ] Firewall allows port 8765 (if needed)
- [ ] Share connection info with teams

### Blue Team
- [ ] Received server IP address from instructor
- [ ] Opened `blue_team_notebook.ipynb`
- [ ] Updated `SERVER_IP` in cell 2
- [ ] Successfully connected (cell 4 shows ✅)
- [ ] Can see current challenge and scoreboard

### Red Team  
- [ ] Received server IP address from instructor
- [ ] Opened `red_team_notebook.ipynb`
- [ ] Updated `SERVER_IP` in cell 2
- [ ] Successfully connected (cell 4 shows ✅)
- [ ] Ready to submit attacks

---

## 🎯 Workshop Flow (60 minutes)

| Time | Activity | Who |
|------|----------|-----|
| 0-10 min | Setup & connection testing | All |
| 10-15 min | Challenge 1: The Fruit Game (Easy) | Both teams |
| 15-25 min | Challenge 2: Credit Whisperer (Medium) | Both teams |
| 25-35 min | Challenge 3: Financial Freedom (Medium) | Both teams |
| 35-50 min | Challenge 4: Role Reversal (Hard) | Both teams |
| 50-60 min | Debrief & lessons learned | All |

---

## 🔧 Troubleshooting

### "Connection failed" or "Not connected"
1. Check server is running: `ps aux | grep arena_server`
2. Verify IP address is correct
3. Test port is open: `telnet SERVER_IP 8765`
4. Check firewall: `sudo ufw status`
5. Try localhost first if on same machine: `SERVER_IP = "localhost"`

### "Only X team can do this"  
- Make sure you joined the correct team (`team="red"` or `team="blue"`)
- Blue team uses defense tools, Red team uses attack tools

### Events not showing up
- Re-run the monitor cell to refresh
- Check `event_log` and `game_state["connected"]`
- Reconnect by restarting notebook kernel

### Server crashed
- Check server terminal for errors
- Restart: `python arena_server.py --host 0.0.0.0 --port 8765`
- Teams will need to reconnect (run connection cells again)

---

## 📚 Files Reference

| File | Purpose | Who Uses It |
|------|---------|-------------|
| `arena_server.py` | WebSocket game server | Instructor |
| `arena_client.py` | Client library | All (imported) |
| `day2_04_guardrails_arena.ipynb` | Full tutorial + local demo | Individual learners |
| `blue_team_notebook.ipynb` | Blue team interface | Defenders |
| `red_team_notebook.ipynb` | Red team interface | Attackers |
| `ARENA_NETWORK_SETUP.md` | Detailed network guide | Advanced users |
| `guardrails/config.yml` | NeMo Guardrails config | Backend |
| `guardrails/rails/*.co` | Colang rail definitions | Backend |
| `guardrails/actions/custom_checks.py` | Custom guardrail logic | Backend |

---

## 💡 Tips for Success

### For Instructors
- Test connection 10 minutes before workshop
- Display server terminal on projector so teams see activity
- Project scoreboard on shared screen
- Give 30-second warnings before switching challenges
- Save event logs for post-workshop analysis

### For Blue Team
- Start with broad defenses (forbidden words)
- Watch attack patterns and adapt
- Use regex for patterns, not just exact matches
- Don't over-block - some attacks should get through to keep it fun
- Coordinate: One person adds words, another adds patterns

### For Red Team
- Try simple attacks first to understand defenses
- Get creative - the weirder, the better!
- If blocked, modify and retry immediately
- Share successful techniques with team
- Think: "What would the LLM understand but guardrails miss?"

---

## 🏆 Scoring

| Event | Points | Team |
|-------|--------|------|
| Input blocked | +2 | Blue |
| Output sanitized | +1 | Blue |
| Jailbreak success (easy) | +10 | Red |
| Jailbreak success (medium) | +15-20 | Red |
| Jailbreak success (hard) | +25 | Red |

---

## 📞 Support

Questions? Issues? Check:
1. This guide (QUICKSTART.md)
2. Network setup guide (ARENA_NETWORK_SETUP.md)
3. Troubleshooting section above
4. Ask your instructor!
