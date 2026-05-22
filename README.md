# 🏠 Smart Home Voice & Text Assistant

**No apps. No buttons. Just talk or type.**

A simple, natural language interface for controlling your smart home. Designed for everyone—especially those who aren't comfortable with complicated phone apps or remotes.

---

## 💡 What This Does

Instead of learning a phone app or remembering where all the smart devices are hidden, you simply **type what you want** in plain English. The system understands you, finds the right device, and does it.

### Examples of what you can say:

✅ "Turn on the lights in the bedroom"  
✅ "I'm going to sleep — set AC to 20 degrees and lock the front door"  
✅ "Switch off the TV and fan in the living room"  
✅ "Turn on the kitchen light"  
✅ "Lock the back door"  

---

## 🎯 How It Works (In Simple Terms)

```
YOU TYPE 
   ↓
🤖 AI understands what you want
   ↓
🔍 AI finds the right smart device
   ↓
⚡ Device turns on/off
   ↓
✅ Confirmation message
```

No complicated steps. No confusing settings. Just type naturally like you're texting a friend.

---

## 📋 What Devices Can You Control?

### 💡 **Lights**
- Bedroom light (on/off)
- Hall light (on/off)
- Kitchen light (on/off)

### 📺 **TV**
- Hall TV (on/off)
- Bedroom TV (on/off)

### 🌀 **Fan**
- Bedroom fan (on/off)
- Hall fan (on/off)

### ❄️ **AC & Heat**
- Bedroom AC (set temperature)
- Whole house thermostat (set temperature)

### 🔒 **Doors**
- Front door (lock/unlock)
- Back door (lock/unlock)

---

## 🚀 Getting Started

### Step 1: Install (One Time Only)

Make sure you have Python installed on your computer. Ask your grandchild to help if you're unsure.

Open the command prompt (or terminal) and run:

```bash
pip install chromadb langchain-ollama langgraph
```

### Step 2: Run the Program

Go to the Smart Home folder and run:

```bash
python smart_home.py
```

That's it! The system will start and process your commands.

---

## 💬 How to Use It

### Format: Just Type Naturally

You don't need to use special commands. Just type what you want done:

| What You Want | What You Type |
|---|---|
| Turn on bedroom light | "Turn on the lights in the bedroom" |
| Go to sleep mode | "I'm going to sleep — set AC to 20 degrees and lock the front door" |
| Turn off TV | "Switch off the TV" |
| Lock door before leaving | "Lock the front door" |
| Change temperature | "Set the thermostat to 22 degrees" |

---

## 📊 Output Explanation

When you run a command, you'll see something like this:

```
═════════════════════════════════════════════════════
[Orch] Query: Turn on the lights in the bedroom
[Orch] Refined: Turn on bedroom light

[EmbedAI] Searching → 'Turn on bedroom light'
[EmbedAI] Found nodes: ['light_on_bedroom']

[Orch→Embed] APPROVE: Found 1 candidate device(s)

[ToolAI] Planning for nodes: light_on_bedroom
[ToolAI] Plan: [{"function": "light_on_bedroom"}]

[Orch→Tool] APPROVE: Plan has 1 valid action(s)

[Executor] Running plan...
  [IoT] 💡 Light ON  → Bedroom

[Orch→Results] APPROVE: Actions executed successfully

[Response] ✅ Your bedroom light is now on!
```

**Don't worry about all the text in brackets.** Just look for the ✅ checkmark at the end—that means it worked!

---

## ✅ What Each Part Means

| Line | Meaning |
|---|---|
| `[Orch]` | AI is thinking about what you want |
| `[EmbedAI]` | AI is searching for the right device |
| `Found nodes:` | AI found matching device(s) |
| `APPROVE` | Everything looks good, moving forward |
| `[Executor]` | Device is now being controlled |
| `[Response] ✅` | Task complete! |

---

## ❌ Troubleshooting

### Problem: "I typed something but nothing happened"

**Solution:** Try using simpler, more direct language. Instead of "Can you please make it a bit cooler in my room?", try "Set bedroom AC to 20 degrees"

---

### Problem: "It says unknown device or function"

**Solution:** Make sure you're asking for a device that actually exists. Check the list above. For example, there's no "backyard light" or "garage door" yet—only bedroom, hall, and kitchen lights.

---

### Problem: "The system is stuck or keeps retrying"

**Solution:** This is normal for complex commands. Give it 10-15 seconds to think. If it still doesn't work:
- Stop the program (press `Ctrl + C`)
- Try a simpler command next time
- Check that all your devices are powered on

---

## 🔧 Can I Add More Devices?

Yes! You can add more rooms or devices by editing the Python file (ask a tech-savvy friend or family member to help).

Current devices are easy to expand:
- Add lights in different rooms
- Add TVs in other rooms
- Add fans in more areas
- Add new doors

---

## 🎓 For the Tech-Savvy (Optional Reading)

This system uses an **Agentic Supervisor Architecture**:

1. **Orchestrator** - Understands your request
2. **Embedding AI** - Finds matching devices using semantic search
3. **Tool Caller** - Creates an action plan
4. **Executor** - Runs the actual device commands
5. **Responder** - Gives you a friendly confirmation

It has built-in safeguards to prevent mistakes and retry if something goes wrong.

---

## 📝 Tips for Best Results

1. **Be clear and direct:**
   - ✅ "Turn on bedroom light"
   - ❌ "Can you make things brighter in my bedroom?"

2. **Mention the room:**
   - ✅ "Hall TV on"
   - ❌ "TV on" (ambiguous)

3. **Include specific values for temperature:**
   - ✅ "Set thermostat to 22 degrees"
   - ❌ "Make it cooler" (too vague)

4. **One action at a time (usually):**
   - ✅ "Turn on bedroom light" then "Lock front door"
   - ✅ "Set AC to 20 and lock front door" (complex but works)

---

## 📞 Need Help?

If something isn't working:

1. Check that all devices are powered on
2. Make sure you have a stable internet/network connection
3. Try rephrasing your request in simpler words
4. Ask a tech-savvy friend or family member

---

## 🎉 Enjoy!

This system is designed to make your smart home **easier**, not more complicated. Just type naturally, and let the AI do the rest.

**Happy smart home living!** 🏠✨
