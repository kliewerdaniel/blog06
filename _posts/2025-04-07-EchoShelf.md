---
layout: post
title: EchoShelf
description: Voice-Based Inventory Memory with Daily Team Sync
date:   2025-04-07 01:42:44 -0500
---


# Building EchoShelf: A Voice-Based Memory System for Inventory Teams

In fast-paced retail environments, small oversights can ripple into larger issues—especially when it comes to inventory discrepancies. A misplaced product, a mislabeled bin, or an undocumented damage report can quickly snowball into inaccurate counts and wasted time. I’ve worked in that environment, and I’ve seen how often simple notes or observations never get passed along because there’s no easy system to capture and share them. That’s what inspired EchoShelf.

EchoShelf is a lightweight app designed to give front-line workers a quick and intuitive way to document what they see, using just their voice. Whether they’re replenishing shelves or auditing counts, they can record a brief note on the spot. The system then transcribes that note, tags it with the item and location, and stores it for later use. Over time, EchoShelf becomes a searchable memory—an evolving log of what happened, where, and why.

The core idea is simple: let people speak what they know while they’re working, and make that knowledge useful for the entire team.

Recently, I added a daily sync feature that pushes out a summary of new updates to everyone on the team. Whether it’s through a dashboard, email, or printed sheet, everyone can stay informed about what changed, why an item’s count is off, or where a product was moved. It’s a small shift with a big payoff—less confusion, fewer repeated mistakes, and faster onboarding for new employees.

In this guide, I’ll walk through how to build EchoShelf using open-source tools. We’ll combine voice transcription with Whisper, local AI using Ollama, and a simple system for logging and surfacing updates across the team. You don’t need a huge tech stack to make something genuinely useful—just a clear problem to solve and a willingness to build.

If you’re ready to follow along, the next section will show you how to set up the basic project structure and start capturing your first voice notes. From there, we’ll build the daily sync system and explore features like search, role-based views, and even barcode integration down the road.

Let’s get started.

**EchoShelf** is a lightweight, voice-first app that captures quick spoken notes from employees as they restock or audit shelves. It transcribes and links those notes to specific items and locations. Over time, it becomes a searchable, AI-assisted memory of why counts were off, where items were moved, and what issues occurred.

**New enhancement:** Each day, the system generates a summary of newly added discrepancy notes and pushes them to a shared feed (via a simple web dashboard, SMS/email, or PDF printout for managers), so the whole team stays aligned.

## How It Works (with Team Sync)

1. **Voice Memo Entry**  
   Workers speak a quick explanation:  
   “Item missing from aisle 3 because it was moved to the freezer due to heat damage.”

2. **Memo Processing**  
   Transcribed with Whisper, tagged with item + location, embedded via Ollama and stored.

3. **Daily Sync**  
   A cron job (scheduled task) runs at a set time to:  
   - Pull all new discrepancy memos  
   - Summarize them using the local LLM (like “4 new product issues reported today”)  
   - Format as a short, readable briefing (Markdown, HTML email, or printable page)  
   - Push to dashboard, email list, or even Slack/webhook integration

## Example Daily Update

**EchoShelf Daily Report – April 7, 2025**

1. **Aisle 4, Canned Goods** — Item moved to endcap due to space issue  
2. **Freezer Section** — Two damaged items discarded, count adjusted  
3. **Produce Bay 2** — Apples miscounted due to old labels remaining in bin  

Notes entered by: Daniel, Jamie

## Stretch Goals

- Role-based access (manager vs employee views)
- QR or barcode scanning to tag locations
- Integration with handheld scanner tools
- Multilingual transcription support (for diverse team environments)

---

## Project Structure

/EchoShelf
├── /backend
│   ├── /src
│   │   ├── /app.py             # FastAPI app for backend
│   │   ├── /whisper.py         # Whisper transcription logic
│   │   ├── /llm.py             # Ollama-based query generation
│   │   ├── /data.py            # DB models, data handling
│   │   ├── /cron.py            # Cron job for daily summary generation
│   │   ├── /utils.py           # Helper functions (e.g. email sending)
│   ├── Dockerfile              # Docker container for backend
│   ├── requirements.txt        # Python dependencies
├── /frontend
│   ├── /pages
│   │   ├── index.js            # Landing page for the app
│   │   ├── /memo.js            # Page to record and view voice memos
│   ├── /components
│   │   ├── MemoList.js         # List view of memos with search
│   │   ├── MemoForm.js         # Form to record new memos
│   ├── /public
│   │   ├── /styles.css         # Basic styling
│   ├── package.json            # Node dependencies
│   ├── next.config.js          # Next.js configuration
├── /scripts
│   ├── /generate_daily_report.py # Script to generate and push daily summary
├── /docs
│   ├── /README.md             # Documentation
├── docker-compose.yml         # Docker Compose configuration for full stack

---

## Backend Setup (FastAPI with Whisper + Ollama)

### 1. `backend/src/app.py` - FastAPI Setup

```python
from fastapi import FastAPI, HTTPException
from whisper import transcribe
from llm import query_llm
from data import save_memo, get_memos
from cron import generate_daily_report

app = FastAPI()

@app.post("/add_memo")
async def add_memo(item: str, location: str, voice_file: str):
    transcription = transcribe(voice_file)
    if not transcription:
        raise HTTPException(status_code=400, detail="Transcription failed")
    memo = save_memo(item, location, transcription)
    return {"message": "Memo saved successfully", "memo": memo}

@app.get("/memos")
async def get_all_memos():
    memos = get_memos()
    return {"memos": memos}

@app.get("/generate_report")
async def generate_report():
    report = generate_daily_report()
    return {"report": report}

2. backend/src/whisper.py - Transcription Logic (Whisper)

import whisper

def transcribe(audio_file: str) -> str:
    model = whisper.load_model("base")  # Use Whisper's base model
    result = model.transcribe(audio_file)
    return result['text']

3. backend/src/llm.py - Query Handling (Ollama)

import ollama

def query_llm(query: str) -> str:
    response = ollama.chat(model="llama2", messages=[{"role": "user", "content": query}])
    return response['text']

4. backend/src/data.py - Data Handling (Simple DB Storage)

import sqlite3

def init_db():
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS memos 
                      (id INTEGER PRIMARY KEY, item TEXT, location TEXT, transcription TEXT)''')
    conn.commit()
    conn.close()

def save_memo(item, location, transcription):
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO memos (item, location, transcription) VALUES (?, ?, ?)",
                   (item, location, transcription))
    conn.commit()
    conn.close()

def get_memos():
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT item, location, transcription FROM memos")
    memos = cursor.fetchall()
    conn.close()
    return memos

5. backend/src/cron.py - Daily Summary Generation

import smtplib
from email.mime.text import MIMEText
from data import get_memos

def generate_daily_report():
    memos = get_memos()
    report = "\n".join([f"Item: {memo[0]}, Location: {memo[1]}, Memo: {memo[2]}" for memo in memos])
    # Send report via email
    send_email(report)
    return report

def send_email(report):
    msg = MIMEText(report)
    msg['Subject'] = "Daily Inventory Discrepancy Report"
    msg['From'] = 'sender@example.com'
    msg['To'] = 'team@example.com'

    with smtplib.SMTP('smtp.example.com') as server:
        server.send_message(msg)
```


⸻

Frontend Setup (Next.js)
```javascript
1. frontend/pages/index.js - Main Page

import Link from "next/link";

export default function Home() {
  return (
    <div>
      <h1>Welcome to EchoShelf</h1>
      <p>Record and view inventory discrepancies in real-time!</p>
      <Link href="/memo">Record New Memo</Link>
    </div>
  );
}

2. frontend/pages/memo.js - Memo Recording Page

import { useState } from "react";

export default function Memo() {
  const [file, setFile] = useState(null);
  const [item, setItem] = useState("");
  const [location, setLocation] = useState("");

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append("item", item);
    formData.append("location", location);
    formData.append("voice_file", file);

    const response = await fetch("/api/add_memo", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    console.log(data);
  };

  return (
    <div>
      <h1>Record Memo</h1>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="Item"
          value={item}
          onChange={(e) => setItem(e.target.value)}
        />
        <input
          type="text"
          placeholder="Location"
          value={location}
          onChange={(e) => setLocation(e.target.value)}
        />
        <input type="file" onChange={handleFileChange} />
        <button type="submit">Submit</button>
      </form>
    </div>
  );
}
```


⸻

Docker Setup
```yml
docker-compose.yml - Full Stack Setup

version: '3.8'
services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - ./backend:/app
    command: uvicorn src.app:app --reload

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    volumes:
      - ./frontend:/app
    command: next dev
```


⸻
