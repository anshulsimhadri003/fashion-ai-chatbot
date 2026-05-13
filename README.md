# Fashion-Tech RAG Conversational AI Chatbot

A complete assessment-ready GenAI chatbot for a fashion company. The bot assists users with saree and blouse styling, event/color/mood-based recommendations, and platform help-desk FAQs.

This upgraded version is intentionally more realistic than a basic rule-based demo. It includes:

- FastAPI backend
- React/Vite frontend
- LangGraph conversational orchestration
- LangChain `Document`-based knowledge ingestion
- FAISS vector search
- Embeddings-based retrieval
- RAG prompt construction
- Intent recognition and entity extraction
- Short-term session memory
- Optional OpenAI response generation
- Local fallback responses when no API key is configured
- Structured recommendations and retrieved source display

---

## 1. Architecture Overview

```text
User
 ↓
React/Vite Chat UI
 ↓
FastAPI Backend /chat
 ↓
LangGraph Conversation Flow
 ├── load_memory
 ├── understand_intent
 ├── extract_preferences
 │    └── entity extraction from current message + memory
 ├── retrieve_context
 │    ├── embed query
 │    ├── create/load saved FAISS index
 │    ├── hybrid reranking
 │    └── RAG context assembly with sources
 ├── generate_response
 │    ├── OpenAI grounded response, if configured
 │    └── deterministic local fallback
 └── update_memory
      └── short-term memory update
 ↓
Structured response + recommendations + sources + context
```

### Why this design is job-assessment friendly

The assessment asks for a domain-focused conversational chatbot that accepts natural language, maintains context, implements intent recognition, and demonstrates backend/API handling with optional LLM or retrieval systems. This project implements that cleanly with a real RAG flow and a maintainable backend structure.

---

## 2. Backend Flow

### A. Intent Recognition

The backend classifies each message into:

- `greeting`
- `saree_info`
- `blouse_info`
- `style_recommendation`
- `color_recommendation`
- `faq`
- `clarification`
- `fallback`

This uses a practical hybrid approach: deterministic domain rules first, then the LLM uses the detected intent and retrieved context to produce the final answer.

### B. Entity Extraction

The bot extracts styling signals such as:

- event: wedding, haldi, reception, office, party, festival
- color: pastel, red, black, gold, emerald, etc.
- fabric: silk, organza, georgette, cotton, linen, etc.
- mood: modern, royal, minimal, elegant, bold, etc.
- budget, when mentioned
- time of day: day/evening

These are stored in session memory so the bot understands follow-ups like:

```text
User: I need a pastel saree for a wedding.
Bot: ...
User: make it more modern and minimal
```

The second message is interpreted using the earlier wedding + pastel context.

### C. RAG + FAISS Retrieval

Knowledge is stored in JSON files:

```text
backend/knowledge_base/
├── sarees.json
├── blouses.json
├── fashion_rules.json
└── faqs.json
```

At backend startup:

1. JSON records are converted into LangChain `Document` objects.
2. Documents are embedded.
3. Dense vectors are indexed in FAISS.
4. The FAISS index, vector matrix, and metadata are saved under `backend/vector_store/`.
5. On restart, the saved index is loaded when the KB fingerprint and embedding provider match.
6. User queries are embedded and searched against FAISS.
7. Results are reranked using vector similarity + intent/category boost + entity matches.
8. Retrieved documents are passed into the response generation step and returned as structured `sources`.

### D. Embeddings

The project supports two embedding modes:

1. **OpenAI embeddings** using `text-embedding-3-large` when `OPENAI_API_KEY` is set.
2. **Local deterministic hashing embeddings** when no key is available.

The local fallback still builds dense vectors and uses FAISS, so the project remains runnable during demos without paid API access.

### E. LangGraph Flow

The chatbot pipeline is implemented as a LangGraph state machine:

```text
load_memory → understand_intent → extract_preferences → retrieve_context → generate_response → update_memory
```

This makes the app more realistic than a single monolithic `/chat` function and shows agent/workflow engineering knowledge.

---

## 3. Folder Structure

```text
fashion-ai-chatbot/
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── .env.example
│   ├── app/
│   │   ├── config.py
│   │   ├── graph.py
│   │   ├── knowledge.py
│   │   ├── llm.py
│   │   ├── memory.py
│   │   ├── nlp.py
│   │   ├── retriever.py
│   │   └── schemas.py
│   └── knowledge_base/
│       ├── sarees.json
│       ├── blouses.json
│       ├── fashion_rules.json
│       └── faqs.json
│
├── frontend/
│   ├── package.json
│   ├── index.html
│   ├── .env.example
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       └── App.css
│
├── tests/
│   └── sample_test_cases.md
├── .gitignore
└── README.md
```

---

## 4. Backend Setup

### macOS/Linux

```bash
cd fashion-ai-chatbot/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### Windows PowerShell

```powershell
cd fashion-ai-chatbot\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Backend URLs:

```text
Health: http://127.0.0.1:8000/health
Docs:   http://127.0.0.1:8000/docs
Chat:   POST http://127.0.0.1:8000/chat
```

---

## 5. OpenAI Configuration

The project works without OpenAI using local fallback responses and local embeddings. For stronger responses, update `backend/.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-5.4-mini
USE_OPENAI=true

OPENAI_EMBEDDING_MODEL=text-embedding-3-large
USE_OPENAI_EMBEDDINGS=true
```

`gpt-4.1-mini` is a good fallback model if your account or environment does not have access to `gpt-5.4-mini`.

The backend uses the OpenAI Responses API and intentionally avoids setting `temperature` by default, because newer reasoning-capable models may reject unsupported generation parameters.

Then restart the backend.

---

## 6. Frontend Setup

Open a second terminal:

```bash
cd fashion-ai-chatbot/frontend
npm install
cp .env.example .env
npm run dev
```

Open the Vite URL, usually:

```text
http://localhost:5173
```

Default frontend environment:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

---

## 7. API Test Commands

### Health check

```bash
curl http://127.0.0.1:8000/health
```

Expected fields include:

```json
{
  "status": "ok",
  "embedding_provider": "local-hashing-embeddings",
  "vector_store": "faiss.IndexFlatIP",
  "graph_enabled": true
}
```

### Style recommendation

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo-user","message":"Suggest a pastel saree look for a day wedding"}'
```

### Context follow-up

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo-user","message":"make it more modern and minimal"}'
```

### FAQ handling

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"faq-user","message":"How can I track my order?"}'
```

### Read session memory

```bash
curl http://127.0.0.1:8000/memory/demo-user
```

### Rebuild knowledge base and FAISS index

```bash
curl -X POST http://127.0.0.1:8000/kb/rebuild
```

The `/chat` response is structured for UI rendering:

```json
{
  "session_id": "demo-user",
  "intent": "style_recommendation",
  "reply": "Natural stylist response...",
  "recommendations": [],
  "sources": [],
  "context": {
    "entities": {
      "event": "wedding",
      "preferred_color": "pastel",
      "mood": "modern"
    },
    "memory": {
      "last_intent": "style_recommendation",
      "history_turns": 1
    }
  }
}
```

---

## 8. Example Conversation

```text
User: I need a saree for my friend's wedding. I like pastel colors.
Bot: For a pastel wedding look, I’d start with a Pastel Organza Saree...

User: make it more modern and minimal
Bot: Since we’re still working with the pastel wedding direction, keep the organza base but pair it with a Pearl-Work Blouse...

User: what blouse would work best?
Bot: A Pearl-Work Blouse is the strongest choice because it keeps the look elegant, soft, and modern...

User: something lighter
Bot: Keep the direction lightweight with organza or chiffon and avoid heavy jewellery...

User: what about for a reception?
Bot: For a reception, increase the polish with embellished georgette or silk, depending on whether you want modern glam or traditional richness...
```

---

## 9. Validation Commands

Run these from the project root after installing dependencies:

```bash
cd backend
python -m compileall app main.py
uvicorn main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo-user","message":"Suggest a pastel saree look for a day wedding"}'
curl -X POST http://127.0.0.1:8000/kb/rebuild

cd ../frontend
npm run build
```

---

## 10. Troubleshooting

- If OpenAI calls fail, confirm `OPENAI_API_KEY` is set in `backend/.env`, then restart FastAPI.
- If your account cannot use `gpt-5.4-mini`, set `OPENAI_MODEL=gpt-4.1-mini`.
- If embeddings fail, keep `USE_OPENAI_EMBEDDINGS=true`; the app will fall back to local hashing embeddings when no key is available.
- If retrieval looks stale after editing JSON files, call `POST /kb/rebuild`.
- If the frontend cannot connect, check `frontend/.env` has `VITE_API_BASE_URL=http://127.0.0.1:8000` and that the backend is running.

---

## 11. What Makes This More Intelligent

Compared to a basic chatbot, this version includes:

- Semantic document retrieval instead of only keyword matching
- FAISS vector search
- Embeddings-backed query understanding
- Entity-aware reranking
- Short-term conversation memory
- LangGraph workflow orchestration
- RAG-grounded LLM prompting
- Local fallback so demos do not break
- Structured response contract for frontend rendering

---

## 12. Notes for Submission

You can submit this as a GitHub repository with:

- Source code
- Working demo using the React UI or FastAPI docs
- Architecture explanation in this README
- Setup instructions
- Sample test cases under `tests/sample_test_cases.md`

For a strong job submission, record a short demo showing:

1. A wedding recommendation.
2. A natural follow-up that uses memory.
3. A blouse pairing question.
4. A platform FAQ question.
5. The frontend displaying retrieved sources and recommendation cards.
