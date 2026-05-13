# Fashion-Tech RAG Conversational AI Chatbot

## 1. Project Overview

Fashion-Tech RAG Conversational AI Chatbot is a full-stack GenAI application for saree and blouse styling assistance, event-based outfit recommendations, color and mood guidance, and fashion platform FAQs.

The project combines a FastAPI backend, a React/Vite frontend, OpenAI-compatible response generation, FAISS retrieval, LangGraph workflow orchestration, short-term session memory, and a JSON-backed fashion knowledge base. It can run with OpenAI models and embeddings when configured, or in local fallback mode without an API key.

## 2. Features

- Natural-language fashion styling conversations
- Saree, blouse, event, color, fabric, mood, and budget-aware recommendations
- Intent recognition for styling, product information, color guidance, FAQs, clarification, and fallback handling
- Short-term memory for contextual follow-up questions
- RAG pipeline over a JSON knowledge base
- FAISS vector search with embeddings-backed retrieval
- Entity-aware reranking of retrieved documents
- LangGraph-based backend conversation flow
- Optional OpenAI response generation
- Local fallback responses and local deterministic embeddings when OpenAI is not configured
- Structured API responses with intent, reply, recommendations, sources, and context
- React/Vite chat interface for frontend usage

## 3. Architecture

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

The backend keeps the conversation pipeline modular: natural-language understanding, preference extraction, retrieval, response generation, and memory updates are separate steps. This makes the system easier to test, extend, and adapt to additional fashion categories.

## 4. Backend Flow

### Intent Recognition

The backend classifies each message into one of the following intents:

- `greeting`
- `saree_info`
- `blouse_info`
- `style_recommendation`
- `color_recommendation`
- `faq`
- `clarification`
- `fallback`

Intent recognition uses deterministic domain rules first, then passes the detected intent and retrieved context into response generation.

### Entity Extraction

The bot extracts styling signals such as:

- event: wedding, haldi, reception, office, party, festival
- color: pastel, red, black, gold, emerald, and similar values
- fabric: silk, organza, georgette, cotton, linen, and related materials
- mood: modern, royal, minimal, elegant, bold, and similar preferences
- budget, when mentioned
- time of day: day or evening

Extracted entities are stored in session memory so follow-up messages can reuse earlier context.

```text
User: I need a pastel saree for a wedding.
Bot: ...
User: make it more modern and minimal
```

The second message is interpreted using the earlier wedding and pastel preferences.

### Response Contract

The `/chat` endpoint returns structured data for UI rendering:

```json
{
  "session_id": "style-user",
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

## 5. RAG + FAISS Retrieval

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
5. On restart, the saved index is loaded when the knowledge base fingerprint and embedding provider match.
6. User queries are embedded and searched against FAISS.
7. Results are reranked using vector similarity, intent/category boosts, and entity matches.
8. Retrieved documents are passed into response generation and returned as structured `sources`.

### Embeddings

The project supports two embedding modes:

1. OpenAI embeddings using `text-embedding-3-large` when `OPENAI_API_KEY` is set.
2. Local deterministic hashing embeddings when no API key is available.

The local fallback still builds dense vectors and uses FAISS, so retrieval remains available without external model access.

## 6. LangGraph Conversation Flow

The chatbot pipeline is implemented as a LangGraph state machine:

```text
load_memory → understand_intent → extract_preferences → retrieve_context → generate_response → update_memory
```

Each node handles one stage of the conversation lifecycle:

- `load_memory`: loads recent session state.
- `understand_intent`: identifies the user’s current request.
- `extract_preferences`: extracts event, color, fabric, mood, budget, and time-of-day signals.
- `retrieve_context`: searches the FAISS index and assembles RAG context.
- `generate_response`: creates a grounded OpenAI response or deterministic local fallback response.
- `update_memory`: stores the latest turn and extracted preferences.

## 7. Folder Structure

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

## 8. Setup Instructions

### Backend Setup on macOS/Linux

```bash
cd fashion-ai-chatbot/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Backend Setup on Windows PowerShell

```powershell
cd fashion-ai-chatbot\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

### Frontend Setup

```bash
cd fashion-ai-chatbot/frontend
npm install
cp .env.example .env
```

## 9. Environment Variables

### Backend

Configure backend settings in `backend/.env`.

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-5.4-mini
USE_OPENAI=true

OPENAI_EMBEDDING_MODEL=text-embedding-3-large
USE_OPENAI_EMBEDDINGS=true
```

The project works without OpenAI by using local fallback responses and local deterministic embeddings. If OpenAI is enabled, restart the backend after updating environment variables.

The backend uses the OpenAI Responses API and avoids setting `temperature` by default because some reasoning-capable models may reject unsupported generation parameters.

### Frontend

Configure the API base URL in `frontend/.env`.

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## 10. Running Backend

Run the FastAPI server from the backend directory:

```bash
cd fashion-ai-chatbot/backend
source .venv/bin/activate
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

On Windows PowerShell:

```powershell
cd fashion-ai-chatbot\backend
.\.venv\Scripts\Activate.ps1
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Backend URLs:

```text
Health: http://127.0.0.1:8000/health
Docs:   http://127.0.0.1:8000/docs
Chat:   POST http://127.0.0.1:8000/chat
```

## 11. Running Frontend

Run the Vite development server from the frontend directory:

```bash
cd fashion-ai-chatbot/frontend
npm run dev
```

Open the local Vite URL:

```text
http://localhost:5173
```

## 12. API Test Commands

### Health Check

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

### Style Recommendation

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"style-user","message":"Suggest a pastel saree look for a day wedding"}'
```

### Context Follow-Up

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"style-user","message":"make it more modern and minimal"}'
```

### FAQ Handling

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"faq-user","message":"How can I track my order?"}'
```

### Read Session Memory

```bash
curl http://127.0.0.1:8000/memory/style-user
```

### Rebuild Knowledge Base and FAISS Index

```bash
curl -X POST http://127.0.0.1:8000/kb/rebuild
```

## 13. Example Conversations

```text
User: I need a saree for my friend's wedding. I like pastel colors.
Bot: For a pastel wedding look, start with a lightweight organza saree in blush, mint, or powder blue with subtle embroidery.

User: make it more modern and minimal
Bot: Since the direction is pastel and wedding-appropriate, keep the saree soft and pair it with a clean pearl-work or sleeveless blouse.

User: what blouse would work best?
Bot: A pearl-work blouse works well because it keeps the outfit polished without making the look too heavy.

User: something lighter
Bot: Choose organza or chiffon, keep the border delicate, and use minimal jewellery so the look stays airy.

User: what about for a reception?
Bot: For a reception, move toward embellished georgette or silk if you want a richer evening look.
```

```text
User: How can I track my order?
Bot: You can track your order from the order section of your account. Use the tracking link once the shipment is dispatched.
```

## 14. Troubleshooting

- If OpenAI calls fail, confirm `OPENAI_API_KEY` is set in `backend/.env`, then restart FastAPI.
- If the configured OpenAI model is unavailable for your account, update `OPENAI_MODEL` to a model your account can access.
- If embeddings fail, the app can use local deterministic embeddings when OpenAI embeddings are unavailable.
- If retrieval looks stale after editing JSON files, call `POST /kb/rebuild`.
- If the frontend cannot connect, confirm `frontend/.env` has `VITE_API_BASE_URL=http://127.0.0.1:8000` and that the backend is running.
- If dependencies fail to install, verify the active Python environment for the backend and the installed Node.js version for the frontend.

## Future Improvements

- Add persistent memory using Redis or PostgreSQL.
- Add product catalog integration.
- Add user profile-based personalization.
- Add authentication for production deployment.
- Add evaluation metrics for retrieval quality and response accuracy.