# Sample Test Cases and Conversations

Use these commands after starting the backend on `127.0.0.1:8000`.

---

## 1. Health / Architecture Check

```bash
curl http://127.0.0.1:8000/health
```

Expected:

- `status` should be `ok`
- `vector_store` should show `faiss.IndexFlatIP` when FAISS is installed
- `graph_enabled` should be `true` when LangGraph is installed
- `embedding_provider` should show either OpenAI embeddings or local hashing embeddings

---

## 2. Event + Color Styling Recommendation

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"style-test","message":"Suggest a pastel saree look for a day wedding"}'
```

Expected behavior:

- Intent: `style_recommendation`
- Entity extraction should include event/color/time of day
- Sources should include relevant saree/fashion-rule documents
- Recommendations should include a saree and/or blouse option

---

## 3. Context-Aware Follow-up

Run after test 2 with the same `session_id`:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"style-test","message":"make it more modern and minimal"}'
```

Expected behavior:

- The bot should remember the wedding/pastel context from the previous turn.
- It should not ask from scratch.
- It should suggest a modern/minimal direction.

---

## 4. Blouse Recommendation

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"blouse-test","message":"What blouse suits a Banarasi silk saree for a reception?"}'
```

Expected behavior:

- Intent: `blouse_info`
- Sources should include blouse and fashion-rule documents
- Recommendation should include a high-neck, embroidered, or structured blouse option

---

## 5. Haldi Styling

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"haldi-test","message":"I need something yellow and comfortable for haldi"}'
```

Expected behavior:

- Intent: `style_recommendation`
- Entity extraction should include haldi and yellow
- Sources should include haldi/mehendi styling rules
- Response should mention bright, comfortable, festive styling

---

## 6. Platform FAQ

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"faq-test","message":"How can I track my order?"}'
```

Expected behavior:

- Intent: `faq`
- Response should come from FAQ knowledge base
- It should not invent unsupported platform policy details

---

## 7. Memory Inspection

```bash
curl http://127.0.0.1:8000/memory/style-test
```

Expected behavior:

- Shows extracted entities
- Shows recent chat history
- Shows last intent

---

## 8. Rebuild RAG Index

```bash
curl -X POST http://127.0.0.1:8000/kb/rebuild
```

Expected behavior:

- Reloads JSON knowledge base
- Recreates embeddings
- Rebuilds FAISS index
- Returns counts and retrieval backend details
