import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Bot, Send, Sparkles, RefreshCcw, ShieldCheck, Shirt, HelpCircle, Palette, Database, GitBranch } from 'lucide-react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

function getSessionId() {
  const existing = localStorage.getItem('fashion_chat_session_id');
  if (existing) return existing;
  const next = crypto.randomUUID();
  localStorage.setItem('fashion_chat_session_id', next);
  return next;
}

const starterPrompts = [
  'Suggest a pastel saree look for a day wedding',
  'What blouse suits a Banarasi silk saree?',
  'I need something yellow for haldi',
  'Make it more modern and minimal',
  'How can I track my order?',
];

const featureCards = [
  { icon: Shirt, title: 'Stylist-level reasoning', text: 'Event, fabric, color, mood, and blouse-pairing-aware suggestions.' },
  { icon: Database, title: 'FAISS RAG retrieval', text: 'Embeds saree, blouse, fashion-rule, and FAQ knowledge for semantic search.' },
  { icon: GitBranch, title: 'LangGraph flow', text: 'Routes every turn through context, intent, retrieval, generation, and memory.' },
  { icon: HelpCircle, title: 'Help-desk FAQs', text: 'Grounded answers for order tracking, returns, delivery, payment, and customization.' },
];

function MessageBubble({ message }) {
  const isUser = message.role === 'user';
  const contextEntries = Object.entries(message.context?.entities || {}).filter(([, value]) => value !== undefined && value !== null && value !== '');
  return (
    <div className={`message-row ${isUser ? 'user-row' : 'bot-row'}`}>
      {!isUser && (
        <div className="avatar bot-avatar">
          <Bot size={18} />
        </div>
      )}
      <div className={`message-bubble ${isUser ? 'user-bubble' : 'bot-bubble'}`}>
        <p>{message.content}</p>
        {message.recommendations?.length > 0 && (
          <div className="recommendation-grid">
            {message.recommendations.map((item) => (
              <div className="recommendation-card" key={item.source_id}>
                <span>{item.category}</span>
                <strong>{item.name}</strong>
                <p>{item.reason}</p>
              </div>
            ))}
          </div>
        )}
        {message.sources?.length > 0 && (
          <details className="source-details">
            <summary>Retrieved sources</summary>
            <div className="source-list">
              {message.sources.map((source) => (
                <div className="source-chip" key={source.id}>
                  <strong>{source.name}</strong>
                  <span>{source.category} · score {source.score}</span>
                </div>
              ))}
            </div>
          </details>
        )}
        {(contextEntries.length > 0 || message.context?.memory) && (
          <div className="context-chip-row">
            {contextEntries.map(([key, value]) => (
              <span className="context-chip" key={key}>
                <Palette size={12} />
                {key.replaceAll('_', ' ')}: {String(value)}
              </span>
            ))}
            {message.context?.memory && (
              <span className="context-chip">
                Memory: {message.context.memory.history_turns || 0} turns
              </span>
            )}
          </div>
        )}
        {message.meta && (
          <div className="meta-line">
            <span>Intent: {message.meta.intent}</span>
            <span>{message.meta.used_llm ? 'OpenAI response' : 'Local response'}</span>
            <span>{message.meta.rag_enabled ? 'RAG on' : 'RAG off'}</span>
            <span>{message.meta.graph_enabled ? 'LangGraph' : 'Manual graph fallback'}</span>
          </div>
        )}
      </div>
    </div>
  );
}

export default function App() {
  const [sessionId, setSessionId] = useState(getSessionId());
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content:
        'Hi! I’m Nira, your saree and blouse styling assistant. Tell me the occasion, color, mood, or saree type you have in mind — I’ll use semantic retrieval and context to guide the look.',
      recommendations: [],
    },
  ]);
  const chatEndRef = useRef(null);

  const canSend = useMemo(() => input.trim().length > 0 && !isLoading, [input, isLoading]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  async function sendMessage(customText) {
    const text = (customText || input).trim();
    if (!text || isLoading) return;

    setError('');
    setInput('');
    setMessages((prev) => [...prev, { role: 'user', content: text }]);
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, message: text }),
      });

      if (!response.ok) {
        const details = await response.json().catch(() => ({}));
        throw new Error(details.detail || `API request failed with status ${response.status}`);
      }

      const data = await response.json();
      if (data.session_id && data.session_id !== sessionId) {
        setSessionId(data.session_id);
        localStorage.setItem('fashion_chat_session_id', data.session_id);
      }

      const assistantText = data.follow_up_question ? `${data.reply}\n\n${data.follow_up_question}` : data.reply;
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: assistantText,
          recommendations: data.recommendations || [],
          sources: data.sources || [],
          context: data.context || { entities: data.entities || {} },
          meta: {
            intent: data.intent,
            used_llm: data.used_llm,
            rag_enabled: data.rag_enabled,
            graph_enabled: data.graph_enabled,
          },
        },
      ]);
    } catch (err) {
      setError(err.message || 'Something went wrong while contacting the backend.');
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: 'I could not reach the backend. Please make sure FastAPI is running on the configured API URL.',
          recommendations: [],
          sources: [],
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  }

  function resetChat() {
    const next = crypto.randomUUID();
    localStorage.setItem('fashion_chat_session_id', next);
    setSessionId(next);
    setMessages([
      {
        role: 'assistant',
        content: 'New styling session started. What outfit or platform question can I help with?',
        recommendations: [],
      },
    ]);
    setError('');
  }

  return (
    <main className="app-shell">
      <section className="hero-panel">
        <div className="brand-pill">
          <Sparkles size={16} />
          Fashion-Tech AI Demo
        </div>
        <h1>Saree & Blouse Conversational Assistant</h1>
        <p className="hero-copy">
          A GenAI fashion assistant with FAISS vector search, RAG, embeddings, LangGraph state flow, short-term memory, and optional OpenAI-powered responses.
        </p>

        <div className="feature-list">
          {featureCards.map((card) => {
            const Icon = card.icon;
            return (
              <div className="feature-card" key={card.title}>
                <Icon size={20} />
                <div>
                  <strong>{card.title}</strong>
                  <p>{card.text}</p>
                </div>
              </div>
            );
          })}
        </div>

        <div className="tech-card">
          <ShieldCheck size={20} />
          <div>
            <strong>Architecture</strong>
            <p>React UI → FastAPI /chat → LangGraph → intent + entities → FAISS RAG → OpenAI/local response → memory + recommendations.</p>
          </div>
        </div>
      </section>

      <section className="chat-panel">
        <header className="chat-header">
          <div>
            <p className="eyebrow">Live Demo</p>
            <h2>Style Chat</h2>
          </div>
          <button className="icon-button" onClick={resetChat} title="Reset chat">
            <RefreshCcw size={18} />
          </button>
        </header>

        <div className="prompt-row">
          {starterPrompts.map((prompt) => (
            <button key={prompt} onClick={() => sendMessage(prompt)} disabled={isLoading}>
              {prompt}
            </button>
          ))}
        </div>

        <div className="messages-window">
          {messages.map((message, index) => (
            <MessageBubble message={message} key={`${message.role}-${index}`} />
          ))}
          {isLoading && (
            <div className="message-row bot-row">
              <div className="avatar bot-avatar">
                <Bot size={18} />
              </div>
              <div className="message-bubble bot-bubble typing">
                <span />
                <span />
                <span />
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        {error && <div className="error-box">{error}</div>}

        <form
          className="composer"
          onSubmit={(event) => {
            event.preventDefault();
            sendMessage();
          }}
        >
          <input
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="Ask about sarees, blouse designs, colors, events, returns, delivery..."
          />
          <button disabled={!canSend} type="submit">
            <Send size={18} />
            Send
          </button>
        </form>
        <p className="session-text">Session: {sessionId.slice(0, 8)} · API: {API_BASE_URL}</p>
      </section>
    </main>
  );
}
