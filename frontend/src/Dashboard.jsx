import React, { useRef, useState } from "react";
import {
  HeartPulse,
  LayoutDashboard,
  History,
  Info,
  Circle,
  Upload,
  X,
  Activity,
  Sparkles,
  BookOpen,
  Send,
  Bot,
  User,
  ShieldAlert,
  CheckCircle2,
} from "lucide-react";
import "./Dashboard.css";

/* -------------------------------------------------------------------------- */
/* API config                                                                  */
/* -------------------------------------------------------------------------- */

// Adjust to match wherever the FastAPI backend is served from.
// Leave as an empty string if the frontend is served from the same origin
// (e.g. behind a reverse proxy) so requests hit "/analyze" directly.
const API_BASE_URL = "http://localhost:8000";

const DEFAULT_QUESTION = "What does this result mean for the patient?";

/* -------------------------------------------------------------------------- */
/* Static / placeholder data                                                  */
/* -------------------------------------------------------------------------- */

const VALVE_CONFIG = [
  { key: "av", label: "Aortic Valve", short: "AV" },
  { key: "mv", label: "Mitral Valve", short: "MV" },
  { key: "pv", label: "Pulmonary Valve", short: "PV" },
  { key: "tv", label: "Tricuspid Valve", short: "TV" },
];

const INITIAL_MESSAGES = [
  {
    id: 1,
    role: "assistant",
    text: "Hi, I'm the RosaNet assistant. Once your analysis is ready, ask me anything about the result.",
  },
  {
    id: 2,
    role: "user",
    text: "What does a heart murmur mean?",
  },
  {
    id: 3,
    role: "assistant",
    text: "Based on the uploaded recordings, further evaluation with echocardiography may be recommended.",
  },
];

/* -------------------------------------------------------------------------- */
/* Sidebar                                                                    */
/* -------------------------------------------------------------------------- */

function Sidebar({ activeNav, setActiveNav }) {
  const navItems = [
    { key: "dashboard", label: "Dashboard", icon: LayoutDashboard },
    { key: "history", label: "Analysis History", icon: History },
    { key: "about", label: "About", icon: Info },
  ];

  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="brand-mark">
          <HeartPulse size={20} strokeWidth={2.4} />
        </div>
        <div className="brand-text">
          <span className="brand-title">Heart Disease</span>
          <span className="brand-subtitle">RAG Assistant</span>
        </div>
      </div>

      <nav className="sidebar-nav">
        {navItems.map((item) => {
          const ActiveIcon = item.icon;
          const isActive = activeNav === item.key;
          return (
            <button
              key={item.key}
              className={`nav-item${isActive ? " nav-item-active" : ""}`}
              onClick={() => setActiveNav(item.key)}
              type="button"
            >
              <ActiveIcon size={17} strokeWidth={2} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      <div className="sidebar-footer">
        <div className="model-status">
          <div className="status-row">
            <span className="status-dot" />
            <span className="status-label">RosaNet Online</span>
          </div>
          <span className="status-meta">Inference model &middot; v2.3</span>
        </div>
      </div>
    </aside>
  );
}

/* -------------------------------------------------------------------------- */
/* Upload Section                                                             */
/* -------------------------------------------------------------------------- */

function Waveform({ active }) {
  // Deterministic pseudo-random bar heights so each card looks distinct
  const bars = Array.from({ length: 34 }, (_, i) => {
    const seed = Math.sin(i * 12.9898) * 43758.5453;
    const frac = seed - Math.floor(seed);
    return 6 + Math.round(frac * 22);
  });

  return (
    <div className={`waveform${active ? " waveform-active" : ""}`}>
      {bars.map((h, i) => (
        <span key={i} className="waveform-bar" style={{ height: `${h}px` }} />
      ))}
    </div>
  );
}

function UploadCard({ valve, file, onFileChange, onRemove }) {
  const inputRef = useRef(null);

  return (
    <div className={`upload-card${file ? " upload-card-filled" : ""}`}>
      <div className="upload-card-head">
        <span className={`valve-chip valve-chip-${valve.key}`}>{valve.short}</span>
        <span className="upload-card-title">{valve.label}</span>
      </div>

      {file ? (
        <>
          <Waveform active />
          <div className="upload-file-row">
            <span className="upload-file-name" title={file.name}>
              {file.name}
            </span>
            <button
              type="button"
              className="icon-btn"
              onClick={() => onRemove(valve.key)}
              aria-label={`Remove ${valve.label} recording`}
            >
              <X size={14} />
            </button>
          </div>
        </>
      ) : (
        <button
          type="button"
          className="upload-dropzone"
          onClick={() => inputRef.current?.click()}
        >
          <Upload size={22} strokeWidth={1.8} />
          <span className="upload-dropzone-text">Drop or browse</span>
        </button>
      )}

      <input
        ref={inputRef}
        type="file"
        accept="audio/*,.wav"
        hidden
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFileChange(valve.key, f);
          e.target.value = "";
        }}
      />

      <div className="upload-card-footer">
        <span className="format-hint">WAV &middot; 5&ndash;15s</span>
        <button
          type="button"
          className="ghost-btn"
          onClick={() => inputRef.current?.click()}
        >
          {file ? "Replace" : "Upload"}
        </button>
      </div>
    </div>
  );
}

function UploadSection({ files, setFiles, onAnalyze, isAnalyzing, errorMsg }) {
  const handleFileChange = (key, file) => {
    setFiles((prev) => ({ ...prev, [key]: file }));
  };

  const handleRemove = (key) => {
    setFiles((prev) => ({ ...prev, [key]: null }));
  };

  const uploadedCount = Object.values(files).filter(Boolean).length;
  const allUploaded = uploadedCount === VALVE_CONFIG.length;

  return (
    <section className="panel upload-panel">
      <div className="panel-head">
        <div>
          <h2 className="panel-title">Upload Recordings</h2>
          <p className="panel-subtitle">
            Add a phonocardiogram recording for each auscultation point.
          </p>
        </div>
        <span className="count-pill">{uploadedCount} / 4 ready</span>
      </div>

      <div className="upload-grid">
        {VALVE_CONFIG.map((valve) => (
          <UploadCard
            key={valve.key}
            valve={valve}
            file={files[valve.key]}
            onFileChange={handleFileChange}
            onRemove={handleRemove}
          />
        ))}
      </div>

      <button
        type="button"
        className="primary-btn analyze-btn"
        onClick={onAnalyze}
        disabled={isAnalyzing || !allUploaded}
      >
        <Activity size={18} strokeWidth={2.2} />
        {isAnalyzing ? "Analyzing heart sounds..." : "Analyze Heart Sounds"}
      </button>

      {errorMsg && (
        <p
          className="format-hint"
          style={{ color: "var(--danger)", marginTop: "10px", fontSize: "12px" }}
        >
          {errorMsg}
        </p>
      )}
    </section>
  );
}

/* -------------------------------------------------------------------------- */
/* Prediction Section                                                        */
/* -------------------------------------------------------------------------- */

function PredictionSection({ result }) {
  if (!result) {
    return (
      <section className="panel prediction-panel prediction-empty">
        <div className="empty-state">
          <HeartPulse size={26} strokeWidth={1.6} />
          <p>Run an analysis to see the AI prediction here.</p>
        </div>
      </section>
    );
  }

  const { label, confidence, explanation } = result;
  const isPresent = label === "Present";

  return (
    <section className="panel prediction-panel">
      <div className="panel-head">
        <h2 className="panel-title">Prediction</h2>
        <span className={`badge ${isPresent ? "badge-warning" : "badge-success"}`}>
          {isPresent ? "Attention advised" : "Normal range"}
        </span>
      </div>

      <div className="prediction-body">
        <div className={`prediction-icon ${isPresent ? "prediction-icon-warn" : "prediction-icon-ok"}`}>
          <HeartPulse size={30} strokeWidth={2} />
        </div>

        <div className="prediction-main">
          <span className="prediction-label-caption">Prediction</span>
          <span className="prediction-value">{label}</span>
        </div>

        <div className="prediction-confidence">
          <div className="confidence-row">
            <span className="prediction-label-caption">Confidence</span>
            <span className="confidence-value">{confidence}%</span>
          </div>
          <div className="confidence-track">
            <div
              className={`confidence-fill ${isPresent ? "confidence-fill-warn" : "confidence-fill-ok"}`}
              style={{ width: `${confidence}%` }}
            />
          </div>
        </div>
      </div>

      <p className="prediction-explanation">{explanation}</p>
    </section>
  );
}

/* -------------------------------------------------------------------------- */
/* RAG Evidence Section                                                       */
/* -------------------------------------------------------------------------- */

const DEFAULT_RAG_TEXT =
  "Heart murmurs are extra or unusual sounds heard during a heartbeat, caused by turbulent blood flow through the heart valves or nearby blood vessels. They can be innocent and harmless, or can signal an underlying structural issue such as valve stenosis or regurgitation. Diagnosis typically combines auscultation findings with imaging, most commonly an echocardiogram, to confirm the source and severity.";

function RagSection({ result }) {
  const sources = result?.sources ?? [];

  return (
    <section className="panel rag-panel">
      <div className="panel-head">
        <div>
          <h2 className="panel-title">Evidence-Based Explanation</h2>
          <p className="panel-subtitle">
            Context retrieved from trusted clinical references.
          </p>
        </div>
        <div className="panel-head-icon">
          <BookOpen size={18} strokeWidth={2} />
        </div>
      </div>

      <p className="rag-text">{result ? result.ragExplanation : DEFAULT_RAG_TEXT}</p>

      {sources.length > 0 && (
        <div className="sources-block">
          <span className="sources-label">Sources</span>
          <div className="chip-row">
            {sources.map((src, idx) => (
              <span
                key={`${src.title ?? src.source ?? "source"}-${idx}`}
                className="source-chip"
                title={src.section ? `${src.source} \u2014 ${src.section}` : src.source}
              >
                {src.title || src.source}
              </span>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

/* -------------------------------------------------------------------------- */
/* Chat Section                                                               */
/* -------------------------------------------------------------------------- */

function ChatSection({ hasAnalysis }) {
  const [messages, setMessages] = useState(INITIAL_MESSAGES);
  const [draft, setDraft] = useState("");
  const [isSending, setIsSending] = useState(false);

  const handleSend = async () => {
    const trimmed = draft.trim();
    if (!trimmed || isSending || !hasAnalysis) return;

    const userMsg = { id: Date.now(), role: "user", text: trimmed };
    const thinkingId = Date.now() + 1;
    const thinkingMsg = {
      id: thinkingId,
      role: "assistant",
      text: "RosaNet is thinking...",
    };

    setMessages((prev) => [...prev, userMsg, thinkingMsg]);
    setDraft("");
    setIsSending(true);

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed }),
      });

      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}.`);
      }

      const data = await response.json();

      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === thinkingId ? { ...msg, text: data.answer } : msg
        )
      );
    } catch (err) {
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === thinkingId
            ? { ...msg, text: "Sorry, I couldn't reach the RosaNet backend." }
            : msg
        )
      );
    } finally {
      setIsSending(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <section className="panel chat-panel">
      <div className="panel-head">
        <div>
          <h2 className="panel-title">Ask RosaNet</h2>
          <p className="panel-subtitle">Follow up on your result in plain language.</p>
        </div>
        <div className="panel-head-icon">
          <Sparkles size={18} strokeWidth={2} />
        </div>
      </div>

      <div className="chat-thread">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`chat-row ${msg.role === "user" ? "chat-row-user" : "chat-row-assistant"}`}
          >
            {msg.role === "assistant" && (
              <div className="chat-avatar chat-avatar-assistant">
                <Bot size={15} />
              </div>
            )}
            <div className={`chat-bubble ${msg.role === "user" ? "chat-bubble-user" : "chat-bubble-assistant"}`}>
              {msg.text}
            </div>
            {msg.role === "user" && (
              <div className="chat-avatar chat-avatar-user">
                <User size={15} />
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="chat-input-row">
        <input
          type="text"
          className="chat-input"
          placeholder={
            hasAnalysis ? "Ask about your result..." : "Analyze recordings first..."
          }
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isSending || !hasAnalysis}
        />
        <button
          type="button"
          className="send-btn"
          onClick={handleSend}
          aria-label="Send message"
          disabled={isSending || !hasAnalysis}
        >
          <Send size={16} />
        </button>
      </div>
    </section>
  );
}

/* -------------------------------------------------------------------------- */
/* Header                                                                     */
/* -------------------------------------------------------------------------- */

function Header() {
  return (
    <header className="content-header">
      <h1 className="content-title">Heart Disease AI Analysis</h1>
      <p className="content-subtitle">
        Upload heart sound recordings and receive AI-powered diagnosis with
        evidence-based explanations.
      </p>
    </header>
  );
}

/* -------------------------------------------------------------------------- */
/* Disclaimer                                                                 */
/* -------------------------------------------------------------------------- */

function Disclaimer() {
  return (
    <footer className="disclaimer">
      <ShieldAlert size={15} strokeWidth={2} />
      <span>
        This tool is a research prototype and does not provide medical
        diagnoses. Always consult a licensed clinician for interpretation of
        heart sound findings.
      </span>
    </footer>
  );
}

/* -------------------------------------------------------------------------- */
/* Dashboard (root)                                                           */
/* -------------------------------------------------------------------------- */

export default function Dashboard() {
  const [activeNav, setActiveNav] = useState("dashboard");
  const [files, setFiles] = useState({ av: null, mv: null, pv: null, tv: null });
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);

  const handleAnalyze = async () => {
    const { av, mv, pv, tv } = files;

    if (!av || !mv || !pv || !tv) {
      setErrorMsg("Please upload a recording for all four valves before analyzing.");
      return;
    }

    setIsAnalyzing(true);
    setErrorMsg(null);
    setResult(null);

    const formData = new FormData();
    formData.append("av", av);
    formData.append("mv", mv);
    formData.append("pv", pv);
    formData.append("tv", tv);
    formData.append("question", DEFAULT_QUESTION);

    try {
      const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        let detail = `Request failed with status ${response.status}.`;
        try {
          const errBody = await response.json();
          if (errBody?.detail) detail = errBody.detail;
        } catch {
          // response body wasn't JSON; fall back to the generic message
        }
        throw new Error(detail);
      }

      const data = await response.json();

      const confidencePct = Math.round((data.confidence ?? 0) * 1000) / 10;

      const probabilitySummary = data.probabilities
        ? Object.entries(data.probabilities)
            .map(([k, v]) => `${k} ${Math.round(v * 100)}%`)
            .join(" \u00b7 ")
        : "";

      setResult({
        label: data.prediction,
        confidence: confidencePct,
        explanation: probabilitySummary
          ? `Model probabilities \u2014 ${probabilitySummary}.`
          : "",
        ragExplanation: data.answer,
        sources: data.sources ?? [],
      });
    } catch (err) {
      setErrorMsg(
        err instanceof Error
          ? err.message
          : "Something went wrong while analyzing the recordings. Please try again."
      );
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="app-shell">
      <Sidebar activeNav={activeNav} setActiveNav={setActiveNav} />

      <main className="main-content">
        <Header />

        <div className="content-grid">
          <UploadSection
            files={files}
            setFiles={setFiles}
            onAnalyze={handleAnalyze}
            isAnalyzing={isAnalyzing}
            errorMsg={errorMsg}
          />
          <PredictionSection result={result} />
          <RagSection result={result} />
          <ChatSection hasAnalysis={result !== null} />
        </div>

        <Disclaimer />
      </main>
    </div>
  );
}