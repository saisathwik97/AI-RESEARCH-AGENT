
import { useState } from "react";
import "./App.css";

function App() {
  const [question, setQuestion] = useState("");
  const [lastQuestion, setLastQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);

  async function askAgent() {
    if (!question.trim() || loading) return;

    const currentQuestion = question.trim();

    setLastQuestion(currentQuestion);
    setQuestion("");
    setAnswer("");
    setLoading(true);

    try {
      const url = `http://127.0.0.1:8000/ask?question=${encodeURIComponent(
        currentQuestion
      )}`;

      const response = await fetch(url);

      if (!response.ok) {
        throw new Error(`HTTP error: ${response.status}`);
      }

      const data = await response.json();

      setAnswer(data.answer);
    } catch (error) {
      console.error(error);
      setAnswer("Something went wrong. Please check your backend.");
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      askAgent();
    }
  }

  return (
    <div className="chat-app">

      <header className="chat-header">
        <div className="brand">
          <div className="brand-icon">✦</div>

          <div>
            <h2>AI Research Agent</h2>
            <span>Research assistant</span>
          </div>
        </div>

        <div className="status">
          <span className="status-dot"></span>
          Online
        </div>
      </header>


      <main className="chat-content">

        {!lastQuestion && !answer && !loading && (
          <div className="welcome">

            <div className="welcome-icon">✦</div>

            <h1>
              How can I help you
              <span> today?</span>
            </h1>

            <p>
              Ask me anything and I'll research, analyze,
              and explain it for you.
            </p>

            <div className="suggestions">

              <button
                onClick={() =>
                  setQuestion("Explain how large language models work")
                }
              >
                <span>⌁</span>
                Explain how LLMs work
              </button>

              <button
                onClick={() =>
                  setQuestion("What are the latest trends in AI?")
                }
              >
                <span>◈</span>
                Latest AI trends
              </button>

              <button
                onClick={() =>
                  setQuestion("Compare RAG and fine-tuning")
                }
              >
                <span>◇</span>
                Compare RAG vs Fine-tuning
              </button>

            </div>

          </div>
        )}


        {lastQuestion && (
          <div className="message user-message">

            <div className="avatar user-avatar">
              You
            </div>

            <div className="message-content">

              <div className="message-name">
                You
              </div>

              <div className="user-text">
                {lastQuestion}
              </div>

            </div>

          </div>
        )}


        {loading && (
          <div className="message ai-message">

            <div className="avatar ai-avatar">
              ✦
            </div>

            <div className="message-content">

              <div className="message-name">
                AI Research Agent
              </div>

              <div className="typing">
                <span></span>
                <span></span>
                <span></span>
              </div>

            </div>

          </div>
        )}


        {answer && !loading && (
          <div className="message ai-message">

            <div className="avatar ai-avatar">
              ✦
            </div>

            <div className="message-content">

              <div className="message-name">
                AI Research Agent
              </div>

              <div className="ai-text">
                {answer}
              </div>

            </div>

          </div>
        )}

      </main>


      <div className="input-wrapper">

        <div className="input-box">

          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Message AI Research Agent..."
            rows="1"
          />

          <button
            className="send-button"
            onClick={askAgent}
            disabled={!question.trim() || loading}
          >
            ↑
          </button>

        </div>

        <p className="input-hint">
          AI can make mistakes. Check important information.
        </p>

      </div>

    </div>
  );
}

export default App;
