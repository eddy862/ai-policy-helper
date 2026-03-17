import Chat from '@/components/Chat';
import AdminPanel from '@/components/AdminPanel';

export default function Page() {
  return (
    <div className="app-shell">
      <aside className="app-sidebar">
        <div className="brand-block">
          <div className="brand-dot" />
          <div>
            <h1 className="brand-title">AI Policy Helper</h1>
            <p className="brand-subtitle">Local-first RAG assistant</p>
          </div>
        </div>

        <AdminPanel />

        <div className="tips-card">
          <h3>Try these prompts</h3>
          <ul>
            <li>Can a customer return a damaged blender after 20 days?</li>
            <li>What is the shipping SLA to East Malaysia for bulky items?</li>
          </ul>
        </div>
      </aside>

      <main className="app-main">
        <header className="chat-header">
          <h2>Chat</h2>
          <p>Ask a question and inspect citations and source chunks.</p>
        </header>
        <Chat />
      </main>
    </div>
  );
}
