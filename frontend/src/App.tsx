import {
  Activity,
  Archive,
  Brain,
  CheckCircle2,
  Database,
  FileSearch,
  FolderInput,
  Gauge,
  MessageSquareText,
  Settings,
  Sparkles,
  Trash2,
  UserRound
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  askQuestion,
  getFiles,
  getHealth,
  getJobs,
  getProfile,
  getStatus,
  listProviders,
  listOutputs,
  runHealth,
  runIngest,
  saveProvider,
  saveProfile,
  testProvider,
  listBackgroundServices,
  updateBackgroundService,
  updateOutput
} from "./api/client";

type View = "ask" | "ingest" | "outputs" | "health" | "settings";

const navItems: { id: View; label: string; icon: LucideIcon }[] = [
  { id: "ask", label: "Ask", icon: MessageSquareText },
  { id: "ingest", label: "Ingest", icon: FolderInput },
  { id: "outputs", label: "Outputs", icon: FileSearch },
  { id: "health", label: "Health", icon: Gauge },
  { id: "settings", label: "Settings", icon: Settings }
];

export function App() {
  const [view, setView] = useState<View>("ask");
  const [status, setStatus] = useState<string>("connecting");
  const [health, setHealth] = useState<any>(null);
  const [profile, setProfile] = useState<any>(() => ({
    username: "cognix-user",
    display_name: "Cognix User",
    theme: localStorage.getItem("cognix-theme") || "light",
    default_answer_style: "memo",
    raw_data_note: ""
  }));

  async function refreshHealth() {
    setHealth(await getHealth());
  }

  useEffect(() => {
    getStatus()
      .then((data) => setStatus(data.status))
      .catch(() => setStatus("offline"));
    getProfile()
      .then((data) => setProfile(data))
      .catch(() => undefined);
    refreshHealth().catch(() => undefined);
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = profile.theme || "light";
    localStorage.setItem("cognix-theme", profile.theme || "light");
  }, [profile.theme]);

  const active = useMemo(() => navItems.find((item) => item.id === view), [view]);

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <Brain size={24} />
          </div>
          <div>
            <strong>Cognix</strong>
            <span>Personal Neural Library</span>
          </div>
        </div>
        <nav className="nav-list">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <button className={view === item.id ? "active" : ""} key={item.id} onClick={() => setView(item.id)}>
                <Icon size={18} />
                {item.label}
              </button>
            );
          })}
        </nav>
        <div className="system-card">
          <Activity size={18} />
          <div>
            <strong>{status === "ok" ? "System ready" : "System unavailable"}</strong>
            <span>{health ? `Library health ${health.score}` : "Checking services"}</span>
          </div>
        </div>
      </aside>
      <section className="workspace">
        <header className="topbar">
          <div>
            <span className="eyebrow">Local-first knowledge system</span>
            <h1>{active?.label}</h1>
          </div>
          <div className="topbar-pill">
            <Sparkles size={16} />
            Draft-safe automation
          </div>
        </header>
        <div className={view === "ask" ? "view-panel active" : "view-panel"} aria-hidden={view !== "ask"}>
          <AskPage onSaved={refreshHealth} defaultStyle={profile.default_answer_style || "memo"} />
        </div>
        <div className={view === "ingest" ? "view-panel active" : "view-panel"} aria-hidden={view !== "ingest"}>
          <IngestPage onIngested={refreshHealth} />
        </div>
        <div className={view === "outputs" ? "view-panel active" : "view-panel"} aria-hidden={view !== "outputs"}>
          <OutputsPage />
        </div>
        <div className={view === "health" ? "view-panel active" : "view-panel"} aria-hidden={view !== "health"}>
          <HealthPage health={health} setHealth={setHealth} />
        </div>
        <div className={view === "settings" ? "view-panel active" : "view-panel"} aria-hidden={view !== "settings"}>
          <SettingsPage profile={profile} setProfile={setProfile} />
        </div>
      </section>
    </main>
  );
}

function AskPage({ onSaved, defaultStyle }: { onSaved: () => void; defaultStyle: string }) {
  const [question, setQuestion] = useState("");
  const [style, setStyle] = useState(defaultStyle);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState("");

  async function submit() {
    if (!question.trim()) return;
    setLoading(true);
    setError("");
    try {
      const response = await askQuestion(question, style);
      setResult(response);
      onSaved();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Question failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid two">
      <section className="panel ask-panel">
        <label>Question</label>
        <textarea value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask Cognix about your library..." />
        <div className="controls-row">
          <SegmentedControl
            value={style}
            onChange={setStyle}
            options={[
              { value: "memo", label: "Memo" },
              { value: "brief", label: "Brief" },
              { value: "deep", label: "Deep" }
            ]}
          />
          <button className="primary" onClick={submit} disabled={loading}>
            <MessageSquareText size={18} />
            {loading ? "Researching" : "Ask Cognix"}
          </button>
        </div>
        {error && <p className="error">{error}</p>}
      </section>
      <section className="panel result-panel">
        {!result && <EmptyState title="No answer yet" text="Ingest files first, then ask a question. Answers are saved as draft analysis pages." />}
        {result && (
          <>
            <div className="answer-text">{result.answer}</div>
            <h3>Sources</h3>
            <div className="source-list">
              {result.sources.map((source: any) => (
                <article key={source.chunk_id} className="source-item">
                  <strong>{source.source_path}</strong>
                  <span>Score {source.score}</span>
                  <p>{source.excerpt}</p>
                </article>
              ))}
            </div>
          </>
        )}
      </section>
    </div>
  );
}

function IngestPage({ onIngested }: { onIngested: () => void }) {
  const [files, setFiles] = useState<any[]>([]);
  const [jobs, setJobs] = useState<any>({ jobs: [], errors: [] });
  const [loading, setLoading] = useState(false);
  const activeIngest = Boolean((jobs.jobs || []).some((job: any) => job.kind === "ingest" && ["queued", "running", "processing", "started"].includes(job.status)));
  const ingestBusy = loading || activeIngest;
  const latestIngest = (jobs.jobs || []).find((job: any) => job.kind === "ingest");

  async function refresh() {
    setFiles(await getFiles());
    setJobs(await getJobs());
  }

  useEffect(() => {
    refresh().catch(() => undefined);
    const interval = window.setInterval(() => {
      refresh().catch(() => undefined);
    }, 3000);
    return () => window.clearInterval(interval);
  }, []);

  async function ingest() {
    setLoading(true);
    try {
      await runIngest();
      await refresh();
      onIngested();
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="stack">
      <section className="panel action-panel">
        <div>
          <h2>Raw File Ingest</h2>
          <p>Drop files into <code>data/raw/</code>, then start ingest. Originals remain untouched.</p>
        </div>
        <button className="primary" onClick={ingest} disabled={ingestBusy}>
          <FolderInput size={18} />
          {ingestBusy ? "Running ingest" : "Run ingest"}
        </button>
      </section>
      {ingestBusy && <p className="activity-note">Ingest is running. You can switch tabs; this page will keep updating.</p>}
      {!ingestBusy && latestIngest && <p className="activity-note">{latestIngest.message}</p>}
      <section className="grid two">
        <DataTable title="Recent Files" rows={files} columns={["relative_path", "status", "source_type"]} />
        <DataTable title="Recent Jobs" rows={jobs.jobs || []} columns={["kind", "status", "message", "completed", "failed"]} />
      </section>
      <DataTable title="Ingest Errors" rows={jobs.errors || []} columns={["path", "error_type", "message"]} />
    </div>
  );
}

function OutputsPage() {
  const [outputs, setOutputs] = useState<any[]>([]);

  async function refresh() {
    setOutputs(await listOutputs());
  }

  useEffect(() => {
    refresh().catch(() => undefined);
  }, []);

  async function setStatus(id: number, status: string) {
    await updateOutput(id, status);
    await refresh();
  }

  return (
    <section className="panel">
      <div className="panel-heading">
        <h2>Draft Outputs</h2>
        <button onClick={refresh}>Refresh</button>
      </div>
      <div className="output-list">
        {outputs.map((output) => (
          <article className="output-item" key={output.id}>
            <div>
              <strong>{output.title}</strong>
              <span>{output.status} | {output.type}</span>
              <p>{output.answer_preview}</p>
              <code>{output.path}</code>
            </div>
            <div className="icon-actions">
              <button title="Promote" onClick={() => setStatus(output.id, "promoted")}><CheckCircle2 size={17} /></button>
              <button title="Archive" onClick={() => setStatus(output.id, "archived")}><Archive size={17} /></button>
              <button title="Delete record" onClick={() => setStatus(output.id, "deleted")}><Trash2 size={17} /></button>
            </div>
          </article>
        ))}
        {!outputs.length && <EmptyState title="No outputs yet" text="Ask a question to generate a draft analysis page." />}
      </div>
    </section>
  );
}

function HealthPage({ health, setHealth }: { health: any; setHealth: (value: any) => void }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function run() {
    setLoading(true);
    setError("");
    try {
      setHealth(await runHealth());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Health check failed");
    } finally {
      setLoading(false);
    }
  }
  return (
    <div className="stack">
      <section className="panel action-panel">
        <div>
          <h2>Library Health</h2>
          <p>Checks structure, ingest state, errors, chunks, and generated health reports.</p>
        </div>
        <button className="primary" onClick={run} disabled={loading}>
          <Activity size={18} />
          {loading ? "Running health check" : "Run health check"}
        </button>
      </section>
      {error && <p className="error">{error}</p>}
      {health && (
        <>
          <div className="metric-grid">
            <Metric label="Score" value={health.score} icon={Gauge} />
            <Metric label="Files" value={health.totals.files} icon={Database} />
            <Metric label="Chunks" value={health.totals.chunks} icon={Brain} />
            <Metric label="Errors" value={health.totals.errors} icon={Activity} />
          </div>
          <DataTable title="Open Findings" rows={health.findings} columns={["severity", "category", "title", "message"]} />
        </>
      )}
    </div>
  );
}

function SettingsPage({ profile, setProfile }: { profile: any; setProfile: (value: any) => void }) {
  const [draft, setDraft] = useState(profile);
  const [saved, setSaved] = useState("");

  useEffect(() => {
    setDraft(profile);
  }, [profile]);

  async function submit() {
    const savedProfile = await saveProfile(draft);
    setProfile(savedProfile);
    setSaved("Saved locally");
    window.setTimeout(() => setSaved(""), 1600);
  }

  return (
    <div className="stack">
      <section className="panel">
        <div className="panel-heading">
          <h2>Local Profile</h2>
          <UserRound size={20} />
        </div>
        <div className="form-grid">
          <label>
            <span>Name</span>
            <input
              value={draft.display_name || ""}
              onChange={(event) => setDraft({ ...draft, display_name: event.target.value, username: event.target.value })}
              onBlur={submit}
              onKeyDown={(event) => {
                if (event.key === "Enter") submit();
              }}
            />
          </label>
          <label>
            <span>Theme</span>
            <SegmentedControl
              value={draft.theme || "light"}
              onChange={(theme) => {
                const next = { ...draft, theme };
                setDraft(next);
                setProfile({ ...profile, theme });
              }}
              options={[
                { value: "light", label: "White" },
                { value: "dark", label: "Dark" }
              ]}
            />
          </label>
          <label>
            <span>Default answer style</span>
            <SegmentedControl
              value={draft.default_answer_style || "memo"}
              onChange={(default_answer_style) => setDraft({ ...draft, default_answer_style })}
              options={[
                { value: "memo", label: "Memo" },
                { value: "brief", label: "Brief" },
                { value: "deep", label: "Deep" }
              ]}
            />
          </label>
          <label>
            <span>Raw data note</span>
            <textarea
              className="compact-textarea"
              value={draft.raw_data_note || ""}
              onChange={(event) => setDraft({ ...draft, raw_data_note: event.target.value })}
              placeholder="Optional note about where this user keeps imports, backups, or source exports."
            />
          </label>
        </div>
        <div className="controls-row">
          <button className="primary" onClick={submit}>Save profile</button>
          {saved && <span className="saved-note">{saved}</span>}
        </div>
      </section>
      <section className="panel">
        <ProviderSettings />
      </section>
      <section className="panel">
        <BackgroundSettings />
      </section>
    </div>
  );
}

function BackgroundSettings() {
  const [services, setServices] = useState<any[]>([]);
  const [message, setMessage] = useState("");

  async function refresh() {
    setServices(await listBackgroundServices());
  }

  useEffect(() => {
    refresh().catch(() => undefined);
  }, []);

  async function toggle(service: any, enabled: boolean) {
    const updated = await updateBackgroundService(service.name, {
      enabled,
      interval_seconds: service.interval_seconds
    });
    setServices((current) => current.map((item) => (item.name === service.name ? updated : item)));
    setMessage(`${service.name} ${enabled ? "enabled" : "disabled"}`);
  }

  async function updateInterval(service: any, interval_seconds: number) {
    const updated = await updateBackgroundService(service.name, {
      enabled: Boolean(service.enabled),
      interval_seconds
    });
    setServices((current) => current.map((item) => (item.name === service.name ? updated : item)));
    setMessage(`${service.name} interval saved`);
  }

  return (
    <>
      <div className="panel-heading">
        <h2>Background Work</h2>
        {message && <span className="saved-note">{message}</span>}
      </div>
      <div className="provider-list">
        {services.map((service) => (
          <article className="provider-card" key={service.name}>
            <div className="provider-card-head">
              <div>
                <strong>{service.name === "watcher" ? "File watcher" : "Scheduled ingest"}</strong>
                <span>{service.last_message || "Idle"}</span>
              </div>
              <span className={`connection-pill ${service.enabled ? "connected" : ""}`}>{service.enabled ? "Enabled" : "Off"}</span>
            </div>
            <div className="form-grid">
              <label className="inline-toggle">
                <input type="checkbox" checked={Boolean(service.enabled)} onChange={(event) => toggle(service, event.target.checked)} />
                <span>Run automatically</span>
              </label>
              <label>
                <span>Interval seconds</span>
                <input
                  type="number"
                  min="5"
                  value={service.interval_seconds}
                  onChange={(event) => updateInterval(service, Number(event.target.value))}
                />
              </label>
              {service.last_run_at && <span className="muted-line">Last run: {service.last_run_at}</span>}
            </div>
          </article>
        ))}
      </div>
    </>
  );
}

function ProviderSettings() {
  const [providers, setProviders] = useState<any[]>([]);
  const [drafts, setDrafts] = useState<Record<string, { enabled: boolean; api_key: string; model: string }>>({});
  const [message, setMessage] = useState("");

  async function refresh() {
    const data = await listProviders();
    setProviders(data);
    const nextDrafts: Record<string, { enabled: boolean; api_key: string; model: string }> = {};
    for (const provider of data) {
      nextDrafts[provider.provider] = {
        enabled: Boolean(provider.enabled),
        api_key: "",
        model: provider.model
      };
    }
    setDrafts(nextDrafts);
  }

  useEffect(() => {
    refresh().catch(() => undefined);
  }, []);

  async function persistProviderDraft(provider: string) {
    const draft = drafts[provider];
    if (!draft) return null;
    const data = await saveProvider(provider, draft);
    setProviders((current) => current.map((item) => (item.provider === provider ? data : item)));
    setDrafts((current) => ({ ...current, [provider]: { enabled: Boolean(data.enabled), api_key: "", model: data.model } }));
    return data;
  }

  async function saveOne(provider: string) {
    const data = await persistProviderDraft(provider);
    if (!data) return;
    setMessage(`${data.label} settings saved locally`);
  }

  async function testOne(provider: string) {
    setMessage("Saving settings, then testing connection...");
    await persistProviderDraft(provider);
    const data = await testProvider(provider);
    setProviders((current) => current.map((item) => (item.provider === provider ? data : item)));
    setDrafts((current) => ({ ...current, [provider]: { enabled: Boolean(data.enabled), api_key: "", model: data.model } }));
    setMessage(data.connected ? `${data.label} connected` : data.message || "Connection failed");
  }

  function updateDraft(provider: string, values: Partial<{ enabled: boolean; api_key: string; model: string }>) {
    setDrafts((current) => ({ ...current, [provider]: { ...current[provider], ...values } }));
  }

  return (
    <>
      <div className="panel-heading">
        <h2>Model Providers</h2>
        {message && <span className="saved-note">{message}</span>}
      </div>
      <div className="provider-list">
        {providers.map((provider) => {
          const draft = drafts[provider.provider] || { enabled: false, api_key: "", model: provider.model };
          const connected = provider.last_status === "connected" || provider.connected;
          const status = provider.status || provider.last_status;
          const failed = ["failed", "missing_key", "auth_failed", "model_unavailable", "quota_exhausted", "rate_limited"].includes(status);
          const configured = provider.configured;
          const needsKey = provider.env_var;
          const statusClass = connected ? "connected" : status === "quota_exhausted" ? "quota" : status === "rate_limited" ? "limited" : failed ? "failed" : configured ? "configured" : "";
          const statusLabel =
            connected
              ? "Connected"
              : status === "quota_exhausted"
                ? "Exhausted"
                : status === "rate_limited"
                  ? "Rate limited"
                  : status === "auth_failed"
                    ? "Bad key"
                    : status === "model_unavailable"
                      ? "Model missing"
                      : failed
                        ? "Failed"
                        : configured
                          ? "Configured"
                          : "Not connected";
          return (
            <article className="provider-card" key={provider.provider}>
              <div className="provider-card-head">
                <div>
                  <strong>{provider.label}</strong>
                  <span>{provider.provider}</span>
                </div>
                <span className={`connection-pill ${statusClass}`}>{statusLabel}</span>
              </div>
              <div className="form-grid">
                <label>
                  <span>Model</span>
                  <input
                    list={`${provider.provider}-models`}
                    value={draft.model}
                    onChange={(event) => updateDraft(provider.provider, { model: event.target.value })}
                    onBlur={() => saveOne(provider.provider)}
                    placeholder={provider.provider === "ollama" ? "Type an installed Ollama model" : "Model name"}
                  />
                  {provider.installed_models?.length > 0 && (
                    <datalist id={`${provider.provider}-models`}>
                      {provider.installed_models.map((model: string) => (
                        <option value={model} key={model} />
                      ))}
                    </datalist>
                  )}
                </label>
                {needsKey && (
                  <label>
                    <span>API key</span>
                    <input
                      type="password"
                      value={draft.api_key}
                      onChange={(event) => updateDraft(provider.provider, { api_key: event.target.value })}
                      placeholder={configured ? "Leave blank to keep existing key" : `Paste ${provider.label} API key`}
                    />
                  </label>
                )}
                <label className="inline-toggle">
                  <input
                    type="checkbox"
                    checked={draft.enabled}
                    onChange={(event) => updateDraft(provider.provider, { enabled: event.target.checked })}
                  />
                  <span>Use this provider for answers</span>
                </label>
                {provider.key_source && <span className="muted-line">Key source: {provider.key_source} {provider.masked_key ? `(${provider.masked_key})` : ""}</span>}
                {provider.provider === "ollama" && (
                  <span className="muted-line">
                    Installed models: {provider.installed_models?.length ? provider.installed_models.join(", ") : "none reported by Ollama"}
                  </span>
                )}
                {provider.last_message && <span className="muted-line">Last check: {provider.last_message}</span>}
                <div className="controls-row">
                  <button className="primary" onClick={() => saveOne(provider.provider)}>Save</button>
                  <button onClick={() => testOne(provider.provider)}>Test</button>
                </div>
              </div>
            </article>
          );
        })}
        {!providers.length && <EmptyState title="Providers loading" text="Cognix is reading local provider settings." />}
      </div>
      <div className="provider-note">
        Keys are local to this machine. Environment variables are preferred over keys stored in SQLite.
      </div>
    </>
  );
}

function SegmentedControl({
  value,
  onChange,
  options
}: {
  value: string;
  onChange: (value: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <div className="segmented-control">
      {options.map((option) => (
        <button
          className={value === option.value ? "selected" : ""}
          key={option.value}
          type="button"
          onClick={() => onChange(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

function DataTable({ title, rows, columns }: { title: string; rows: any[]; columns: string[] }) {
  return (
    <section className="panel">
      <div className="panel-heading">
        <h2>{title}</h2>
        <span>{rows.length}</span>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              {columns.map((column) => <th key={column}>{column}</th>)}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIndex) => (
              <tr key={row.id || rowIndex}>
                {columns.map((column) => <td key={column}>{String(row[column] ?? "")}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {!rows.length && <EmptyState title="Nothing here yet" text="This section will populate as Cognix works." />}
    </section>
  );
}

function Metric({ label, value, icon: Icon }: { label: string; value: number; icon: LucideIcon }) {
  return (
    <section className="metric">
      <Icon size={20} />
      <span>{label}</span>
      <strong>{value}</strong>
    </section>
  );
}

function EmptyState({ title, text }: { title: string; text: string }) {
  return (
    <div className="empty">
      <strong>{title}</strong>
      <span>{text}</span>
    </div>
  );
}
