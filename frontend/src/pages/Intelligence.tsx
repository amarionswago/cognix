import { Activity, AlertTriangle, Brain, CheckCircle2, FileText, GitBranch, Lightbulb, Search } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  askQuestion,
  getLatestBriefing,
  listContradictions,
  listGaps,
  resolveContradiction,
  runIntelligence,
  type BriefingResponse,
  type IntelligenceFinding
} from "../api/client";
import { ConceptGraph } from "../components/ConceptGraph";
import { ContradictionCard } from "../components/ContradictionCard";
import { GapSuggestion } from "../components/GapSuggestion";
import { MorningBrief } from "./MorningBrief";

export function IntelligencePage() {
  const [gaps, setGaps] = useState<IntelligenceFinding[]>([]);
  const [contradictions, setContradictions] = useState<IntelligenceFinding[]>([]);
  const [briefing, setBriefing] = useState<BriefingResponse | null>(null);
  const [running, setRunning] = useState(false);
  const [askingConcept, setAskingConcept] = useState(false);
  const [error, setError] = useState("");
  const [suggestedQuestion, setSuggestedQuestion] = useState("");
  const [conceptAnswer, setConceptAnswer] = useState("");

  async function refresh() {
    const [nextGaps, nextContradictions] = await Promise.all([listGaps(), listContradictions()]);
    setGaps(nextGaps);
    setContradictions(nextContradictions);
    try {
      setBriefing(await getLatestBriefing());
    } catch {
      setBriefing(null);
    }
  }

  async function run() {
    setRunning(true);
    setError("");
    try {
      await runIntelligence(false);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Intelligence run failed");
    } finally {
      setRunning(false);
    }
  }

  async function resolve(id: number) {
    await resolveContradiction(id);
    await refresh();
  }

  async function askConcept(concept: string) {
    const question = `What does my library say about ${concept}?`;
    setSuggestedQuestion(question);
    setAskingConcept(true);
    setError("");
    try {
      const response = await askQuestion(question, "brief");
      setConceptAnswer(response.answer);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Concept question failed");
    } finally {
      setAskingConcept(false);
    }
  }

  useEffect(() => {
    refresh().catch(() => undefined);
  }, []);

  const graphConcepts = useMemo(
    () => gaps.map((finding) => finding.title.replace("Knowledge gap: ", "")),
    [gaps]
  );
  const reviewCount = gaps.length + contradictions.length;
  const topGap = gaps[0]?.title.replace("Knowledge gap: ", "") || "No urgent gap";
  const topConflict = contradictions[0]?.title || "No open conflict";

  return (
    <div className="stack intelligence-layout">
      <section className="panel intelligence-hero">
        <div className="hero-copy">
          <span className="eyebrow">Proactive library audit</span>
          <h2>Find what needs your attention.</h2>
          <p>Cognix reviews indexed sources for missing concepts, conflicting claims, stale knowledge, and briefable changes.</p>
          <div className="audit-flow" aria-label="Intelligence workflow">
            <span><Search size={14} /> Scan sources</span>
            <span><GitBranch size={14} /> Compare claims</span>
            <span><FileText size={14} /> Write brief</span>
          </div>
        </div>
        <div className="hero-action">
          <button className="primary" onClick={run} disabled={running}>
            <Activity size={18} />
            {running ? "Running audit" : "Run audit"}
          </button>
          <span>{running ? "Checking indexed chunks now" : "Uses already ingested data"}</span>
        </div>
      </section>
      {error && <p className="error">{error}</p>}
      <div className="metric-grid intelligence-metrics">
        <MetricCard icon={Lightbulb} label="Gaps to compile" value={gaps.length} detail={topGap} />
        <MetricCard icon={AlertTriangle} label="Conflicts to review" value={contradictions.length} detail={topConflict} tone={contradictions.length ? "warning" : "success"} />
        <MetricCard icon={FileText} label="Latest brief" value={briefing ? briefing.brief_date : "None"} detail={briefing ? briefing.summary : "Run an audit to generate one"} />
        <MetricCard icon={CheckCircle2} label="Open review queue" value={reviewCount} detail={reviewCount ? "Review findings below" : "Nothing needs review"} tone={reviewCount ? "info" : "success"} />
      </div>

      <section className="grid intelligence-grid">
        <div className="panel intelligence-section">
          <div className="section-kicker">
            <Lightbulb size={16} />
            <span>Missing concepts</span>
          </div>
          <div className="panel-heading">
            <div>
              <h2>Knowledge gaps</h2>
              <p>Topics Cognix sees often but has not compiled into durable wiki concepts.</p>
            </div>
            <span className="count-pill">{gaps.length}</span>
          </div>
          <div className="finding-list">
            {gaps.slice(0, 12).map((finding) => (
              <GapSuggestion key={finding.id} finding={finding} />
            ))}
            {!gaps.length && <EmptyAuditState title="No gaps open" text="Run an audit after ingesting more sources to surface missing concepts." />}
          </div>
        </div>
        <div className="panel intelligence-section">
          <div className="section-kicker">
            <Brain size={16} />
            <span>Concept map</span>
          </div>
          <div className="panel-heading">
            <div>
              <h2>Concept graph pulse</h2>
              <p>Click a surfaced concept to ask Cognix what your library already says about it.</p>
            </div>
          </div>
          <ConceptGraph concepts={graphConcepts} onAskConcept={askConcept} />
          {suggestedQuestion && (
            <div className="suggested-question">
              <strong>Ask next</strong>
              <code>{suggestedQuestion}</code>
              {askingConcept && <span>Researching concept…</span>}
              {conceptAnswer && <pre>{conceptAnswer}</pre>}
            </div>
          )}
        </div>
      </section>

      <section className="panel intelligence-section">
        <div className="section-kicker">
          <AlertTriangle size={16} />
          <span>Claim conflicts</span>
        </div>
        <div className="panel-heading">
          <div>
            <h2>Contradictions</h2>
            <p>Claim pairs that may disagree. Candidates are warnings until confirmed by a stronger judge.</p>
          </div>
          <span className="count-pill">{contradictions.length}</span>
        </div>
        <div className="finding-list">
          {contradictions.map((finding) => (
            <ContradictionCard key={finding.id} finding={finding} onResolve={resolve} />
          ))}
          {!contradictions.length && <EmptyAuditState title="No conflicts open" text="Cognix has not found unresolved contradiction candidates in the indexed library." />}
        </div>
      </section>
      <MorningBrief briefing={briefing} />
    </div>
  );
}

function MetricCard({
  icon: Icon,
  label,
  value,
  detail,
  tone = "neutral"
}: {
  icon: LucideIcon;
  label: string;
  value: string | number;
  detail: string;
  tone?: "neutral" | "warning" | "success" | "info";
}) {
  return (
    <article className={`metric intelligence-metric ${tone}`}>
      <div>
        <Icon size={17} />
        <span>{label}</span>
      </div>
      <strong>{value}</strong>
      <small>{detail}</small>
    </article>
  );
}

function EmptyAuditState({ title, text }: { title: string; text: string }) {
  return (
    <div className="audit-empty">
      <CheckCircle2 size={18} />
      <strong>{title}</strong>
      <span>{text}</span>
    </div>
  );
}
