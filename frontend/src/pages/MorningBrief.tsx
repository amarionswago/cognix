import type { BriefingResponse } from "../api/client";

type MorningBriefProps = {
  briefing: BriefingResponse | null;
};

export function MorningBrief({ briefing }: MorningBriefProps) {
  if (!briefing) {
    return (
      <section className="panel">
        <h2>Intelligence Brief</h2>
        <p>No brief generated yet.</p>
      </section>
    );
  }

  return (
    <section className="panel brief-panel">
      <div className="panel-heading">
        <div>
          <h2>{briefing.title}</h2>
          <span>{briefing.summary}</span>
        </div>
      </div>
      <pre>{briefing.content}</pre>
    </section>
  );
}

