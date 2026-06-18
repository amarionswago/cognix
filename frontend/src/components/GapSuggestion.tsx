import type { IntelligenceFinding } from "../api/client";

type GapSuggestionProps = {
  finding: IntelligenceFinding;
};

export function GapSuggestion({ finding }: GapSuggestionProps) {
  const concept = finding.title.replace("Knowledge gap: ", "");
  return (
    <article className="finding-card gap-card">
      <div className="finding-card-header">
        <div>
          <span className="finding-label">Gap</span>
          <strong>{concept}</strong>
          <span>{readableSeverity(finding.severity)} · {Math.round(finding.confidence * 100)}% confidence</span>
        </div>
      </div>
      <p>{finding.description}</p>
      <small>{finding.suggested_action}</small>
    </article>
  );
}

function readableSeverity(value: string): string {
  if (value === "error") return "Needs review";
  if (value === "warning") return "Worth checking";
  return "Informational";
}
