import type { IntelligenceFinding } from "../api/client";

type ContradictionCardProps = {
  finding: IntelligenceFinding;
  onResolve: (id: number) => void;
};

export function ContradictionCard({ finding, onResolve }: ContradictionCardProps) {
  const parts = finding.description.split("\n\n");
  const left = cleanClaim(parts[0] || finding.description);
  const right = cleanClaim(parts[1] || "");
  const confirmed = finding.finding_type === "contradiction";
  return (
    <article className="finding-card contradiction-card">
      <div className="finding-card-header">
        <div>
          <span className={confirmed ? "finding-label danger" : "finding-label"}>{confirmed ? "Confirmed conflict" : "Candidate"}</span>
          <strong>{finding.title}</strong>
          <span>{readableSeverity(finding.severity)} · {Math.round(finding.confidence * 100)}% confidence</span>
        </div>
        <button onClick={() => onResolve(finding.id)}>Resolve</button>
      </div>
      <div className="claim-grid">
        <div>
          <span>Claim A</span>
          <p>{left}</p>
        </div>
        <div>
          <span>Claim B</span>
          <p>{right}</p>
        </div>
      </div>
      <small>{finding.suggested_action}</small>
    </article>
  );
}

function cleanClaim(value: string): string {
  return value.replace(/^Claim [AB]:\s*/i, "").trim();
}

function readableSeverity(value: string): string {
  if (value === "error") return "High priority";
  if (value === "warning") return "Review suggested";
  return "Informational";
}
