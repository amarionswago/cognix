type ConfidenceBarProps = {
  score: number;
  label: "high" | "medium" | "low";
};

export function ConfidenceBar({ score, label }: ConfidenceBarProps) {
  return (
    <div className={`confidence-strip ${label}`}>
      <div>
        <strong>Evidence confidence</strong>
        <span>{label.toUpperCase()} · {Math.round(score * 100)}%</span>
      </div>
      <div className="confidence-track">
        <div />
      </div>
    </div>
  );
}

