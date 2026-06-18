import { Minus, MoveDown, MoveLeft, MoveRight, MoveUp, Plus, RotateCcw } from "lucide-react";
import { KeyboardEvent, useMemo, useState } from "react";

type ConceptGraphProps = {
  concepts: string[];
  onAskConcept: (concept: string) => void;
};

type GraphNode = {
  concept: string;
  x: number;
  y: number;
};

export function ConceptGraph({ concepts, onAskConcept }: ConceptGraphProps) {
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const nodes = useMemo(() => layoutNodes(concepts.slice(0, 14)), [concepts]);
  const center = { x: 240, y: 150 };

  return (
    <div className="concept-graph" aria-label="Concept graph preview">
      <div className="graph-toolbar">
        <button title="Zoom in" onClick={() => setZoom((value) => Math.min(1.8, value + 0.15))}><Plus size={15} /></button>
        <button title="Zoom out" onClick={() => setZoom((value) => Math.max(0.7, value - 0.15))}><Minus size={15} /></button>
        <button title="Pan left" onClick={() => setPan((value) => ({ ...value, x: value.x - 18 }))}><MoveLeft size={15} /></button>
        <button title="Pan right" onClick={() => setPan((value) => ({ ...value, x: value.x + 18 }))}><MoveRight size={15} /></button>
        <button title="Pan up" onClick={() => setPan((value) => ({ ...value, y: value.y - 18 }))}><MoveUp size={15} /></button>
        <button title="Pan down" onClick={() => setPan((value) => ({ ...value, y: value.y + 18 }))}><MoveDown size={15} /></button>
        <button title="Reset graph" onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }}><RotateCcw size={15} /></button>
      </div>
      <svg viewBox="0 0 480 300" role="group" aria-label="Knowledge concept graph">
        <g transform={`translate(${pan.x} ${pan.y}) scale(${zoom})`}>
          {nodes.map((node) => (
            <line key={`${node.concept}-edge`} x1={center.x} y1={center.y} x2={node.x} y2={node.y} />
          ))}
          <circle className="graph-core" cx={center.x} cy={center.y} r="28" />
          <text x={center.x} y={center.y + 4} textAnchor="middle">Cognix</text>
          {nodes.map((node) => (
            <g
              key={node.concept}
              className="graph-node"
              onClick={() => onAskConcept(node.concept)}
              onKeyDown={(event) => activateNode(event, () => onAskConcept(node.concept))}
              tabIndex={0}
              role="button"
              aria-label={`Ask Cognix about ${node.concept}`}
            >
              <circle cx={node.x} cy={node.y} r="24" />
              <text x={node.x} y={node.y + 4} textAnchor="middle">{shortLabel(node.concept)}</text>
            </g>
          ))}
        </g>
      </svg>
    </div>
  );
}

function activateNode(event: KeyboardEvent<SVGGElement>, action: () => void) {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    action();
  }
}

function layoutNodes(concepts: string[]): GraphNode[] {
  if (!concepts.length) {
    return [];
  }
  const centerX = 240;
  const centerY = 150;
  const radiusX = 160;
  const radiusY = 96;
  return concepts.map((concept, index) => {
    const angle = (Math.PI * 2 * index) / concepts.length - Math.PI / 2;
    return {
      concept,
      x: centerX + Math.cos(angle) * radiusX,
      y: centerY + Math.sin(angle) * radiusY,
    };
  });
}

function shortLabel(concept: string): string {
  return concept.length > 11 ? `${concept.slice(0, 10)}…` : concept;
}
