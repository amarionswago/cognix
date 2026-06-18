const API_BASE = "/api";

export type SourceSnippet = {
  chunk_id: number;
  source_path: string;
  excerpt: string;
  score: number;
};

export type AskResponse = {
  answer: string;
  sources: SourceSnippet[];
  retrieval_summary: string;
  retrieval_diagnostics: {
    chunk_count?: number;
    unique_source_count?: number;
    unique_sources?: string[];
    max_score?: number;
    mean_score?: number;
    extension_filter?: string[];
    subquery_count?: number;
    keyword_group_count?: number;
    keyword_group_hits?: number;
    notes?: string[];
  };
  confidence: {
    score: number;
    label: "high" | "medium" | "low";
    breakdown: Record<string, unknown>;
  };
  output_id: number | null;
  output_path: string | null;
};

export type IntelligenceFinding = {
  id: number;
  finding_type: string;
  severity: string;
  title: string;
  description: string;
  source_refs_json: string;
  suggested_action: string;
  status: string;
  confidence: number;
  metadata_json: string;
  created_at: string;
  updated_at: string;
  resolved_at: string | null;
};

export type BriefingResponse = {
  id: number;
  brief_date: string;
  title: string;
  path: string;
  summary: string;
  finding_counts_json: string;
  status: string;
  content: string;
  created_at: string;
  updated_at: string;
};

export type MlCapability = {
  name: string;
  state: "ready" | "configured" | "fallback" | "missing";
  message: string;
  detail: Record<string, unknown>;
};

export type MlReadiness = {
  summary: Record<string, number>;
  capabilities: MlCapability[];
};

export async function getStatus() {
  return request("/status");
}

export async function runIngest() {
  return request("/ingest", {
    method: "POST",
    body: JSON.stringify({ source: "web-ui" })
  });
}

export async function askQuestion(question: string, style: string): Promise<AskResponse> {
  return request("/ask", {
    method: "POST",
    body: JSON.stringify({ question, style, save: true })
  });
}

export async function runIntelligence(useLlm = false) {
  return request("/intelligence/run", {
    method: "POST",
    body: JSON.stringify({ use_llm: useLlm })
  });
}

export async function listIntelligenceFindings(): Promise<IntelligenceFinding[]> {
  return request("/intelligence/findings");
}

export async function listGaps(): Promise<IntelligenceFinding[]> {
  return request("/gaps");
}

export async function listContradictions(): Promise<IntelligenceFinding[]> {
  return request("/contradictions");
}

export async function resolveContradiction(id: number): Promise<IntelligenceFinding> {
  return request(`/contradictions/${id}/resolve`, { method: "POST" });
}

export async function getLatestBriefing(): Promise<BriefingResponse> {
  return request("/briefings/latest");
}

export async function getMlReadiness(): Promise<MlReadiness> {
  return request("/ml/readiness");
}

export async function listOutputs() {
  return request("/outputs");
}

export async function updateOutput(id: number, status: string) {
  return request(`/outputs/${id}`, {
    method: "PATCH",
    body: JSON.stringify({ status })
  });
}

export async function getHealth() {
  return request("/health");
}

export async function runHealth() {
  return request("/health/run", { method: "POST" });
}

export async function getJobs() {
  return request("/jobs");
}

export async function getFiles() {
  return request("/files");
}

export async function getProfile() {
  return request("/settings/profile");
}

export async function saveProfile(profile: {
  username?: string;
  display_name: string;
  theme: "light" | "dark";
  default_answer_style: "brief" | "memo" | "deep";
  raw_data_note: string;
}) {
  return request("/settings/profile", {
    method: "PUT",
    body: JSON.stringify(profile)
  });
}

export async function listProviders() {
  return request("/providers");
}

export async function saveProvider(provider: string, settings: {
  enabled: boolean;
  api_key: string;
  model: string;
}) {
  return request(`/providers/${provider}`, {
    method: "PUT",
    body: JSON.stringify(settings)
  });
}

export async function testProvider(provider: string) {
  return request(`/providers/${provider}/test`, { method: "POST" });
}

export async function listBackgroundServices() {
  return request("/background");
}

export async function updateBackgroundService(name: string, settings: { enabled: boolean; interval_seconds?: number }) {
  return request(`/background/${name}`, {
    method: "PUT",
    body: JSON.stringify(settings)
  });
}

async function request(path: string, init: RequestInit = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init.headers || {})
    },
    ...init
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed: ${response.status}`);
  }
  return response.json();
}
