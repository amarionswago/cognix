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
  output_id: number | null;
  output_path: string | null;
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
