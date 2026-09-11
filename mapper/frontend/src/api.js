async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers ?? {})
    },
    ...options
  });

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      detail = payload.detail ?? detail;
    } catch {
      // Ignore JSON parse errors for plain text failures.
    }
    throw new Error(detail);
  }

  return response.json();
}

export function fetchJobs() {
  return request("/api/jobs");
}

export function fetchJob(jobId) {
  return request(`/api/jobs/${jobId}`);
}

export function fetchResults(jobId) {
  return request(`/api/results/${jobId}`);
}

export function analyzeJob(jobId) {
  return request(`/api/analyze/${jobId}`, { method: "POST" });
}

export function fetchAnalyzeStatus(jobId) {
  return request(`/api/jobs/${jobId}/analyze-status`);
}

export function fetchLayout(jobId) {
  return request(`/api/jobs/${jobId}/layout`);
}

export function updateLayout(jobId, payload) {
  return request(`/api/jobs/${jobId}/layout`, {
    method: "PUT",
    body: JSON.stringify(payload)
  });
}

export function createJob(jobId) {
  return request("/api/jobs", {
    method: "POST",
    body: JSON.stringify({ job_id: jobId })
  });
}

export async function uploadImages(jobId, files) {
  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file);
  }
  const response = await fetch(`/api/jobs/${jobId}/images`, {
    method: "POST",
    body: formData
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      detail = payload.detail ?? detail;
    } catch {
      // ignore
    }
    throw new Error(detail);
  }
  return response.json();
}
