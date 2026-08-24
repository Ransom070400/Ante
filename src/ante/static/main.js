const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function fetchWithTimeout(url, options = {}, timeoutMs = 2000) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(id);
  }
}

async function fetchStatus(timeoutMs = 15000) {
  const loadingText = document.querySelector("#loading p");
  let attempts = 0;
  const deadline = Date.now() + timeoutMs;
  while (true) {
    attempts += 1;
    try {
      const url = new URL("/status", window.location.origin);
      url.searchParams.set("_", Date.now().toString());
      const resp = await fetchWithTimeout(url, {}, 2000);
      if (resp.ok) return await resp.json();
    } catch (e) {}
    if (loadingText) {
      loadingText.textContent = attempts > 8 ? "Starting backend…" : "Loading…";
    }
    if (Date.now() >= deadline) return null;
    await sleep(500);
  }
}

function show(el, flag) {
  el.classList.toggle("hidden", !flag);
}

// The 0G realtime server is reached over a websocket the app opens itself, so the only
// truthful thing this page can report is what the app's own status endpoint observes.
function render(status) {
  const title = document.getElementById("backend-title");
  const chip = document.getElementById("backend-chip");
  const detail = document.getElementById("backend-detail");
  const endpoint = document.getElementById("backend-endpoint");
  const help = document.getElementById("help-panel");

  if (!status) {
    title.textContent = "App not reachable";
    chip.textContent = "Offline";
    chip.className = "chip";
    detail.textContent = "The Ante app is not responding. Start it, then reload this page.";
    show(help, true);
    return;
  }

  const host = status.hf_direct_host;
  const port = status.hf_direct_port;
  endpoint.textContent = host ? `Realtime endpoint: ws://${host}:${port}/v1/realtime` : "";

  if (status.backend_connected) {
    title.textContent = "Connected to 0G";
    chip.textContent = "Connected";
    chip.className = "chip chip-ok";
    detail.textContent = "Ante is listening. Ask it anything.";
    show(help, false);
    return;
  }

  if (!status.has_hf_connection) {
    title.textContent = "No endpoint configured";
    chip.textContent = "Setup needed";
    chip.className = "chip";
    detail.textContent =
      "Set HF_REALTIME_CONNECTION_MODE=local and HF_REALTIME_WS_URL in .env, then restart Ante.";
    show(help, true);
    return;
  }

  title.textContent = "Waiting for the 0G server";
  chip.textContent = status.backend_connection_state || "Disconnected";
  chip.className = "chip";
  detail.textContent =
    status.backend_error || "The endpoint is configured but nothing is answering on it yet.";
  show(help, true);
}

async function init() {
  const loading = document.getElementById("loading");
  show(loading, true);
  render(await fetchStatus());
  show(loading, false);

  // The server is started by hand in a terminal, so poll rather than making the user reload.
  setInterval(async () => {
    const status = await fetchStatus(2000);
    if (status) render(status);
  }, 5000);
}

window.addEventListener("DOMContentLoaded", init);
