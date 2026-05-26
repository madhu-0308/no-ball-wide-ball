"use strict";

// Backend URL is provided by web/config.js (or set window.BACKEND_URL
// before this file loads).  Strip any trailing slash for safety.
const BACKEND_URL = (window.BACKEND_URL || "").replace(/\/+$/, "");
const api = (path) => `${BACKEND_URL}${path.startsWith("/") ? path : "/" + path}`;

if (!BACKEND_URL) {
  // Show the configuration banner so the user knows what to do.
  document.addEventListener("DOMContentLoaded", () => {
    const b = document.getElementById("backend-banner");
    if (b) b.style.display = "block";
  });
}

const POINT_COLORS = ["#ffd166", "#ef476f", "#06d6a0", "#118ab2"];

const state = {
  jobId: null,
  videoSize: null,
  canvasScale: 1,
  points: [],
  frameImage: null,
};

const $ = (sel) => document.querySelector(sel);

// ───────────────────────────────────────────────────────────────── upload
$("#upload-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const file = $("#file-input").files[0];
  if (!file) return;
  if (!BACKEND_URL) {
    setStatus("upload-status",
      "Backend URL is not configured. Set window.BACKEND_URL in config.js.",
      "error");
    return;
  }

  setStatus("upload-status", "Uploading…");
  const fd = new FormData();
  fd.append("video", file);

  try {
    const resp = await fetch(api("/api/upload"), { method: "POST", body: fd });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ error: resp.statusText }));
      throw new Error(err.error || "upload failed");
    }
    const data = await resp.json();
    state.jobId = data.job_id;
    state.videoSize = [data.width, data.height];
    setStatus(
      "upload-status",
      `Uploaded ${file.name}  •  ${data.width}×${data.height}  •  ` +
        `${data.frames} frames  •  ${data.fps.toFixed(1)} fps`,
      "ok"
    );
    await loadFrameIntoCanvas(api(data.frame_url));
    revealStep("step-calibrate");
  } catch (e) {
    setStatus("upload-status", `Error: ${e.message}`, "error");
  }
});

const fileDrop = document.querySelector(".file-drop");
["dragenter", "dragover"].forEach((ev) =>
  fileDrop.addEventListener(ev, (e) => {
    e.preventDefault();
    fileDrop.style.borderColor = "var(--accent)";
  })
);
["dragleave", "drop"].forEach((ev) =>
  fileDrop.addEventListener(ev, (e) => {
    e.preventDefault();
    fileDrop.style.borderColor = "";
  })
);
fileDrop.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files[0];
  if (f) {
    $("#file-input").files = e.dataTransfer.files;
    $("#upload-form").dispatchEvent(new Event("submit"));
  }
});

// ─────────────────────────────────────────────────────────────── calibrate
async function loadFrameIntoCanvas(url) {
  const img = new Image();
  img.crossOrigin = "anonymous";
  await new Promise((res, rej) => {
    img.onload = res;
    img.onerror = rej;
    img.src = url;
  });
  state.frameImage = img;

  const canvas = $("#calib-canvas");
  const maxDisplayWidth = Math.min(960, window.innerWidth - 80);
  const [w, h] = state.videoSize;
  state.canvasScale = Math.min(1, maxDisplayWidth / w);
  canvas.width = w;
  canvas.height = h;
  canvas.style.width = `${w * state.canvasScale}px`;
  canvas.style.height = `${h * state.canvasScale}px`;
  state.points = [];
  drawCanvas();
  highlightNextInstruction();
}

function drawCanvas() {
  const canvas = $("#calib-canvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (state.frameImage)
    ctx.drawImage(state.frameImage, 0, 0, canvas.width, canvas.height);

  if (state.points.length >= 2) {
    ctx.strokeStyle = "rgba(255,138,61,0.95)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    state.points.forEach((p, i) => {
      i === 0 ? ctx.moveTo(p[0], p[1]) : ctx.lineTo(p[0], p[1]);
    });
    if (state.points.length === 4) ctx.closePath();
    ctx.stroke();
  }

  state.points.forEach(([x, y], i) => {
    ctx.fillStyle = POINT_COLORS[i];
    ctx.beginPath();
    ctx.arc(x, y, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = "#fff";
    ctx.font = "bold 14px sans-serif";
    ctx.fillText(String(i + 1), x + 10, y - 10);
  });
}

$("#calib-canvas").addEventListener("click", (e) => {
  if (state.points.length >= 4) return;
  const canvas = $("#calib-canvas");
  const rect = canvas.getBoundingClientRect();
  const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
  const y = ((e.clientY - rect.top) / rect.height) * canvas.height;
  state.points.push([Math.round(x), Math.round(y)]);
  drawCanvas();
  highlightNextInstruction();
  $("#calib-save").disabled = state.points.length !== 4;
});

function highlightNextInstruction() {
  for (let i = 1; i <= 4; i++) {
    const li = $(`#pt-${i}`);
    li.classList.remove("active", "done");
    if (i <= state.points.length) li.classList.add("done");
    else if (i === state.points.length + 1) li.classList.add("active");
  }
}

$("#calib-undo").addEventListener("click", () => {
  state.points.pop();
  drawCanvas();
  highlightNextInstruction();
  $("#calib-save").disabled = state.points.length !== 4;
});
$("#calib-reset").addEventListener("click", () => {
  state.points = [];
  drawCanvas();
  highlightNextInstruction();
  $("#calib-save").disabled = true;
});

$("#calib-save").addEventListener("click", async () => {
  if (state.points.length !== 4) return;
  setStatus("calib-status", "Saving calibration…");
  try {
    const resp = await fetch(api(`/api/job/${state.jobId}/calibrate`), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ points: state.points }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.error || "save failed");
    }
    setStatus("calib-status", "Calibration saved.", "ok");
    revealStep("step-detect");
  } catch (e) {
    setStatus("calib-status", `Error: ${e.message}`, "error");
  }
});

// ─────────────────────────────────────────────────────────────── detect
$("#run-detect").addEventListener("click", async () => {
  if (!state.jobId) return;
  $("#run-detect").disabled = true;
  $("#detect-progress").classList.remove("hidden");
  setStatus("detect-status", "");

  try {
    const resp = await fetch(api(`/api/job/${state.jobId}/detect`), {
      method: "POST",
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.error || "detection failed");
    }
    const result = await resp.json();
    renderResults(result);
    revealStep("step-results");
    setStatus("detect-status", "Done.", "ok");
  } catch (e) {
    setStatus("detect-status", `Error: ${e.message}`, "error");
  } finally {
    $("#run-detect").disabled = false;
    $("#detect-progress").classList.add("hidden");
  }
});

// ─────────────────────────────────────────────────────────────── results
function renderResults(r) {
  const v = $("#result-video");
  v.src = api(r.output_url);
  v.load();

  const wide = $("#verdict-wide");
  const nb = $("#verdict-noball");
  wide.classList.remove("fired", "wide");
  nb.classList.remove("fired", "no_ball");

  wide.querySelector(".verdict-value").textContent =
    r.wide ? "CALLED" : "not called";
  nb.querySelector(".verdict-value").textContent =
    r.no_ball ? "CALLED" : "not called";
  if (r.wide)    wide.classList.add("fired", "wide");
  if (r.no_ball) nb.classList.add("fired", "no_ball");

  const rows = [
    ["Frames processed",   r.frames],
    ["Ball detections",    r.ball_detections],
    ["Bowler detections",  r.bowler_detections],
    ["Video size",         `${r.video_size[0]} × ${r.video_size[1]}`],
    ["FPS",                r.fps.toFixed(1)],
  ];
  const tbody = $("#stats-table tbody");
  tbody.innerHTML = rows
    .map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`)
    .join("");
}

$("#reset-all").addEventListener("click", () => {
  state.jobId = null;
  state.points = [];
  ["step-calibrate", "step-detect", "step-results"].forEach((id) =>
    $("#" + id).classList.add("hidden")
  );
  $("#upload-form").reset();
  setStatus("upload-status", "");
  setStatus("calib-status", "");
  setStatus("detect-status", "");
});

// ─────────────────────────────────────────────────────────────── helpers
function revealStep(id) {
  $("#" + id).classList.remove("hidden");
  setTimeout(() => $("#" + id).scrollIntoView({ behavior: "smooth" }), 50);
}

function setStatus(id, msg, kind = "") {
  const el = $("#" + id);
  el.textContent = msg;
  el.className = "status" + (kind ? " " + kind : "");
}
