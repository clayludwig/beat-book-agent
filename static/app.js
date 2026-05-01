// ── Beat Book Builder — Frontend Logic ─────────────────────────────────
(() => {
  "use strict";

  // ── DOM refs ─────────────────────────────────────────────────────────
  const dropZone        = document.getElementById("drop-zone");
  const fileInput       = document.getElementById("file-input");
  const fileListEl      = document.getElementById("file-list");
  const uploadBtn       = document.getElementById("upload-btn");
  const uploadStatus    = document.getElementById("upload-status");
  const progressStep    = document.getElementById("progress-step");
  const progressBar     = document.getElementById("progress-bar");
  const progressDetail  = document.getElementById("progress-detail");

  const interviewFormHost = document.getElementById("interview-form-host");

  const generatingLabel   = document.getElementById("generating-label");
  const generatingDetail  = document.getElementById("generating-detail");
  const generatingStats   = document.getElementById("generating-stats");
  const generatingElapsed = document.getElementById("generating-elapsed");
  const stepperEl         = document.getElementById("stepper");
  const shimmerBar        = document.querySelector(".shimmer-bar");
  const shimmerFill       = document.querySelector(".shimmer-bar-fill");

  const doneSubtitle    = document.getElementById("done-subtitle");
  const doneViewerLink  = document.getElementById("done-viewer-link");
  const doneMarkdownLink = document.getElementById("done-markdown-link");

  const sessionInfoEls = document.querySelectorAll(
    "#interview-session-info, #generating-session-info, #done-session-info"
  );

  // ── State ────────────────────────────────────────────────────────────
  let selectedFiles = [];
  let ws = null;
  let activeInterview = null;
  const stats = { storiesRead: 0, searches: 0, topicsListed: 0 };

  let elapsedTimer = null;
  let elapsedStart = null;

  // ── Screen routing ───────────────────────────────────────────────────
  function switchScreen(name) {
    document.querySelectorAll(".screen").forEach(s => s.classList.remove("active"));
    const target = document.getElementById(`${name}-screen`);
    if (target) target.classList.add("active");
  }

  // ── File selection ───────────────────────────────────────────────────
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    addFiles([...e.dataTransfer.files].filter(f => f.name.endsWith(".json")));
  });
  dropZone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    addFiles([...fileInput.files]);
    fileInput.value = "";
  });

  function addFiles(files) {
    for (const f of files) {
      if (!selectedFiles.find(x => x.name === f.name)) selectedFiles.push(f);
    }
    renderFileList();
  }

  function renderFileList() {
    if (selectedFiles.length === 0) {
      fileListEl.hidden = true;
      uploadBtn.disabled = true;
      return;
    }
    fileListEl.hidden = false;
    uploadBtn.disabled = false;
    fileListEl.innerHTML = selectedFiles.map(f =>
      `<div class="file-item">
        <span class="name">${f.name}</span>
        <span>${(f.size / 1024).toFixed(1)} KB</span>
      </div>`
    ).join("");
  }

  // ── Upload flow ──────────────────────────────────────────────────────
  const STEP_LABELS = {
    embedding: "Generating embeddings",
    reducing:  "Reducing dimensions",
    clustering: "Clustering stories",
    labeling:  "Labeling topics",
  };
  const STEP_WEIGHTS = { embedding: 0.30, reducing: 0.10, clustering: 0.10, labeling: 0.50 };
  const STEP_ORDER   = ["embedding", "reducing", "clustering", "labeling"];

  function calcOverall(step, fraction) {
    let total = 0;
    for (const s of STEP_ORDER) {
      if (s === step) { total += STEP_WEIGHTS[s] * fraction; break; }
      total += STEP_WEIGHTS[s];
    }
    return Math.min(total, 1);
  }

  uploadBtn.addEventListener("click", async () => {
    if (!selectedFiles.length) return;
    uploadBtn.disabled = true;
    uploadStatus.hidden = false;
    progressStep.textContent = "Preparing…";
    progressBar.style.width = "0%";
    progressDetail.textContent = "";

    const form = new FormData();
    for (const f of selectedFiles) form.append("files", f);

    try {
      const resp = await fetch("/upload", { method: "POST", body: form });

      if (!resp.ok) {
        const err = await resp.json();
        progressStep.textContent = err.error || "Upload failed";
        progressBar.style.width = "0%";
        uploadBtn.disabled = false;
        return;
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const msg = JSON.parse(line.slice(6));

          if (msg.type === "progress") {
            const label = STEP_LABELS[msg.step] || msg.step;
            progressStep.textContent = label;
            progressDetail.textContent = msg.detail || "";
            progressBar.style.width = `${Math.round(calcOverall(msg.step, msg.fraction) * 100)}%`;
          }

          if (msg.type === "done") {
            progressStep.textContent = "Done.";
            progressBar.style.width = "100%";
            progressDetail.textContent = `${msg.num_stories} stories · ${msg.num_topics} topics`;
            setTimeout(() => startSession(msg), 600);
          }

          if (msg.type === "error") {
            progressStep.textContent = "Upload failed";
            progressDetail.textContent = msg.error || "Pipeline error";
            progressBar.style.width = "0%";
            uploadBtn.disabled = false;
          }
        }
      }
    } catch (err) {
      progressStep.textContent = `Upload failed: ${err.message}`;
      uploadBtn.disabled = false;
    }
  });

  // ── Elapsed time ticker ──────────────────────────────────────────────
  function startElapsed() {
    if (elapsedTimer) return;
    elapsedStart = Date.now();
    updateElapsed();
    elapsedTimer = setInterval(updateElapsed, 1000);
  }

  function stopElapsed() {
    if (elapsedTimer) {
      clearInterval(elapsedTimer);
      elapsedTimer = null;
    }
  }

  function updateElapsed() {
    if (!generatingElapsed || !elapsedStart) return;
    const secs = Math.floor((Date.now() - elapsedStart) / 1000);
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    generatingElapsed.textContent = `${m}:${s.toString().padStart(2, "0")}`;
  }

  // ── Stepper state ────────────────────────────────────────────────────
  const STAGE_ORDER = ["review", "write", "research", "cite"];

  function setStage(stage) {
    if (!stepperEl) return;
    const idx = STAGE_ORDER.indexOf(stage);
    stepperEl.querySelectorAll(".step").forEach(el => {
      const s = el.getAttribute("data-step");
      const sIdx = STAGE_ORDER.indexOf(s);
      el.classList.remove("active", "done");
      if (sIdx < idx) el.classList.add("done");
      else if (sIdx === idx) el.classList.add("active");
    });
  }

  function markAllStagesDone() {
    if (!stepperEl) return;
    stepperEl.querySelectorAll(".step").forEach(el => {
      el.classList.remove("active");
      el.classList.add("done");
    });
  }

  // ── Shimmer bar control ──────────────────────────────────────────────
  function setShimmerDeterminate(fraction) {
    if (!shimmerBar || !shimmerFill) return;
    shimmerBar.classList.add("determinate");
    shimmerFill.style.width = `${Math.min(Math.max(fraction, 0), 1) * 100}%`;
  }

  function setShimmerIndeterminate() {
    if (!shimmerBar || !shimmerFill) return;
    shimmerBar.classList.remove("determinate");
    shimmerFill.style.width = "";
  }

  // ── Start session: go to generating, open WebSocket ──────────────────
  function startSession(uploadData) {
    const sessionText = `${uploadData.num_stories} stories · ${uploadData.num_topics} topics`;
    sessionInfoEls.forEach(el => { el.textContent = sessionText; });

    setGenerating("Generating your beat book", "Reviewing your coverage…");
    setStage("review");
    setShimmerIndeterminate();
    startElapsed();
    switchScreen("generating");
    startWebSocket(uploadData.session_id);
  }

  // ── Generating screen helpers ────────────────────────────────────────
  function plural(n, single, multi) { return `${n} ${n === 1 ? single : multi}`; }

  function renderStatsChips() {
    if (!generatingStats) return;
    const parts = [];
    if (stats.storiesRead)  parts.push({ label: plural(stats.storiesRead, "story", "stories") + " read" });
    if (stats.searches)     parts.push({ label: plural(stats.searches, "search", "searches") + " run" });
    if (stats.topicsListed) parts.push({ label: plural(stats.topicsListed, "topic", "topics") + " explored" });

    generatingStats.innerHTML = parts
      .map(p => `<span class="chip">${p.label}</span>`)
      .join("");
  }

  function bumpStats(toolName) {
    if (toolName === "read_story") stats.storiesRead++;
    else if (toolName === "search_stories") stats.searches++;
    else if (toolName === "list_stories_in_topic") stats.topicsListed++;
    renderStatsChips();
  }

  function setGenerating(label, detail) {
    if (label) generatingLabel.textContent = label;
    generatingDetail.textContent = detail || "";
  }

  // ── WebSocket ────────────────────────────────────────────────────────
  function startWebSocket(sessionId) {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws/${sessionId}`);

    ws.onopen = () => {
      setGenerating("Generating your beat book", "Reviewing your coverage…");
      setStage("review");
    };

    ws.onmessage = (evt) => {
      const msg = JSON.parse(evt.data);

      switch (msg.type) {
        case "message":
          // Model narration is intentionally suppressed.
          break;

        case "tool_status":
          bumpStats(msg.tool_name);
          setGenerating("Generating your beat book", formatToolDetail(msg));
          break;

        case "questions":
          showInterview(msg);
          break;

        case "research_started":
          setGenerating("Researching context", "Opening the sandbox for the research agent…");
          setStage("research");
          setShimmerIndeterminate();
          break;

        case "research_tool_status":
          setGenerating("Researching context", formatToolDetail(msg));
          break;

        case "research_progress":
          setGenerating("Researching context", msg.detail || msg.stage || "");
          break;

        case "research_message":
          // Silent, mirrors the first agent's "message" handling.
          break;

        case "research_complete":
          setGenerating("Research complete", "Handing off to citation matcher…");
          break;

        case "beat_book_markdown_saved":
          setGenerating("Matching citations", "Embedding source sentences…");
          setStage("cite");
          setShimmerDeterminate(0.02);
          break;

        case "citation_progress": {
          const detail = msg.detail || msg.stage || "";
          setGenerating("Matching citations", detail);
          if (typeof msg.fraction === "number") {
            setShimmerDeterminate(msg.fraction);
          }
          break;
        }

        case "beat_book":
          showDone(msg);
          break;

        case "error":
          setGenerating("Something went wrong", msg.text || "Please try again.");
          setShimmerIndeterminate();
          break;
      }
    };

    ws.onclose = () => { /* no-op */ };
  }

  function formatToolDetail(msg) {
    if (msg.detail) return `${msg.tool} — ${msg.detail}`;
    return msg.tool || "";
  }

  // ── Interview rendering ──────────────────────────────────────────────
  function showInterview(msg) {
    activeInterview = {
      intro: msg.intro || "",
      questions: msg.questions || [],
    };

    interviewFormHost.innerHTML = "";

    const form = document.createElement("div");
    form.className = "interview-form";

    const collectors = [];

    activeInterview.questions.forEach((q, i) => {
      const block = document.createElement("div");
      block.className = "question-block";

      const num = document.createElement("div");
      num.className = "question-num";
      num.textContent = `Question ${i + 1} of ${activeInterview.questions.length}`;
      block.appendChild(num);

      const qText = document.createElement("div");
      qText.className = "question-text";
      qText.textContent = q.question;
      block.appendChild(qText);

      const type = q.question_type;
      const options = q.options || [];

      if (type === "free_response") {
        const ta = document.createElement("textarea");
        ta.className = "free-text";
        ta.placeholder = "Type your answer…";
        block.appendChild(ta);
        collectors.push(() => ta.value.trim());
      } else {
        const inputType = type === "single_choice" ? "radio" : "checkbox";
        const list = document.createElement("div");
        list.className = "option-list";

        options.forEach((opt, j) => {
          const item = document.createElement("label");
          item.className = "option-item";
          const id = `q${i}-opt${j}`;
          item.htmlFor = id;

          const input = document.createElement("input");
          input.type = inputType;
          input.name = `q${i}-option`;
          input.value = opt;
          input.id = id;

          const span = document.createElement("span");
          span.textContent = opt;

          item.appendChild(input);
          item.appendChild(span);

          input.addEventListener("change", () => {
            if (inputType === "radio") {
              list.querySelectorAll(".option-item").forEach(el => el.classList.remove("checked"));
            }
            item.classList.toggle("checked", input.checked);
          });

          list.appendChild(item);
        });

        block.appendChild(list);
        collectors.push(() =>
          [...list.querySelectorAll("input:checked")].map(el => el.value)
        );
      }

      form.appendChild(block);
    });

    const row = document.createElement("div");
    row.className = "submit-row";

    const hint = document.createElement("span");
    hint.className = "submit-hint";
    hint.textContent = `${activeInterview.questions.length} question${activeInterview.questions.length === 1 ? "" : "s"}`;
    row.appendChild(hint);

    const submitBtn = document.createElement("button");
    submitBtn.className = "btn primary";
    submitBtn.textContent = "Submit answers";
    submitBtn.addEventListener("click", () => {
      const answers = activeInterview.questions.map((q, i) => ({
        question: q.question,
        answer: collectors[i](),
      }));
      submitInterview(answers);
    });
    row.appendChild(submitBtn);

    form.appendChild(row);
    interviewFormHost.appendChild(form);

    switchScreen("interview");
    window.scrollTo({ top: 0 });
  }

  function submitInterview(answers) {
    activeInterview = null;
    interviewFormHost.innerHTML = "";

    setGenerating("Writing your beat book", "Processing your answers…");
    setStage("write");
    setShimmerIndeterminate();
    switchScreen("generating");

    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ answers }));
    }
  }

  // ── Done screen ──────────────────────────────────────────────────────
  function showDone(msg) {
    const viewerUrl = msg.viewer_url || `/static/viewer/viewer.html?book=${encodeURIComponent(msg.stem || "")}`;
    const markdownPath = msg.markdown_path || `/output/${encodeURIComponent(msg.filename)}`;

    doneViewerLink.href = viewerUrl;
    doneMarkdownLink.href = markdownPath;
    doneMarkdownLink.textContent = `Download raw Markdown (${msg.filename})`;

    markAllStagesDone();
    setShimmerDeterminate(1);
    stopElapsed();

    const parts = [];
    if (stats.storiesRead)  parts.push(plural(stats.storiesRead, "story", "stories") + " read");
    if (stats.searches)     parts.push(plural(stats.searches, "search", "searches") + " run");
    if (stats.topicsListed) parts.push(plural(stats.topicsListed, "topic", "topics") + " explored");
    doneSubtitle.textContent = parts.length
      ? `Built from ${parts.join(" · ")}.`
      : "";

    switchScreen("done");
  }

})();
