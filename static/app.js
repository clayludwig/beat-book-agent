// ── Beat Book Builder — Frontend Logic ─────────────────────────────────
(() => {
  "use strict";

  // DOM refs
  const uploadScreen    = document.getElementById("upload-screen");
  const chatScreen      = document.getElementById("chat-screen");
  const dropZone        = document.getElementById("drop-zone");
  const fileInput       = document.getElementById("file-input");
  const fileListEl      = document.getElementById("file-list");
  const uploadBtn       = document.getElementById("upload-btn");
  const uploadStatus    = document.getElementById("upload-status");
  const chatMessages    = document.getElementById("chat-messages");
  const inputArea       = document.getElementById("input-area");
  const inputContainer  = document.getElementById("input-container");
  const sessionInfoEl   = document.getElementById("session-info");

  let selectedFiles = [];
  let ws = null;

  // ── File selection ────────────────────────────────────────────────────

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

  // ── Upload ────────────────────────────────────────────────────────────

  const progressStep   = document.getElementById("progress-step");
  const progressBar    = document.getElementById("progress-bar");
  const progressDetail = document.getElementById("progress-detail");

  const STEP_LABELS = {
    embedding: "Generating embeddings",
    reducing:  "Reducing dimensions",
    clustering: "Clustering stories",
    labeling:  "Labeling topics",
  };

  // Weights for overall progress (must sum to 1)
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

      // Read the SSE stream
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // Parse SSE lines
        const lines = buffer.split("\n");
        buffer = lines.pop(); // keep incomplete line in buffer

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const msg = JSON.parse(line.slice(6));

          if (msg.type === "progress") {
            const label = STEP_LABELS[msg.step] || msg.step;
            progressStep.textContent = label;
            progressDetail.textContent = msg.detail || "";
            const overall = calcOverall(msg.step, msg.fraction);
            progressBar.style.width = `${Math.round(overall * 100)}%`;
          }

          if (msg.type === "done") {
            progressStep.textContent = "Done!";
            progressBar.style.width = "100%";
            progressDetail.textContent = `${msg.num_stories} stories, ${msg.num_topics} topics`;
            setTimeout(() => switchToChat(msg), 600);
          }
        }
      }
    } catch (err) {
      progressStep.textContent = `Upload failed: ${err.message}`;
      uploadBtn.disabled = false;
    }
  });

  // ── Switch to chat ────────────────────────────────────────────────────

  function switchToChat(uploadData) {
    uploadScreen.classList.remove("active");
    chatScreen.classList.add("active");

    sessionInfoEl.textContent =
      `${uploadData.num_stories} stories · ${uploadData.num_topics} topics`;

    addSystemMsg(`Uploaded ${uploadData.num_stories} stories — ${uploadData.num_topics} topics discovered. Connecting to agent…`);

    startWebSocket(uploadData.session_id);
  }

  // ── WebSocket ─────────────────────────────────────────────────────────

  function startWebSocket(sessionId) {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws/${sessionId}`);

    let thinkingEl = null;

    ws.onopen = () => {
      removeThinking();
      showThinking();
    };

    ws.onmessage = (evt) => {
      const msg = JSON.parse(evt.data);

      switch (msg.type) {
        case "message":
          removeThinking();
          addAgentMsg(msg.text);
          showThinking();
          break;

        case "tool_status":
          updateThinking(msg.tool, msg.detail);
          break;

        case "question":
          removeThinking();
          showQuestion(msg);
          break;

        case "beat_book":
          removeThinking();
          showBeatBook(msg);
          break;

        case "error":
          removeThinking();
          addErrorMsg(msg.text);
          break;
      }
    };

    ws.onclose = () => {
      removeThinking();
    };

    function showThinking(label) {
      if (thinkingEl) return;
      thinkingEl = document.createElement("div");
      thinkingEl.className = "thinking";
      thinkingEl.innerHTML = `<span class="thinking-text">${label || "Agent is thinking"}</span> <span class="dots"><span></span><span></span><span></span></span>`;
      chatMessages.appendChild(thinkingEl);
      scrollToBottom();
    }

    function updateThinking(tool, detail) {
      const label = detail ? `${tool}: ${detail}` : tool;
      if (thinkingEl) {
        const textEl = thinkingEl.querySelector(".thinking-text");
        if (textEl) textEl.textContent = label;
        scrollToBottom();
      } else {
        showThinking(label);
      }
    }

    function removeThinking() {
      if (thinkingEl) {
        thinkingEl.remove();
        thinkingEl = null;
      }
    }
  }

  // ── Chat helpers ──────────────────────────────────────────────────────

  function addAgentMsg(text) {
    const el = document.createElement("div");
    el.className = "msg agent";
    el.textContent = text;
    chatMessages.appendChild(el);
    scrollToBottom();
  }

  function addUserMsg(text) {
    const el = document.createElement("div");
    el.className = "msg user";
    el.textContent = text;
    chatMessages.appendChild(el);
    scrollToBottom();
  }

  function addSystemMsg(text) {
    const el = document.createElement("div");
    el.className = "msg system";
    el.textContent = text;
    chatMessages.appendChild(el);
    scrollToBottom();
  }

  function addErrorMsg(text) {
    const el = document.createElement("div");
    el.className = "msg error";
    el.textContent = text;
    chatMessages.appendChild(el);
    scrollToBottom();
  }

  function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  // ── Show interactive question ─────────────────────────────────────────

  function showQuestion(msg) {
    inputArea.hidden = false;
    inputContainer.innerHTML = "";

    // Question text
    const qEl = document.createElement("div");
    qEl.className = "question-text";
    qEl.textContent = msg.question;
    inputContainer.appendChild(qEl);

    const type = msg.question_type;
    const options = msg.options || [];

    if (type === "free_response") {
      buildFreeResponse();
    } else if (type === "single_choice") {
      buildChoices(options, "radio");
    } else {
      // checklist or multiple_choice → checkboxes
      buildChoices(options, "checkbox");
    }

    scrollToBottom();
  }

  function buildFreeResponse() {
    const ta = document.createElement("textarea");
    ta.className = "free-text";
    ta.placeholder = "Type your answer…";
    inputContainer.appendChild(ta);

    const row = document.createElement("div");
    row.className = "submit-row";
    const btn = document.createElement("button");
    btn.className = "btn submit";
    btn.textContent = "Submit";
    btn.addEventListener("click", () => {
      const answer = ta.value.trim();
      if (!answer) return;
      sendAnswer(answer);
    });
    row.appendChild(btn);
    inputContainer.appendChild(row);

    ta.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        btn.click();
      }
    });

    ta.focus();
  }

  function buildChoices(options, inputType) {
    const list = document.createElement("div");
    list.className = "option-list";

    options.forEach((opt, i) => {
      const item = document.createElement("div");
      item.className = "option-item";

      const input = document.createElement("input");
      input.type = inputType;
      input.name = "q-option";
      input.value = opt;
      input.id = `opt-${i}`;

      const label = document.createElement("label");
      label.htmlFor = `opt-${i}`;
      label.textContent = opt;

      item.appendChild(input);
      item.appendChild(label);

      item.addEventListener("click", (e) => {
        if (e.target !== input) input.click();
      });

      list.appendChild(item);
    });

    inputContainer.appendChild(list);

    const row = document.createElement("div");
    row.className = "submit-row";
    const btn = document.createElement("button");
    btn.className = "btn submit";
    btn.textContent = "Submit";
    btn.addEventListener("click", () => {
      const checked = [...list.querySelectorAll("input:checked")].map(i => i.value);
      if (!checked.length) return;
      sendAnswer(checked.join(", "));
    });
    row.appendChild(btn);
    inputContainer.appendChild(row);
  }

  function sendAnswer(answer) {
    addUserMsg(answer);
    inputArea.hidden = true;
    inputContainer.innerHTML = "";

    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ answer }));
    }
  }

  // ── Beat book display ─────────────────────────────────────────────────

  function showBeatBook(msg) {
    const el = document.createElement("div");
    el.className = "msg beat-book";
    el.innerHTML = `
      <h3>📖 Beat Book Generated</h3>
      <p>Saved as <strong>${msg.filename}</strong></p>
      <p><a href="/output/${msg.filename}" target="_blank">Open beat book →</a></p>
    `;
    chatMessages.appendChild(el);
    scrollToBottom();
  }

})();
