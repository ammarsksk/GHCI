const API_BASE = "";

function setStatus(el, message, kind) {
  if (!el) return;
  el.textContent = message || "";
  el.className = "status" + (kind ? " " + kind : "");
}

function updateProgress(barEl, pct) {
  if (!barEl) return;
  const v = Math.max(0, Math.min(100, pct || 0));
  const fill = barEl.querySelector(".progress-bar-fill");
  if (fill) {
    fill.style.width = v + "%";
  }
}

function appendHistoryRow(source, pred, feedbackText = "Not reviewed") {
  const tbody = document.querySelector("#history-table tbody");
  if (!tbody) return null;
  const tr = document.createElement("tr");
  const cells = [
    source,
    pred.text,
    pred.label,
    (pred.confidence * 100).toFixed(1) + "%",
    pred.coarse_label,
    (pred.coarse_confidence * 100).toFixed(1) + "%",
    pred.needs_review ? "Yes" : "No",
    feedbackText,
  ];
  cells.forEach((val) => {
    const td = document.createElement("td");
    td.textContent = val;
    tr.appendChild(td);
  });
  tbody.prepend(tr);
  return tr;
}

async function postJSON(path, body) {
  const res = await fetch(API_BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json();
}

async function main() {
  // Single prediction elements
  const singleText = document.getElementById("single-text");
  const singlePredict = document.getElementById("single-predict");
  const singleClear = document.getElementById("single-clear");
  const singleResult = document.getElementById("single-result");
  const singleNoise = document.getElementById("single-noise");

  let lastHistoryRow = null;

  const LABEL_OPTIONS = [
    "DINING",
    "GROCERIES",
    "FUEL",
    "UTILITIES_POWER",
    "UTILITIES_TELECOM",
    "UTILITIES_WATER_GAS",
    "SHOPPING_ECOM",
    "SHOPPING_ELECTRONICS",
    "TRAVEL",
    "MOBILITY",
    "HEALTH",
    "ENTERTAINMENT",
    "FEES",
    "OTHER",
  ];

  if (singlePredict) {
    singlePredict.addEventListener("click", async () => {
      const text = (singleText.value || "").trim();
      singleResult.textContent = "";
      if (!text) {
        singleResult.textContent = "Please enter a transaction text.";
        return;
      }

      singlePredict.disabled = true;
      singleResult.textContent = "Running model...";

      try {
        const resp = await postJSON("/predict/text", { texts: [text], low_conf: 0.6 });
        const pred = resp[0];
        const confPct = (pred.confidence * 100).toFixed(1);
        const coarsePct = (pred.coarse_confidence * 100).toFixed(1);
        const needsReview = !!pred.needs_review;

        const badgeClass = needsReview ? "result-badge-review" : "result-badge-ok";
        const badgeText = needsReview ? "Needs review" : "No review needed";

        let explainListHtml = "";
        if (pred.explanation) {
          const parts = String(pred.explanation)
            .split(";")
            .map((p) => p.trim())
            .filter(Boolean);
          if (parts.length) {
            explainListHtml =
              '<div class="explain-block"><div>Top signals:</div><ul>' +
              parts
                .slice(0, 5)
                .map((p) => "<li>" + p + "</li>")
                .join("") +
              "</ul></div>";
          }
        }

        const feedbackSelectId = "fb-select";
        const feedbackYesId = "fb-yes";
        const feedbackNoId = "fb-no";

        singleResult.innerHTML = [
          '<div class="result-card">',
          '  <div class="result-inner">',
          '    <div class="result-left">',
          '      <div class="result-header-row">',
          '        <div class="label-chip">' + pred.label + "</div>",
          '        <div class="result-badge ' + badgeClass + '">' + badgeText + "</div>",
          "      </div>",
          '      <div class="result-row">Coarse: ' +
            pred.coarse_label +
            " | " +
            coarsePct +
            "% routed</div>",
          "      " + explainListHtml,
          '      <div class="feedback-block">',
          '        <div class="feedback-label">Feedback</div>',
          '        <div class="feedback-row">',
          '          <button id="' +
            feedbackYesId +
            '" class="tiny secondary active">Correct</button>',
          '          <button id="' + feedbackNoId + '" class="tiny secondary">Incorrect</button>',
          "        </div>",
          '        <div class="feedback-extra" id="fb-extra" style="display:none;">',
          '          <span class="feedback-extra-label">Set correct label:</span>',
          '          <select id="' + feedbackSelectId + '">',
          '            <option value="">Choose label...</option>',
          LABEL_OPTIONS.map((lab) => '            <option value="' + lab + '">' + lab + "</option>")
            .join("\n"),
          "          </select>",
          "        </div>",
          "      </div>",
          "    </div>",
          '    <div class="result-right">',
          '      <div class="confidence-circle" style="--pct: ' + confPct + ';">',
          '        <span class="confidence-value">' + confPct + "%</span>",
          "      </div>",
          '      <div class="confidence-caption">Confidence</div>',
          "    </div>",
          "  </div>",
          "</div>",
        ].join("\n");

        lastHistoryRow = appendHistoryRow("single", pred, "Accepted");

        const fbYes = document.getElementById(feedbackYesId);
        const fbNo = document.getElementById(feedbackNoId);
        const fbSel = document.getElementById(feedbackSelectId);
        const fbExtra = document.getElementById("fb-extra");

        if (fbYes && lastHistoryRow) {
          fbYes.addEventListener("click", () => {
            const cells = lastHistoryRow.querySelectorAll("td");
            const feedbackCell = cells[cells.length - 1];
            feedbackCell.textContent = "Accepted";
            fbYes.classList.add("active");
            if (fbNo) fbNo.classList.remove("active");
            if (fbExtra) fbExtra.style.display = "none";
          });
        }

        if (fbNo && fbSel && lastHistoryRow) {
          fbNo.addEventListener("click", () => {
            const val = fbSel.value || "OTHER";
            const cells = lastHistoryRow.querySelectorAll("td");
            const feedbackCell = cells[cells.length - 1];
            feedbackCell.textContent = "Corrected: " + val;
            fbNo.classList.add("active");
            if (fbYes) fbYes.classList.remove("active");
            if (fbExtra) fbExtra.style.display = "flex";
          });

          fbSel.addEventListener("change", () => {
            if (!lastHistoryRow) return;
            const val = fbSel.value || "OTHER";
            const cells = lastHistoryRow.querySelectorAll("td");
            const feedbackCell = cells[cells.length - 1];
            feedbackCell.textContent = "Corrected: " + val;
          });
        }
      } catch (err) {
        singleResult.textContent = "Prediction failed: " + err.message;
      } finally {
        singlePredict.disabled = false;
      }
    });
  }

  if (singleClear) {
    singleClear.addEventListener("click", () => {
      singleText.value = "";
      singleResult.textContent = "";
    });
  }

  function addNoiseToText(value) {
    if (!value) return value;
    let s = value;
    s = s
      .split("")
      .map((ch, idx) => {
        if (/[a-zA-Z]/.test(ch) && idx % 3 === 0) {
          return Math.random() > 0.5 ? ch.toUpperCase() : ch.toLowerCase();
        }
        return ch;
      })
      .join("");
    s = s.replace(/\s+/g, (m) => (m.length > 1 ? m : m + " "));
    return s;
  }

  if (singleNoise) {
    singleNoise.addEventListener("click", () => {
      const text = (singleText.value || "").trim();
      if (!text) {
        singleResult.textContent = "Enter a transaction first before adding noise.";
        return;
      }
      singleText.value = addNoiseToText(text);
    });
  }

  // Batch scoring elements
  const csvFile = document.getElementById("csv-file");
  const csvTextColumnSelect = document.getElementById("csv-text-column");
  const lowConfInput = document.getElementById("low-conf");
  const lowConfSlider = document.getElementById("low-conf-slider");
  const csvStart = document.getElementById("csv-start");
  const csvCancel = document.getElementById("csv-cancel");
  const csvStatus = document.getElementById("csv-status");
  const csvStatusMeta = document.getElementById("csv-status-meta");
  const csvStatusPill = document.getElementById("csv-status-pill");
  const csvFileName = document.getElementById("csv-file-name");
  const csvProgressText = document.getElementById("csv-progress-text");
  const csvProgressBar = document.getElementById("csv-progress-bar");
  const csvLog = document.getElementById("csv-log");
  const csvLogToggle = document.getElementById("csv-log-toggle");
  const csvLogBody = document.getElementById("csv-log-body");
  const csvDownload = document.getElementById("csv-download");
  const metricRows = document.getElementById("metric-rows");
  const metricDuration = document.getElementById("metric-duration");
  const metricThroughput = document.getElementById("metric-throughput");
  const metricFlagged = document.getElementById("metric-flagged");

  let currentJobId = null;
  let pollTimer = null;
  const jobStartTimes = {};

  function setStatusPill(mode) {
    if (!csvStatusPill) return;
    let text = "Idle";
    let cls = "status-pill status-idle";
    if (mode === "running") {
      text = "Running";
      cls = "status-pill status-running";
    } else if (mode === "completed") {
      text = "Completed";
      cls = "status-pill status-completed";
    } else if (mode === "cancelled") {
      text = "Aborted";
      cls = "status-pill status-cancelled";
    } else if (mode === "failed") {
      // For this UI, treat all terminal errors as "Aborted"
      text = "Aborted";
      cls = "status-pill status-failed";
    }
    csvStatusPill.textContent = text;
    csvStatusPill.className = cls;
  }

  function resetMetrics() {
    if (metricRows) metricRows.textContent = "0";
    if (metricDuration) metricDuration.textContent = "0.0 s";
    if (metricThroughput) metricThroughput.textContent = "0 rows/s";
    if (metricFlagged) metricFlagged.textContent = "0%";
  }

  resetMetrics();
  if (csvProgressText) csvProgressText.textContent = "Progress: 0%";
  if (csvFileName) csvFileName.textContent = "File: \u2014";
  if (csvLogBody) csvLogBody.style.display = "none";
  if (csvLog) csvLog.textContent = "Waiting for job...";

  if (csvLogToggle && csvLogBody) {
    csvLogToggle.addEventListener("click", () => {
      const isOpen = csvLogBody.style.display !== "none";
      csvLogBody.style.display = isOpen ? "none" : "block";
      csvLogToggle.classList.toggle("open", !isOpen);
    });
  }

  if (lowConfSlider && lowConfInput) {
    lowConfSlider.addEventListener("input", () => {
      const v = parseFloat(lowConfSlider.value || "0.6");
      lowConfInput.value = v.toFixed(2);
    });
    lowConfInput.addEventListener("input", () => {
      const v = parseFloat(lowConfInput.value || "0.6");
      if (isNaN(v)) return;
      const clamped = Math.min(0.9, Math.max(0.1, v));
      lowConfSlider.value = clamped.toFixed(2);
    });
  }

  async function computeMetricsForJob(jobId, totalRows) {
    const startedAt = jobStartTimes[jobId];
    let durationText = "n/a";
    let throughputText = "n/a";
    let latencyText = "n/a";
    let flaggedText = "n/a";

    if (startedAt && totalRows > 0) {
      const secs = (Date.now() - startedAt) / 1000;
      if (secs > 0.05) {
        durationText = secs.toFixed(1) + " s";
        const thr = totalRows / secs;
        throughputText = Math.round(thr).toLocaleString() + " rows/s";
        const latencyMs = (secs * 1000) / totalRows;
        latencyText = latencyMs.toFixed(1) + " ms/tx";
      }
    }

    try {
      const res = await fetch(API_BASE + "/jobs/" + jobId + "/result");
      if (res.ok) {
        const csv = await res.text();
        const lines = csv.split(/\r?\n/).filter((l) => l.trim().length > 0);
        if (lines.length > 1) {
          const header = lines[0].split(",");
          const idx = header.findIndex(
            (h) => h.trim().replace(/^"|"$/g, "") === "needs_review"
          );
          if (idx !== -1) {
            let flagged = 0;
            for (let i = 1; i < lines.length; i++) {
              const cols = lines[i].split(",");
              const val = (cols[idx] || "").trim().toLowerCase();
              if (val === "true" || val === "1" || val === "yes") {
                flagged += 1;
              }
            }
            const pct = (100 * flagged) / (lines.length - 1) || 0;
            flaggedText = pct.toFixed(1) + "%";
          }
        }
      }
    } catch (_) {
      // Best-effort; leave flaggedText as n/a
    }

    if (metricRows) {
      metricRows.textContent = totalRows > 0 ? totalRows.toLocaleString() : "0";
    }
    if (metricDuration) {
      metricDuration.textContent = durationText;
    }
    if (metricThroughput) {
      metricThroughput.textContent = throughputText;
    }
    if (metricFlagged) {
      metricFlagged.textContent = flaggedText;
    }
    if (csvStatusMeta) {
      csvStatusMeta.textContent =
        "Metrics are measured per CSV run on this machine, including file I/O and model scoring.";
    }
  }

  async function pollJob(jobId) {
    try {
      const res = await fetch(API_BASE + "/jobs/" + jobId);
      if (!res.ok) {
        setStatus(csvStatus, "Failed to fetch job status.", "error");
        updateProgress(csvProgressBar, 0);
        setStatusPill("failed");
        if (csvProgressText) csvProgressText.textContent = "Progress: 0%";
        if (csvCancel) csvCancel.disabled = true;
        return;
      }
      const data = await res.json();
      const state = data.state;
      const progress = data.progress || 0;
      const current = data.current;
      const total = data.total;
      const stage = data.stage;
      const logs = data.logs || [];
      const pctText = total ? current + "/" + total + " rows" : progress + "%";

      // Decide pill mode, preferring stage="cancelled" to show "Aborted"
      let pillMode = "idle";
      if (stage === "cancelled") {
        pillMode = "cancelled";
      } else if (state === "SUCCESS") {
        pillMode = "completed";
      } else if (state === "FAILURE") {
        pillMode = "failed";
      } else if (state === "PENDING" || state === "STARTED") {
        pillMode = "running";
      }
      setStatusPill(pillMode);

      setStatus(
        csvStatus,
        "State: " +
          state +
          " | Stage: " +
          stage +
          " | Progress: " +
          progress +
          "% (" +
          pctText +
          ")",
        state === "FAILURE" ? "error" : state === "SUCCESS" ? "success" : "info"
      );
      updateProgress(csvProgressBar, progress);
      if (csvProgressText) {
        csvProgressText.textContent = "Progress: " + progress + "%";
      }
      if (csvLog) {
        csvLog.textContent = logs.join("\n");
        csvLog.scrollTop = csvLog.scrollHeight;
      }

      if (state === "SUCCESS") {
        clearInterval(pollTimer);
        pollTimer = null;
        if (csvCancel) csvCancel.disabled = true;
        const totalRows = total || current || 0;
        computeMetricsForJob(jobId, totalRows);
        const link = document.createElement("a");
        link.href = API_BASE + "/jobs/" + jobId + "/result";
        link.textContent = "Download scored CSV";
        link.download = "txcat_scored.csv";
        if (csvDownload) {
          csvDownload.innerHTML = "";
          csvDownload.appendChild(link);
        }
      } else if (
        state === "FAILURE" ||
        state === "REVOKED" ||
        state === "CANCELLED"
      ) {
        clearInterval(pollTimer);
        pollTimer = null;
        if (csvCancel) csvCancel.disabled = true;
        if (csvStatusMeta) {
          csvStatusMeta.textContent =
            state === "REVOKED" || state === "CANCELLED"
              ? "The job was cancelled. Some rows may already be processed."
              : "The job failed. Check the log for details.";
        }
      }
    } catch (err) {
      setStatus(csvStatus, "Error while polling job: " + err.message, "error");
      updateProgress(csvProgressBar, 0);
      setStatusPill("failed");
      if (csvProgressText) csvProgressText.textContent = "Progress: 0%";
      if (csvCancel) csvCancel.disabled = true;
      if (csvStatusMeta) {
        csvStatusMeta.textContent = "Polling stopped due to an error.";
      }
      if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
      }
    }
  }

  if (csvFile) {
    csvFile.addEventListener("change", () => {
      const file = csvFile.files && csvFile.files[0];
      if (csvTextColumnSelect) {
        csvTextColumnSelect.innerHTML = "";
        const placeholder = document.createElement("option");
        placeholder.value = "";
        placeholder.textContent = "Choose a column...";
        csvTextColumnSelect.appendChild(placeholder);
      }

      if (!file || !csvTextColumnSelect) {
        return;
      }
      const reader = new FileReader();
      reader.onload = () => {
        const text = String(reader.result || "");
        const firstLine = text.split(/\r?\n/)[0] || "";
        if (!firstLine) {
          return;
        }
        const cols = firstLine.split(",").map((c) => c.trim().replace(/^"|"$/g, ""));
        cols.forEach((name) => {
          if (!name) return;
          const opt = document.createElement("option");
          opt.value = name;
          opt.textContent = name;
          csvTextColumnSelect.appendChild(opt);
        });
      };
      reader.readAsText(file.slice(0, 8192));
    });
  }

  if (csvStart) {
    csvStart.addEventListener("click", async () => {
      if (csvDownload) csvDownload.innerHTML = "";
      if (csvLog) csvLog.textContent = "Waiting for job...";
      setStatus(csvStatus, "", null);
      if (csvStatusMeta) {
        csvStatusMeta.textContent =
          "Metrics are measured per CSV run on this machine, including file I/O and model scoring.";
      }
      updateProgress(csvProgressBar, 0);
      resetMetrics();
      if (csvProgressText) csvProgressText.textContent = "Progress: 0%";

      const file = csvFile && csvFile.files && csvFile.files[0];
      const textCol = (csvTextColumnSelect && csvTextColumnSelect.value || "").trim();
      const lowConf = parseFloat((lowConfInput && lowConfInput.value) || "0.6");

      if (!file) {
        setStatus(csvStatus, "Please choose a CSV file.", "error");
        return;
      }
      if (!textCol) {
        setStatus(csvStatus, "Please choose a text column.", "error");
        return;
      }

      if (csvFileName) {
        csvFileName.textContent = "File: " + file.name;
      }

      const formData = new FormData();
      formData.append("file", file);
      formData.append("text_column", textCol);
      formData.append("low_conf", String(lowConf));

      try {
        setStatus(csvStatus, "Submitting job...", "info");
        const res = await fetch(API_BASE + "/jobs/csv", {
          method: "POST",
          body: formData,
        });
        if (!res.ok) {
          const text = await res.text();
          throw new Error(text || res.statusText);
        }
        const data = await res.json();
        currentJobId = data.job_id;
        jobStartTimes[currentJobId] = Date.now();
        setStatus(csvStatus, "Job accepted. Tracking progress...", "info");
        setStatusPill("running");
        if (csvStatusMeta) {
          csvStatusMeta.textContent =
            "The CSV is being processed in the background. You can watch progress below.";
        }
        if (csvCancel) csvCancel.disabled = false;

        if (pollTimer) clearInterval(pollTimer);
        pollTimer = setInterval(() => pollJob(currentJobId), 1000);
      } catch (err) {
        setStatus(csvStatus, "Failed to start job: " + err.message, "error");
        if (csvStatusMeta) csvStatusMeta.textContent = "";
        updateProgress(csvProgressBar, 0);
        if (csvCancel) csvCancel.disabled = true;
        setStatusPill("failed");
      }
    });
  }

  if (csvCancel) {
    csvCancel.addEventListener("click", async () => {
      if (!currentJobId) return;
      try {
        // Stop polling immediately so the UI does not "resume"
        if (pollTimer) {
          clearInterval(pollTimer);
          pollTimer = null;
        }
        await fetch(API_BASE + "/jobs/" + currentJobId + "/cancel", {
          method: "POST",
        });
        setStatus(csvStatus, "Job cancelled by user.", "info");
        if (csvStatusMeta) {
          csvStatusMeta.textContent =
            "A cancel signal has been sent to the worker. Some rows may already be processed.";
        }
        setStatusPill("cancelled");
        if (csvCancel) csvCancel.disabled = true;
        if (csvLog) {
          const prefix = csvLog.textContent ? "\n" : "";
          csvLog.textContent = csvLog.textContent + prefix + "Job aborted by user.";
          csvLog.scrollTop = csvLog.scrollHeight;
        }
      } catch (err) {
        setStatus(csvStatus, "Failed to send cancel request: " + err.message, "error");
      }
    });
  }
}

window.addEventListener("DOMContentLoaded", main);
