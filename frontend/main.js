const API_BASE = "";

function setStatus(el, message, kind) {
  el.textContent = message || "";
  el.className = "status" + (kind ? " " + kind : "");
}

function updateProgress(barEl, pct) {
  const v = Math.max(0, Math.min(100, pct || 0));
  const fill = barEl.querySelector(".progress-bar-fill");
  if (fill) {
    fill.style.width = v + "%";
  }
}

function appendHistoryRow(source, pred) {
  const tbody = document.querySelector("#history-table tbody");
  if (!tbody) return;
  const tr = document.createElement("tr");
  const cells = [
    source,
    pred.text,
    pred.label,
    (pred.confidence * 100).toFixed(1) + "%",
    pred.coarse_label,
    (pred.coarse_confidence * 100).toFixed(1) + "%",
    pred.needs_review ? "Yes" : "No",
  ];
  cells.forEach((val) => {
    const td = document.createElement("td");
    td.textContent = val;
    tr.appendChild(td);
  });
  tbody.prepend(tr);
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
  // Single prediction
  const singleText = document.getElementById("single-text");
  const singlePredict = document.getElementById("single-predict");
  const singleClear = document.getElementById("single-clear");
  const singleResult = document.getElementById("single-result");

  singlePredict.addEventListener("click", async () => {
    const text = (singleText.value || "").trim();
    singleResult.textContent = "";
    if (!text) {
      singleResult.textContent = "Please enter a transaction text.";
      return;
    }
    singlePredict.disabled = true;
    singleResult.textContent = "Running model…";
    try {
      const resp = await postJSON("/predict/text", { texts: [text], low_conf: 0.6 });
      const pred = resp[0];
      singleResult.innerHTML =
        `<strong>Label:</strong> ${pred.label} (${(pred.confidence * 100).toFixed(
          1
        )}% confidence)<br>` +
        `<strong>Coarse:</strong> ${pred.coarse_label} (${(
          pred.coarse_confidence * 100
        ).toFixed(1)}% routed probability)<br>` +
        `<strong>Needs review?</strong> ${pred.needs_review ? "Yes" : "No"}`;
      appendHistoryRow("single", pred);
    } catch (err) {
      singleResult.textContent = "Prediction failed: " + err.message;
    } finally {
      singlePredict.disabled = false;
    }
  });

  singleClear.addEventListener("click", () => {
    singleText.value = "";
    singleResult.textContent = "";
  });

  // Batch scoring
  const csvFile = document.getElementById("csv-file");
  const csvTextColumnSelect = document.getElementById("csv-text-column");
  const lowConfInput = document.getElementById("low-conf");
  const csvStart = document.getElementById("csv-start");
  const csvCancel = document.getElementById("csv-cancel");
  const csvStatus = document.getElementById("csv-status");
  const csvStatusMeta = document.getElementById("csv-status-meta");
  const csvProgressBar = document.getElementById("csv-progress-bar");
  const csvLog = document.getElementById("csv-log");
  const csvDownload = document.getElementById("csv-download");

  let currentJobId = null;
  let pollTimer = null;

  async function pollJob(jobId) {
    try {
      const res = await fetch(API_BASE + "/jobs/" + jobId);
      if (!res.ok) {
        setStatus(csvStatus, "Failed to fetch job status.", "error");
        updateProgress(csvProgressBar, 0);
        csvCancel.disabled = true;
        return;
      }
      const data = await res.json();
      const { state, progress, current, total, stage, logs } = data;
      const pctText = total ? `${current}/${total} rows` : `${progress}%`;
      setStatus(
        csvStatus,
        `State: ${state} · Stage: ${stage} · Progress: ${progress}% (${pctText})`,
        state === "FAILURE" ? "error" : state === "SUCCESS" ? "success" : "info"
      );
      updateProgress(csvProgressBar, progress);
      csvStatusMeta.textContent =
        state === "SUCCESS"
          ? "Job completed successfully. You can download the scored CSV."
          : state === "FAILURE"
          ? "The job failed. Check the log for details."
          : "Job is running in the background. You can safely navigate away and return later with the same browser.";

      csvLog.textContent = (logs || []).join("\n");
      csvLog.scrollTop = csvLog.scrollHeight;

      if (state === "SUCCESS") {
        clearInterval(pollTimer);
        pollTimer = null;
        csvCancel.disabled = true;
        // Show download link
        const link = document.createElement("a");
        link.href = API_BASE + "/jobs/" + jobId + "/result";
        link.textContent = "Download scored CSV";
        link.download = "txcat_scored.csv";
        csvDownload.innerHTML = "";
        csvDownload.appendChild(link);
      } else if (state === "FAILURE") {
        clearInterval(pollTimer);
        pollTimer = null;
        csvCancel.disabled = true;
      }
    } catch (err) {
      setStatus(csvStatus, "Error while polling job: " + err.message, "error");
      updateProgress(csvProgressBar, 0);
      csvCancel.disabled = true;
      csvStatusMeta.textContent = "Polling stopped due to an error.";
      if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
      }
    }
  }

  csvFile.addEventListener("change", () => {
    const file = csvFile.files && csvFile.files[0];
    // Reset column choices
    csvTextColumnSelect.innerHTML = "";
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Choose a column…";
    csvTextColumnSelect.appendChild(placeholder);

    if (!file) {
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
    // Read only the first 8KB which is enough for the header row
    reader.readAsText(file.slice(0, 8192));
  });

  csvStart.addEventListener("click", async () => {
    csvDownload.innerHTML = "";
    csvLog.textContent = "";
    setStatus(csvStatus, "", null);
    csvStatusMeta.textContent = "";
    updateProgress(csvProgressBar, 0);

    const file = csvFile.files && csvFile.files[0];
    const textCol = (csvTextColumnSelect.value || "").trim();
    const lowConf = parseFloat(lowConfInput.value || "0.6");

    if (!file) {
      setStatus(csvStatus, "Please choose a CSV file.", "error");
      return;
    }
    if (!textCol) {
      setStatus(csvStatus, "Please enter the text column name.", "error");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("text_column", textCol);
    formData.append("low_conf", String(lowConf));

    try {
      setStatus(csvStatus, "Submitting job…", "info");
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
      setStatus(csvStatus, "Job accepted. Tracking progress…", "info");
      csvStatusMeta.textContent =
        "The CSV is being processed in the background. You can watch progress below.";
      csvCancel.disabled = false;

      if (pollTimer) clearInterval(pollTimer);
      pollTimer = setInterval(() => pollJob(currentJobId), 1000);
    } catch (err) {
      setStatus(csvStatus, "Failed to start job: " + err.message, "error");
      csvStatusMeta.textContent = "";
      updateProgress(csvProgressBar, 0);
      csvCancel.disabled = true;
    }
  });

  csvCancel.addEventListener("click", async () => {
    if (!currentJobId) return;
    try {
      await fetch(API_BASE + "/jobs/" + currentJobId + "/cancel", {
        method: "POST",
      });
      setStatus(csvStatus, "Cancel requested. The job will stop shortly.", "info");
      csvStatusMeta.textContent =
        "A cancel signal has been sent to the worker. Some rows may already be processed.";
    } catch (err) {
      setStatus(csvStatus, "Failed to send cancel request: " + err.message, "error");
    }
  });
}

window.addEventListener("DOMContentLoaded", main);
