/**
 * CyberGuard AI Frontend Logic
 * Supports Dark/Light mode toggle, dynamic tabs, Chart.js visualizations,
 * unified API prediction calls, audit history filtering, and real-time alerts.
 */

// Global API Base URL (Relative for smooth self-hosted deployment, with fallback when served from local files or alternate ports)
const API_BASE = window.API_BASE || (window.location.protocol === "file:" || (window.location.port && window.location.port !== "8000") ? "http://localhost:8000" : "");

// Utility to escape HTML entities to prevent attribute injection and XSS
function escapeHtml(str) {
  if (str === null || str === undefined) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// Chart.js global instances
let chartPieInstance = null;
let chartChannelInstance = null;
let chartComparisonInstance = null;

// Global history cache for client-side search and filtering
let allHistoryRecords = [];

// ─────────────────────────────────────────────────────────────────────────────
// Initialization & Theme Handling
// ─────────────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  setupTheme();
  setupTabs();
  setupEventListeners();
  loadDashboardData();
  loadComparisonData();
});

function setupTheme() {
  const toggleBtn = document.getElementById("themeToggleBtn");
  const themeIcon = document.getElementById("themeIcon");
  const themeLabel = document.getElementById("themeLabel");
  const html = document.documentElement;

  // Retrieve saved theme or default to dark
  const savedTheme = localStorage.getItem("cyberguard_theme") || "dark";
  html.setAttribute("data-theme", savedTheme);
  updateThemeButton(savedTheme);

  toggleBtn.addEventListener("click", () => {
    const current = html.getAttribute("data-theme");
    const next = current === "dark" ? "light" : "dark";
    html.setAttribute("data-theme", next);
    localStorage.setItem("cyberguard_theme", next);
    updateThemeButton(next);

    // Re-render charts for theme visibility
    if (chartPieInstance) chartPieInstance.update();
    if (chartChannelInstance) chartChannelInstance.update();
    if (chartComparisonInstance) chartComparisonInstance.update();
  });

  function updateThemeButton(theme) {
    if (theme === "dark") {
      themeIcon.textContent = "☀️";
      themeLabel.textContent = "Light Mode";
    } else {
      themeIcon.textContent = "🌙";
      themeLabel.textContent = "Dark Mode";
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tab Navigation
// ─────────────────────────────────────────────────────────────────────────────
function setupTabs() {
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  tabButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      tabButtons.forEach(b => b.classList.remove("active"));
      tabPanes.forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const targetTabId = btn.getAttribute("data-tab");
      const targetPane = document.getElementById(targetTabId);
      if (targetPane) {
        targetPane.classList.add("active");
      }

      // If switching to Dashboard or Compare, refresh charts
      if (targetTabId === "tab-dashboard") {
        loadDashboardData();
      } else if (targetTabId === "tab-compare") {
        loadComparisonData();
      }
    });
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Event Listeners for Prediction Actions
// ─────────────────────────────────────────────────────────────────────────────
function setupEventListeners() {
  // 1. SMS Sample select
  const smsSample = document.getElementById("smsSelectSample");
  if (smsSample) {
    smsSample.addEventListener("change", () => {
      if (smsSample.value) {
        document.getElementById("smsInput").value = smsSample.value;
      }
    });
  }

  // 2. SMS Check Button
  document.getElementById("checkSmsBtn").addEventListener("click", handleSmsCheck);

  // 3. Email Check Button
  document.getElementById("checkEmailBtn").addEventListener("click", handleEmailCheck);

  // 4. Phishing Check Button
  document.getElementById("checkPhishBtn").addEventListener("click", handlePhishingCheck);

  // 5. URL Check Button
  document.getElementById("checkUrlBtn").addEventListener("click", handleUrlCheck);

  // 6. History Table Search & Filter
  document.getElementById("tableSearchInput").addEventListener("input", filterHistoryTable);
  document.getElementById("tableFilterType").addEventListener("change", filterHistoryTable);
  document.getElementById("refreshStatsBtn").addEventListener("click", loadDashboardData);
}

// ─────────────────────────────────────────────────────────────────────────────
// Prediction Handlers
// ─────────────────────────────────────────────────────────────────────────────

async function handleSmsCheck() {
  const text = document.getElementById("smsInput").value.trim();
  const modelSelect = document.getElementById("smsModelSelect");
  const model_name = modelSelect ? modelSelect.value : "svm_rbf";
  const spinner = document.getElementById("smsSpinner");
  const errorBox = document.getElementById("smsError");
  const resultCard = document.getElementById("smsResultCard");

  if (!text) {
    showError(errorBox, "Please provide an SMS message to analyze.");
    return;
  }

  hideError(errorBox);
  setLoading("checkSmsBtn", spinner, true);

  try {
    const res = await fetch(`${API_BASE}/predict/sms`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, model_name })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Prediction request failed.");
    }

    const data = await res.json();
    renderResultCard({
      resultCardId: "smsResultCard",
      badgeId: "smsVerdictBadge",
      modelBadgeId: "smsModelBadge",
      probTextId: "smsProbText",
      fillBarId: "smsProbFill",
      chipsListId: "smsChipsList",
      data: data
    });

  } catch (err) {
    showError(errorBox, err.message);
  } finally {
    setLoading("checkSmsBtn", spinner, false);
  }
}

async function handleEmailCheck() {
  const subject = document.getElementById("emailSubject").value.trim();
  const body = document.getElementById("emailBody").value.trim();
  const spinner = document.getElementById("emailSpinner");
  const errorBox = document.getElementById("emailError");

  if (!body) {
    showError(errorBox, "Please provide an email message body.");
    return;
  }

  hideError(errorBox);
  setLoading("checkEmailBtn", spinner, true);

  try {
    const res = await fetch(`${API_BASE}/predict/email`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ subject, body })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Email prediction failed.");
    }

    const data = await res.json();
    renderResultCard({
      resultCardId: "emailResultCard",
      badgeId: "emailVerdictBadge",
      modelBadgeId: "emailModelBadge",
      probTextId: "emailProbText",
      fillBarId: "emailProbFill",
      chipsListId: "emailChipsList",
      data: data
    });

  } catch (err) {
    showError(errorBox, err.message);
  } finally {
    setLoading("checkEmailBtn", spinner, false);
  }
}

async function handlePhishingCheck() {
  const subject = document.getElementById("phishSubject").value.trim();
  const body = document.getElementById("phishBody").value.trim();
  const spinner = document.getElementById("phishSpinner");
  const errorBox = document.getElementById("phishError");

  if (!body) {
    showError(errorBox, "Please provide the email content to scan for phishing.");
    return;
  }

  hideError(errorBox);
  setLoading("checkPhishBtn", spinner, true);

  try {
    const res = await fetch(`${API_BASE}/predict/phishing`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ subject, body })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Phishing scan failed.");
    }

    const data = await res.json();
    renderResultCard({
      resultCardId: "phishResultCard",
      badgeId: "phishVerdictBadge",
      modelBadgeId: "phishRiskBadge",
      probTextId: "phishProbText",
      fillBarId: "phishProbFill",
      chipsListId: "phishChipsList",
      data: data,
      isPhishingTab: true
    });

  } catch (err) {
    showError(errorBox, err.message);
  } finally {
    setLoading("checkPhishBtn", spinner, false);
  }
}

async function handleUrlCheck() {
  const url = document.getElementById("urlInput").value.trim();
  const spinner = document.getElementById("urlSpinner");
  const errorBox = document.getElementById("urlError");

  if (!url) {
    showError(errorBox, "Please provide a valid URL.");
    return;
  }

  hideError(errorBox);
  setLoading("checkUrlBtn", spinner, true);

  try {
    const res = await fetch(`${API_BASE}/predict/url`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "URL scan failed.");
    }

    const data = await res.json();
    renderResultCard({
      resultCardId: "urlResultCard",
      badgeId: "urlVerdictBadge",
      modelBadgeId: "urlModelBadge",
      probTextId: "urlProbText",
      fillBarId: "urlProbFill",
      chipsListId: "urlLexicalDetails",
      data: data,
      isUrlTab: true
    });

  } catch (err) {
    showError(errorBox, err.message);
  } finally {
    setLoading("checkUrlBtn", spinner, false);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Render Result Card Utility
// ─────────────────────────────────────────────────────────────────────────────

function renderResultCard({ resultCardId, badgeId, modelBadgeId, probTextId, fillBarId, chipsListId, data, isPhishingTab, isUrlTab }) {
  const card = document.getElementById(resultCardId);
  const badge = document.getElementById(badgeId);
  const modelBadge = document.getElementById(modelBadgeId);
  const probText = document.getElementById(probTextId);
  const fillBar = document.getElementById(fillBarId);
  const chipsList = document.getElementById(chipsListId);

  card.style.display = "block";
  const isSpam = data.label === "Spam";
  const probPct = Math.round(data.spam_probability * 100);

  // Update classes
  card.className = `result-card ${isSpam ? "spam" : "ham"}`;
  fillBar.className = `progress-bar-fill ${isSpam ? "spam" : "ham"}`;

  // Update verdict
  if (isPhishingTab && data.details && data.details.verdict) {
    badge.innerHTML = escapeHtml(data.details.verdict);
    modelBadge.textContent = `Risk Score: ${data.details.risk_score ?? "N/A"} / 100`;
  } else if (isUrlTab && data.details && data.details.verdict) {
    badge.innerHTML = escapeHtml(data.details.verdict);
    const lexRisk = data.details.lexical_features?.heuristic_risk_score ?? "N/A";
    modelBadge.textContent = `Lexical Risk: ${lexRisk} / 100`;
  } else {
    badge.innerHTML = isSpam ? "🚨 SPAM DETECTED" : "✅ LEGITIMATE (HAM)";
    modelBadge.textContent = `Model: ${escapeHtml(data.model_used || "Unknown")}`;
  }

  // Update gauge
  probText.textContent = `${probPct}% (Confidence: ${Math.round((data.confidence || 0) * 100)}%)`;
  fillBar.style.width = `${probPct}%`;

  // Render keywords / details
  chipsList.innerHTML = "";
  if (isUrlTab && data.details && data.details.lexical_features) {
    const lex = data.details.lexical_features;
    chipsList.innerHTML = `
      <div style="font-size: 0.85rem; line-height: 1.6; margin-top: 0.5rem; color: var(--text-primary);">
        <div>• <strong>Hostname:</strong> ${escapeHtml(lex.hostname || "N/A")}</div>
        <div>• <strong>Length:</strong> ${Number(lex.url_length || 0)} chars | <strong>Subdomain Dots:</strong> ${Number(lex.num_dots || 0)}</div>
        <div>• <strong>IP Link:</strong> ${lex.has_ip_address ? "🚨 Yes" : "No"} | <strong>@ Symbol:</strong> ${lex.has_at_symbol ? "🚨 Yes" : "No"}</div>
        <div>• <strong>HTTPS:</strong> ${lex.has_https ? "✅ Secure" : "⚠️ Missing"}</div>
      </div>
    `;
  } else if (data.top_keywords && data.top_keywords.length > 0) {
    data.top_keywords.forEach(kw => {
      const chip = document.createElement("span");
      const sentiment = String(kw.sentiment || "");
      const isNegative = (kw.score || 0) > 0 || sentiment.includes("Spam") || sentiment.includes("Trigger") || sentiment.includes("Credential");
      chip.className = `chip ${isNegative ? "spam-chip" : "ham-chip"}`;
      chip.textContent = `${kw.word} (${(kw.score || 0) > 0 ? "+" : ""}${kw.score})`;
      chipsList.appendChild(chip);
    });
  } else {
    chipsList.innerHTML = `<span style="font-size: 0.85rem; color: var(--text-secondary);">No prominent dictionary triggers.</span>`;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Analytics Dashboard Data & Charts
// ─────────────────────────────────────────────────────────────────────────────

async function loadDashboardData() {
  try {
    const [statsRes, historyRes] = await Promise.all([
      fetch(`${API_BASE}/stats`),
      fetch(`${API_BASE}/history?limit=100`)
    ]);

    if (!statsRes.ok || !historyRes.ok) return;

    const stats = await statsRes.json();
    const historyData = await historyRes.json();

    // 1. Update KPI Cards
    document.getElementById("kpiTotal").textContent = Number(stats.total_checked || 0).toLocaleString();
    document.getElementById("kpiSpam").textContent = Number(stats.spam_count || 0).toLocaleString();
    document.getElementById("kpiHam").textContent = Number(stats.ham_count || 0).toLocaleString();
    document.getElementById("kpiRate").textContent = `${stats.spam_percentage || 0}%`;

    // 2. Render Pie Chart (Spam vs Ham)
    renderPieChart(stats.spam_count || 0, stats.ham_count || 0);

    // 3. Render Channel Volume Bar Chart
    renderChannelChart(stats.inspections_by_channel || {});

    // 4. Populate History Table
    allHistoryRecords = historyData.history || [];
    renderHistoryTable(allHistoryRecords);

  } catch (err) {
    console.warn("Could not load dashboard data:", err);
  }
}

function renderPieChart(spam, ham) {
  const ctx = document.getElementById("chartPie").getContext("2d");
  if (chartPieInstance) chartPieInstance.destroy();

  const isDark = document.documentElement.getAttribute("data-theme") === "dark";
  const textColor = isDark ? "#f8fafc" : "#0f172a";

  chartPieInstance = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Spam / Threats", "Legitimate (Ham)"],
      datasets: [{
        data: [spam, ham],
        backgroundColor: ["#ef4444", "#10b981"],
        borderWidth: 0
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          position: "bottom",
          labels: { color: textColor }
        }
      }
    }
  });
}

function renderChannelChart(channels) {
  const ctx = document.getElementById("chartChannel").getContext("2d");
  if (chartChannelInstance) chartChannelInstance.destroy();

  const isDark = document.documentElement.getAttribute("data-theme") === "dark";
  const textColor = isDark ? "#f8fafc" : "#0f172a";
  const gridColor = isDark ? "#334155" : "#e2e8f0";

  const labels = Object.keys(channels);
  const data = Object.values(channels);

  chartChannelInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels.length ? labels : ["SMS", "Email", "Phishing", "URL"],
      datasets: [{
        label: "Inspections Count",
        data: data.length ? data : [0, 0, 0, 0],
        backgroundColor: "#3b82f6",
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      scales: {
        x: { ticks: { color: textColor }, grid: { display: false } },
        y: { ticks: { color: textColor }, grid: { color: gridColor }, beginAtZero: true }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });
}

function renderHistoryTable(records) {
  const tbody = document.getElementById("historyTableBody");
  tbody.innerHTML = "";

  if (!records || records.length === 0) {
    tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-secondary);">No threat inspection history yet.</td></tr>`;
    return;
  }

  records.forEach(r => {
    const isSpam = r.prediction === "Spam";
    const tr = document.createElement("tr");
    const msg = String(r.message_text || "");
    tr.innerHTML = `
      <td>#${r.id}</td>
      <td style="white-space: nowrap; font-size: 0.8rem; color: var(--text-secondary);">${escapeHtml(r.timestamp || "")}</td>
      <td><span style="font-weight: 600;">${escapeHtml(r.type || "")}</span></td>
      <td style="max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${escapeHtml(msg)}">${escapeHtml(msg)}</td>
      <td><span class="badge ${isSpam ? "badge-spam" : "badge-ham"}">${escapeHtml(r.prediction || "")}</span></td>
      <td>${Math.round((r.probability || 0) * 100)}%</td>
      <td style="font-size: 0.8rem; color: var(--text-secondary);">${escapeHtml(r.model_used || "")}</td>
    `;
    tbody.appendChild(tr);
  });
}

function filterHistoryTable() {
  const searchEl = document.getElementById("tableSearchInput");
  const filterEl = document.getElementById("tableFilterType");
  const query = (searchEl ? searchEl.value : "").toLowerCase().trim();
  const filterType = filterEl ? filterEl.value : "ALL";

  const filtered = allHistoryRecords.filter(r => {
    const msgText = String(r.message_text || "").toLowerCase();
    const userFb = String(r.user_feedback || "").toLowerCase();
    const matchesQuery = !query || msgText.includes(query) || userFb.includes(query);
    const matchesType = filterType === "ALL" || String(r.type || "").toUpperCase() === filterType.toUpperCase();
    return matchesQuery && matchesType;
  });

  renderHistoryTable(filtered);
}

// ─────────────────────────────────────────────────────────────────────────────
// Model Comparison Page
// ─────────────────────────────────────────────────────────────────────────────

async function loadComparisonData() {
  try {
    const res = await fetch(`${API_BASE}/models/compare`);
    if (!res.ok) return;

    const data = await res.json();
    const models = data.all_models || [];

    // Render Table
    const tbody = document.getElementById("compareTableBody");
    tbody.innerHTML = "";

    models.sort((a, b) => (b.f1_score || 0) - (a.f1_score || 0)).forEach(m => {
      const tr = document.createElement("tr");
      const rep = String(m.representation || "");
      tr.innerHTML = `
        <td><strong style="color: ${rep === 'TFIDF' ? '#e67e22' : '#3498db'};">${escapeHtml(rep)}</strong></td>
        <td>${escapeHtml(m.model_name || "")}</td>
        <td>${((m.accuracy || 0) * 100).toFixed(2)}%</td>
        <td>${((m.precision || 0) * 100).toFixed(2)}%</td>
        <td><strong>${((m.recall || 0) * 100).toFixed(2)}%</strong></td>
        <td><strong>${(m.f1_score || 0).toFixed(4)}</strong></td>
        <td>${(m.roc_auc || 0).toFixed(4)}</td>
        <td style="font-size: 0.8rem;">${m.latency_per_sample_ms ?? 0} ms</td>
      `;
      tbody.appendChild(tr);
    });

    // Render Comparison Chart
    renderModelComparisonChart(models);

  } catch (err) {
    console.warn("Could not load model comparison:", err);
  }
}

function renderModelComparisonChart(models) {
  const ctx = document.getElementById("chartModelComparison").getContext("2d");
  if (chartComparisonInstance) chartComparisonInstance.destroy();

  const isDark = document.documentElement.getAttribute("data-theme") === "dark";
  const textColor = isDark ? "#f8fafc" : "#0f172a";
  const gridColor = isDark ? "#334155" : "#e2e8f0";

  // Group unique model architectures
  const uniqueArchs = Array.from(new Set(models.map(m => m.model_name)));
  const bowScores = uniqueArchs.map(arch => {
    const m = models.find(x => x.model_name === arch && x.representation === "BOW");
    return m ? m.f1_score : 0;
  });
  const tfidfScores = uniqueArchs.map(arch => {
    const m = models.find(x => x.model_name === arch && x.representation === "TFIDF");
    return m ? m.f1_score : 0;
  });

  chartComparisonInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels: uniqueArchs,
      datasets: [
        {
          label: "Bag-of-Words (BoW)",
          data: bowScores,
          backgroundColor: "#3498db",
          borderRadius: 4
        },
        {
          label: "TF-IDF Representation",
          data: tfidfScores,
          backgroundColor: "#e67e22",
          borderRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      scales: {
        x: { ticks: { color: textColor }, grid: { display: false } },
        y: {
          ticks: { color: textColor },
          grid: { color: gridColor },
          min: 0.85,
          max: 1.0
        }
      },
      plugins: {
        legend: {
          labels: { color: textColor }
        }
      }
    }
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

function setLoading(btnId, spinner, isLoading) {
  const btn = document.getElementById(btnId);
  if (!btn) return;
  const text = btn.querySelector(".btn-text");

  if (isLoading) {
    spinner.style.display = "inline-block";
    btn.disabled = true;
    if (text) text.style.opacity = "0.7";
  } else {
    spinner.style.display = "none";
    btn.disabled = false;
    if (text) text.style.opacity = "1";
  }
}

function showError(box, msg) {
  box.textContent = `⚠️ ${msg}`;
  box.style.display = "block";
}

function hideError(box) {
  box.style.display = "none";
}

