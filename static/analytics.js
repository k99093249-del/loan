function fmtNumber(n) {
  const x = Number(n || 0);
  return x.toLocaleString(undefined);
}

function fmtMoneyUSD(m) {
  const x = Number(m || 0);
  if (x >= 1e9) return `$${(x / 1e9).toFixed(1)}B`;
  if (x >= 1e6) return `$${(x / 1e6).toFixed(1)}M`;
  if (x >= 1e3) return `$${(x / 1e3).toFixed(1)}K`;
  return `$${x.toFixed(0)}`;
}

function statusBadge(status) {
  const s = String(status || "").toLowerCase();
  if (s.includes("approve") || s === "y" || s === "yes") return ["Approved", "b-approved"];
  if (s.includes("reject") || s === "n" || s === "no") return ["Rejected", "b-rejected"];
  if (s.includes("pending")) return ["Pending", "b-pending"];
  return [status || "Pending", "b-pending"];
}

function predictionTag(label, probability) {
  const p = probability == null ? null : Number(probability);
  let txt = String(label || "").trim();
  if (!txt) txt = "—";
  if (p != null && !Number.isNaN(p)) {
    // if value is 0..1 convert to %
    const pct = p <= 1 ? Math.round(p * 100) : Math.round(p);
    return `${txt} (${pct}%)`;
  }
  return txt;
}

let tableState = {
  rows: [],
  query: "",
  page: 1,
  pageSize: 8,
  sortKey: "application_id",
  sortDir: "asc",
};

function applyTableState() {
  const q = tableState.query.trim().toLowerCase();
  let filtered = tableState.rows;
  if (q) {
    filtered = filtered.filter((r) =>
      Object.values(r)
        .join(" ")
        .toLowerCase()
        .includes(q)
    );
  }

  const key = tableState.sortKey;
  const dir = tableState.sortDir === "asc" ? 1 : -1;
  filtered = filtered.slice().sort((a, b) => {
    const av = a[key] ?? "";
    const bv = b[key] ?? "";
    const an = typeof av === "number" ? av : Number(av);
    const bn = typeof bv === "number" ? bv : Number(bv);
    if (!Number.isNaN(an) && !Number.isNaN(bn)) return (an - bn) * dir;
    return String(av).localeCompare(String(bv)) * dir;
  });

  const totalPages = Math.max(1, Math.ceil(filtered.length / tableState.pageSize));
  tableState.page = Math.min(tableState.page, totalPages);
  const start = (tableState.page - 1) * tableState.pageSize;
  const pageRows = filtered.slice(start, start + tableState.pageSize);
  document.getElementById("pageLabel").textContent = `${tableState.page} / ${totalPages}`;

  const tbody = document.querySelector("#recentTable tbody");
  tbody.innerHTML = "";
  for (const r of pageRows) {
    const tr = document.createElement("tr");
    const [sTxt, sClass] = statusBadge(r.status);
    const predTxt = predictionTag(r.prediction, r.probability);

    tr.innerHTML = `
      <td>${r.application_id ?? "-"}</td>
      <td>${r.applicant_name ?? "-"}</td>
      <td>${r.loan_type ?? "-"}</td>
      <td>${fmtMoneyUSD(r.amount || 0)}</td>
      <td><span class="badge ${sClass}">${sTxt}</span></td>
      <td><span class="tag">${predTxt}</span></td>
      <td><button class="eye" type="button" title="View">👁️</button></td>
    `;
    tbody.appendChild(tr);
  }
}

function wireTable() {
  document.getElementById("tableSearch").addEventListener("input", (e) => {
    tableState.query = e.target.value;
    tableState.page = 1;
    applyTableState();
  });
  document.getElementById("prevPage").addEventListener("click", () => {
    tableState.page = Math.max(1, tableState.page - 1);
    applyTableState();
  });
  document.getElementById("nextPage").addEventListener("click", () => {
    tableState.page += 1;
    applyTableState();
  });

  document.querySelectorAll("#recentTable thead th[data-sort]").forEach((th) => {
    th.addEventListener("click", () => {
      const k = th.getAttribute("data-sort");
      if (tableState.sortKey === k) {
        tableState.sortDir = tableState.sortDir === "asc" ? "desc" : "asc";
      } else {
        tableState.sortKey = k;
        tableState.sortDir = "asc";
      }
      applyTableState();
    });
  });
}

function buildInsights(items) {
  const root = document.getElementById("insights");
  root.innerHTML = "";
  const iconMap = {
    trend: "📈",
    star: "🏅",
    map: "🗺️",
    spark: "✨",
  };
  for (const it of items || []) {
    const el = document.createElement("div");
    el.className = "insight";
    el.innerHTML = `
      <div class="i">${iconMap[it.icon] || "✨"}</div>
      <div>
        <div class="t">${it.title || ""}</div>
        <div class="b">${it.body || ""}</div>
      </div>
    `;
    root.appendChild(el);
  }
}

function buildRisk(items) {
  const root = document.getElementById("riskFactors");
  root.innerHTML = "";
  for (const it of items || []) {
    const p = Math.max(0, Math.min(100, Number(it.percent || 0)));
    const row = document.createElement("div");
    row.className = "risk-row";
    row.innerHTML = `
      <div class="risk-top"><span>${it.label || ""}</span><strong>${p.toFixed(0)}%</strong></div>
      <div class="risk-bar"><div style="width:${p}%;"></div></div>
    `;
    root.appendChild(row);
  }
}

async function loadAnalytics() {
  const res = await fetch("/api/analytics");
  if (!res.ok) return;
  const data = await res.json();

  const k = data.kpis || {};
  document.getElementById("kpiTotal").textContent = fmtNumber(k.total_applications);
  document.getElementById("kpiApproved").textContent = fmtNumber(k.approved_loans);
  document.getElementById("kpiRejected").textContent = fmtNumber(k.rejected_loans);
  document.getElementById("kpiAmount").textContent = fmtMoneyUSD(k.total_loan_amount);

  const pct = Math.round((k.approval_rate || 0) * 1000) / 10;
  document.getElementById("approvalPct").textContent = `${pct}%`;
  document.getElementById("donutApproved").textContent = fmtNumber(k.approved_loans);
  document.getElementById("donutRejected").textContent = fmtNumber(k.rejected_loans);
  document.getElementById("donutTotal").textContent = fmtNumber(k.total_applications);

  buildInsights(data.insights || []);
  buildRisk(data.risk_factors || []);

  // Trend chart
  const trend = data.trend || [];
  const months = trend.map((x) => x.month);
  const apps = trend.map((x) => x.applications);
  const appr = trend.map((x) => x.approved);
  const rej = trend.map((x) => x.rejected);

  const trendCtx = document.getElementById("trendChart");
  new Chart(trendCtx, {
    type: "line",
    data: {
      labels: months,
      datasets: [
        {
          label: "Applications",
          data: apps,
          borderColor: "#2563eb",
          backgroundColor: "rgba(37,99,235,0.10)",
          tension: 0.35,
          fill: true,
          pointRadius: 2,
        },
        {
          label: "Approved",
          data: appr,
          borderColor: "#16a34a",
          backgroundColor: "rgba(22,163,74,0.10)",
          tension: 0.35,
          fill: false,
          pointRadius: 2,
        },
        {
          label: "Rejected",
          data: rej,
          borderColor: "#ef4444",
          backgroundColor: "rgba(239,68,68,0.10)",
          tension: 0.35,
          fill: false,
          pointRadius: 2,
        },
      ],
    },
    options: {
      responsive: true,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { position: "top" } },
      scales: {
        x: { grid: { display: false } },
        y: { beginAtZero: true, ticks: { precision: 0 } },
      },
    },
  });

  // Donut
  const donutCtx = document.getElementById("donutChart");
  new Chart(donutCtx, {
    type: "doughnut",
    data: {
      labels: ["Approved", "Rejected"],
      datasets: [
        {
          data: [k.approved_loans || 0, k.rejected_loans || 0],
          backgroundColor: ["#16a34a", "#ef4444"],
          borderWidth: 0,
          hoverOffset: 6,
        },
      ],
    },
    options: {
      cutout: "70%",
      plugins: { legend: { display: false } },
    },
  });

  // Income bar
  const inc = data.income_distribution || [];
  const incomeCtx = document.getElementById("incomeChart");
  new Chart(incomeCtx, {
    type: "bar",
    data: {
      labels: inc.map((x) => x.range),
      datasets: [
        {
          label: "Applicants",
          data: inc.map((x) => x.count),
          borderRadius: 10,
          backgroundColor: ["rgba(37,99,235,0.65)"],
        },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: { x: { grid: { display: false } }, y: { beginAtZero: true, ticks: { precision: 0 } } },
    },
  });

  // Purpose donut
  const purpose = data.loan_purpose || [];
  const purposeCtx = document.getElementById("purposeChart");
  new Chart(purposeCtx, {
    type: "doughnut",
    data: {
      labels: purpose.map((x) => x.label),
      datasets: [
        {
          data: purpose.map((x) => x.count),
          backgroundColor: ["#2563eb", "#22c55e", "#a78bfa", "#f59e0b", "#ef4444", "#06b6d4", "#64748b"],
          borderWidth: 0,
        },
      ],
    },
    options: { cutout: "64%", plugins: { legend: { position: "right" } } },
  });

  // Region hint: if state data exists, show top state
  const region = data.region || [];
  const hint = document.getElementById("regionHint");
  if (region.length) {
    hint.textContent = `Top state/region: ${region[0].state} (${region[0].applications} apps)`;
  } else {
    hint.textContent = "State/region columns were not found in the Excel dataset.";
  }

  // Table
  tableState.rows = data.recent || [];
  applyTableState();
}

wireTable();
loadAnalytics();

