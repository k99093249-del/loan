async function loadDashboard() {
  const res = await fetch("/api/dashboard");
  if (!res.ok) return;
  const data = await res.json();

  document.getElementById("kpiTotal").textContent = data.total;
  document.getElementById("kpiApp").textContent = data.approvals;
  document.getElementById("kpiRej").textContent = data.rejections;
  document.getElementById("kpiRate").textContent =
    `${Math.round((data.approval_rate || 0) * 100)}%`;

  const ctx1 = document.getElementById("chartMain");
  if (!ctx1) return;
  new Chart(ctx1, {
    type: "bar",
    data: {
      labels: ["Approved", "Rejected"],
      datasets: [
        {
          label: "Count",
          data: [data.approvals || 0, data.rejections || 0],
          backgroundColor: ["#22d3ee", "#a78bfa"],
        },
      ],
    },
    options: { responsive: true, plugins: { legend: { display: false } } },
  });

  const empOrder = data.employment_order || ["Student", "Employed", "Self Employed"];
  const br = data.employment_breakdown || {};
  const ctx2 = document.getElementById("chartEmp");
  new Chart(ctx2, {
    type: "bar",
    data: {
      labels: empOrder,
      datasets: [
        {
          label: "Approved",
          data: empOrder.map((k) => (br[k] && br[k].Y) || 0),
          backgroundColor: "#22d3ee",
        },
        {
          label: "Rejected",
          data: empOrder.map((k) => (br[k] && br[k].N) || 0),
          backgroundColor: "#a78bfa",
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        x: { stacked: false, grid: { display: false } },
        y: { beginAtZero: true, ticks: { precision: 0 } },
      },
    },
  });

  const propLabels = Object.keys(data.by_property_area || {});
  const propVals = propLabels.map((k) => data.by_property_area[k]);
  const ctx3 = document.getElementById("chartProp");
  new Chart(ctx3, {
    type: "pie",
    data: {
      labels: propLabels.length ? propLabels : ["No data"],
      datasets: [{ data: propVals.length ? propVals : [1], backgroundColor: ["#22d3ee", "#a78bfa", "#34d399"] }],
    },
    options: { responsive: true },
  });
}

loadDashboard();
