// ─── Config ──────────────────────────────────────────────────────────────────
const API = "";

// ─── Leaflet Map ─────────────────────────────────────────────────────────────
const map = L.map("map", { preferCanvas: true, zoomControl: true }).setView([12.9716, 77.5946], 12);
const cellLayer = L.layerGroup().addTo(map);
const hotspotLayer = L.layerGroup().addTo(map);

L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
  attribution: "&copy; OpenStreetMap contributors &copy; CARTO",
  subdomains: "abcd",
  maxZoom: 19,
}).addTo(map);

// ─── State ───────────────────────────────────────────────────────────────────
let forecastChart, hourlyChart, junctionChart, importanceChart;
let activeHotspotId = null;
let cachedHotspots = [];

// ─── Utilities ───────────────────────────────────────────────────────────────
function pcisColor(pcis) {
  if (pcis >= 0.85) return "#ff5d73";
  if (pcis >= 0.72) return "#ffb020";
  if (pcis >= 0.55) return "#4f8cff";
  return "#22c997";
}

function riskClass(level) {
  return (level || "moderate").toLowerCase();
}

async function fetchJson(url) {
  const res = await fetch(`${API}${url}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// ─── Stats Cards ─────────────────────────────────────────────────────────────
function renderStats(summary) {
  const cards = [
    ["Total Violations", summary.total_violations.toLocaleString(), "Cleaned & deduplicated", "#4f8cff"],
    ["Critical / Severe", `${summary.critical_hotspots} / ${summary.severe_hotspots || 0}`, `Avg PCIS ${summary.avg_pcis}`, "#ff5d73"],
    ["Peak Hour (IST)", `${summary.peak_hour_ist}:00`, summary.top_police_station, "#ffb020"],
    ["Est. Daily Delay", `${Math.round(summary.estimated_city_delay_min_per_day).toLocaleString()} min`, "Preventable congestion", "#22c997"],
  ];
  document.getElementById("stats-grid").innerHTML = cards
    .map(([label, value, sub, glow]) => `
      <div class="stat-card">
        <div class="label">${label}</div>
        <div class="value" style="text-shadow:0 0 20px ${glow}55">${value}</div>
        <div class="sub">${sub}</div>
      </div>`)
    .join("");
}

// ─── Heatmap ──────────────────────────────────────────────────────────────────
function renderHeatmap(cells) {
  cellLayer.clearLayers();
  document.getElementById("heatmap-count").textContent = `${cells.length} H3 cells mapped`;

  const chunk = 150;
  let i = 0;
  const draw = () => {
    const end = Math.min(i + chunk, cells.length);
    for (; i < end; i++) {
      const c = cells[i];
      const color = pcisColor(c.pcis);
      const r = Math.max(5, c.pcis * 16);

      // outer glow ring
      L.circleMarker([c.latitude, c.longitude], {
        radius: r + 5,
        color: color,
        fillColor: color,
        fillOpacity: 0.08,
        weight: 0,
      }).addTo(cellLayer);

      // main dot
      L.circleMarker([c.latitude, c.longitude], {
        radius: r,
        color: "rgba(255,255,255,0.1)",
        fillColor: color,
        fillOpacity: 0.78,
        weight: 1,
      })
        .bindPopup(`<strong>PCIS ${c.pcis.toFixed(2)}</strong><br/>Violations: ${c.violation_count || ""}`)
        .addTo(cellLayer);
    }
    if (i < cells.length) requestAnimationFrame(draw);
  };
  requestAnimationFrame(draw);
}

// ─── Hotspot Markers ──────────────────────────────────────────────────────────
function renderHotspotMarkers(hotspots) {
  hotspotLayer.clearLayers();
  hotspots.slice(0, 15).forEach(h => {
    const color = pcisColor(h.pcis);
    const icon = L.divIcon({
      html: `<div style="
        width:16px;height:16px;border-radius:50%;
        background:${color};
        border:2.5px solid rgba(255,255,255,0.9);
        box-shadow:0 0 14px ${color}cc,0 0 4px rgba(0,0,0,0.6);
      "></div>`,
      className: "",
      iconSize: [16, 16],
      iconAnchor: [8, 8],
    });

    L.marker([h.latitude, h.longitude], { icon })
      .bindPopup(`<strong>${h.name}</strong><br/>PCIS: ${h.pcis} &nbsp; Risk: ${h.risk_level}`)
      .bindTooltip(h.name.slice(0, 28), { direction: "top" })
      .addTo(hotspotLayer);
  });
}

// ─── Hotspot Table ────────────────────────────────────────────────────────────
function renderHotspots(hotspots) {
  cachedHotspots = hotspots;
  document.getElementById("hotspots-body").innerHTML = hotspots
    .map(h => `
      <tr class="hotspot-row ${h.hotspot_id === activeHotspotId ? "active" : ""}"
          data-id="${h.hotspot_id}" data-lat="${h.latitude}" data-lon="${h.longitude}">
        <td>${h.name.slice(0, 40)}${h.name.length > 40 ? "…" : ""}</td>
        <td><strong>${Number(h.pcis).toFixed(2)}</strong></td>
        <td><span class="chip ${riskClass(h.risk_level)}">${h.risk_level}</span></td>
        <td>${Number(h.estimated_delay_min_per_hour).toFixed(1)}m</td>
      </tr>`)
    .join("");

  document.querySelectorAll(".hotspot-row").forEach(row => {
    row.addEventListener("click", () => {
      activeHotspotId = row.dataset.id;
      map.flyTo([Number(row.dataset.lat), Number(row.dataset.lon)], 14, { duration: 1.2 });
      loadForecast(activeHotspotId);
      renderHotspots(cachedHotspots);
    });
  });

  renderHotspotMarkers(hotspots);
  if (hotspots[0] && !activeHotspotId) {
    activeHotspotId = hotspots[0].hotspot_id;
    loadForecast(activeHotspotId);
  }
}

// ─── Plan Table ───────────────────────────────────────────────────────────────
function renderPlan(plan) {
  document.getElementById("plan-body").innerHTML = plan
    .map(p => `
      <tr>
        <td>${p.rank}</td>
        <td>${p.name.slice(0, 34)}</td>
        <td>${p.recommended_window}</td>
        <td>${p.officers_needed}${p.tow_truck ? " + 🚗" : ""}</td>
        <td style="color:var(--accent-2);font-weight:600;">${Number(p.estimated_delay_saved_min).toFixed(1)}m</td>
      </tr>`)
    .join("");
}

// ─── Metrics ──────────────────────────────────────────────────────────────────
function renderMetrics(metrics) {
  document.getElementById("model-grade").textContent = metrics.model_grade || "GridLock ML Engine";
  const boxes = [
    ["Clustering Silhouette", metrics.clustering_silhouette?.toFixed(3), null],
    ["Forecast MAPE", metrics.forecast_mape != null ? `${metrics.forecast_mape.toFixed(1)}%` : null, "#22c997"],
    ["Forecast RMSE", metrics.forecast_rmse?.toFixed(2), null],
    ["PCIS R²", metrics.pcis_r2?.toFixed(3), "#22c997"],
    ["PCIS MAE", metrics.pcis_mae?.toFixed(3), null],
    ["Validation Records", metrics.validation_records?.toLocaleString(), null],
  ];
  document.getElementById("metrics-grid").innerHTML = boxes
    .map(([name, score, color]) => `
      <div class="metric-box">
        <div class="name">${name}</div>
        <div class="score"${color ? ` style="color:${color};text-shadow:0 0 14px ${color}66"` : ""}>${score ?? "–"}</div>
      </div>`)
    .join("");
}

// ─── Chart Helpers ────────────────────────────────────────────────────────────
function makeChart(id, existing, config) {
  if (existing) existing.destroy();
  return new Chart(document.getElementById(id), config);
}

function baseOpts(yLabel = "") {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 500 },
    plugins: { legend: { display: false } },
    scales: {
      x: {
        ticks: { color: "#93a0bd", maxRotation: 0, font: { family: "'Plus Jakarta Sans', sans-serif", size: 11 } },
        grid: { display: false },
        border: { display: false },
      },
      y: {
        ticks: { color: "#93a0bd", font: { family: "'Plus Jakarta Sans', sans-serif", size: 11 } },
        grid: { color: "rgba(255,255,255,0.04)", borderDash: [4, 4] },
        border: { display: false },
        title: { display: !!yLabel, text: yLabel, color: "#93a0bd" },
      },
    },
  };
}

function mkGrad(ctx, x0, y0, x1, y1, c1, c2) {
  const g = ctx.createLinearGradient(x0, y0, x1, y1);
  g.addColorStop(0, c1);
  g.addColorStop(1, c2);
  return g;
}

// ─── Forecast Chart ───────────────────────────────────────────────────────────
async function loadForecast(hotspotId) {
  try {
    const data = await fetchJson(`/api/v1/hotspots/${hotspotId}/forecast`);
    const ctx = document.getElementById("forecast-chart").getContext("2d");
    const areaGrad = mkGrad(ctx, 0, 0, 0, 130, "rgba(79,140,255,0.5)", "rgba(79,140,255,0.0)");

    forecastChart = makeChart("forecast-chart", forecastChart, {
      type: "line",
      data: {
        labels: data.map(d => `${d.hour}:00`),
        datasets: [{
          data: data.map(d => d.predicted_violations),
          borderColor: "#4f8cff",
          backgroundColor: areaGrad,
          fill: true, tension: 0.42,
          pointBackgroundColor: "#0b1020",
          pointBorderColor: "#4f8cff",
          pointBorderWidth: 2,
          pointRadius: 3,
          pointHoverRadius: 6,
        }],
      },
      options: baseOpts("Predicted violations"),
    });
  } catch (e) { console.warn("Forecast:", e.message); }
}

// ─── Hourly Chart ─────────────────────────────────────────────────────────────
function renderHourlyChart(data) {
  const ctx = document.getElementById("hourly-chart").getContext("2d");
  const colors = data.map(d => {
    const p = 0.35 + d.intensity * 0.65;
    const c = pcisColor(p);
    const r = parseInt(c.slice(1, 3), 16), g = parseInt(c.slice(3, 5), 16), b = parseInt(c.slice(5, 7), 16);
    return mkGrad(ctx, 0, 0, 0, 110, `rgba(${r},${g},${b},0.9)`, `rgba(${r},${g},${b},0.2)`);
  });

  hourlyChart = makeChart("hourly-chart", hourlyChart, {
    type: "bar",
    data: {
      labels: data.map(d => `${d.hour}:00`),
      datasets: [{ data: data.map(d => d.violations), backgroundColor: colors, borderRadius: 5, borderSkipped: false }],
    },
    options: baseOpts("Count"),
  });
}

// ─── Junction Chart ───────────────────────────────────────────────────────────
function renderJunctionChart(data) {
  const ctx = document.getElementById("junction-chart").getContext("2d");
  const grad = mkGrad(ctx, 300, 0, 0, 0, "rgba(79,140,255,0.9)", "rgba(79,140,255,0.25)");

  junctionChart = makeChart("junction-chart", junctionChart, {
    type: "bar",
    data: {
      labels: data.map(d => d.junction.slice(0, 18)),
      datasets: [{ data: data.map(d => d.violations), backgroundColor: grad, borderRadius: 5, borderSkipped: false }],
    },
    options: { ...baseOpts("Count"), indexAxis: "y" },
  });
}

// ─── Importance Chart ─────────────────────────────────────────────────────────
function renderImportanceChart(data) {
  const ctx = document.getElementById("importance-chart").getContext("2d");
  const grad = mkGrad(ctx, 300, 0, 0, 0, "rgba(34,201,151,0.9)", "rgba(34,201,151,0.25)");

  importanceChart = makeChart("importance-chart", importanceChart, {
    type: "bar",
    data: {
      labels: data.map(d => d.feature),
      datasets: [{ data: data.map(d => d.importance), backgroundColor: grad, borderRadius: 5, borderSkipped: false }],
    },
    options: { ...baseOpts(""), indexAxis: "y" },
  });
}

// ─── Simulator ────────────────────────────────────────────────────────────────
async function runSimulator() {
  const btn = document.getElementById("run-sim");
  btn.textContent = "Running…"; btn.disabled = true;
  try {
    const extra = document.getElementById("extra-teams").value;
    const result = await fetchJson(`/api/v1/simulate?extra_teams=${extra}&duration_hours=2`);
    document.getElementById("sim-result").innerHTML = `
      <div style="font-size:0.72rem;color:var(--muted);letter-spacing:.08em;text-transform:uppercase;margin-bottom:10px;">⚡ Mission Briefing</div>
      <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px;gap:8px;">
        <div>
          <div style="color:var(--muted);font-size:0.75rem;margin-bottom:2px;">Baseline delay</div>
          <div style="font-size:1.1rem;font-weight:600;">${result.baseline_delay_min_per_hour}
            <span style="font-size:0.78rem;color:var(--muted);">min/hr</span></div>
        </div>
        <div style="text-align:right;">
          <div style="color:var(--muted);font-size:0.75rem;margin-bottom:2px;">Saving</div>
          <div class="sim-val">−${result.projected_delay_reduction_min} min</div>
        </div>
      </div>
      <div style="display:flex;gap:18px;flex-wrap:wrap;margin-bottom:8px;">
        <div><div style="color:var(--muted);font-size:0.75rem;">Teams Deployed</div><strong>${result.extra_teams} × ${result.duration_hours}h</strong></div>
        <div><div style="color:var(--muted);font-size:0.75rem;">New Avg PCIS</div><strong>${result.projected_avg_pcis_after}</strong></div>
      </div>
      <div style="font-size:0.78rem;color:var(--muted);">Priority: ${result.top_zones_covered.slice(0, 3).join(", ")}…</div>`;
  } catch (e) {
    document.getElementById("sim-result").textContent = `Error: ${e.message}`;
  }
  btn.textContent = "Run What-If"; btn.disabled = false;
}

// ─── Load Plan ────────────────────────────────────────────────────────────────
async function loadPlan() {
  const btn = document.getElementById("refresh-plan");
  btn.textContent = "Loading…"; btn.disabled = true;
  try {
    const officers = document.getElementById("officer-count").value;
    const plan = await fetchJson(`/api/v1/enforcement/plan?officers=${officers}`);
    renderPlan(plan);
  } finally { btn.textContent = "Generate Plan"; btn.disabled = false; }
}

// ─── Bootstrap ────────────────────────────────────────────────────────────────
async function bootstrap() {
  try {
    const t0 = performance.now();
    const data = await fetchJson("/api/v1/dashboard/init?officers=6");
    renderStats(data.summary);
    renderHeatmap(data.heatmap);
    renderHotspots(data.hotspots);
    renderMetrics(data.metrics);
    renderHourlyChart(data.hourly);
    renderJunctionChart(data.junctions);
    renderImportanceChart(data.importance);
    renderPlan(data.plan);
    console.log(`✅ GridLock loaded in ${(performance.now() - t0).toFixed(0)}ms`);
  } catch (err) {
    console.error("Bootstrap error:", err);
    document.getElementById("model-grade").textContent = "⚠ Run python scripts/train_pipeline.py";
    document.getElementById("sim-result").textContent = `Error: ${err.message}`;
  }
}

document.getElementById("refresh-plan").addEventListener("click", loadPlan);
document.getElementById("run-sim").addEventListener("click", runSimulator);
bootstrap();
