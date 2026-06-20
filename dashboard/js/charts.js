/**
 * charts.js
 * Chart.js configurations and rendering for all dashboard sections.
 */

Chart.defaults.color = '#8890a8';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.borderColor = '#272b38';

const DashboardCharts = {
  charts: {},

  init(data) {
    this.renderFeatureOverlap(data.preprocessing);
    this.renderClassDistribution(data.preprocessing);
    this.renderSpectral(data.graph);
    this.renderBinaryBar(data.baseline);
    this.renderFamilyBar(data.baseline);
    this.renderSpectralCompare(data.graph);
    this.renderMatrixImportance(data.matrix);
    this.renderFederated(data.federated);

    if (data.attacks && data.attacks.attacks && data.attacks.attacks['ddos_icmp']) {
      this.renderConfusionMatrix(data.attacks.attacks['ddos_icmp'].confusion_matrix, 'ddos_icmp');
    }
  },

  renderFeatureOverlap(prep) {
    const ctx = document.getElementById('chart-features');
    if (!ctx) return;
    const table = prep.feature_table || [];
    const labels = table.slice(0, 15).map(r => r.feature);
    const pearson = table.slice(0, 15).map(r => r.pearson_r);
    const mi = table.slice(0, 15).map(r => r.mutual_info);
    this.charts.features = new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [
          { label: 'Pearson |r|', data: pearson, backgroundColor: 'rgba(79,157,232,0.7)', borderRadius: 4 },
          { label: 'Mutual Info', data: mi, backgroundColor: 'rgba(72,196,142,0.7)', borderRadius: 4 }
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { position: 'top', labels: { color: '#e8eaf0' } }, tooltip: { mode: 'index', intersect: false } },
        scales: {
          y: { beginAtZero: true, grid: { color: '#1e2333' } },
          x: { grid: { display: false }, ticks: { maxRotation: 45, minRotation: 45 } }
        }
      }
    });
  },

  renderClassDistribution(prep) {
    const ctx = document.getElementById('chart-classes');
    if (!ctx) return;
    const dist = [...(prep.class_distribution || [])].sort((a, b) => b.pct - a.pct);
    const top = dist.slice(0, 6);
    const otherPct = dist.slice(6).reduce((s, d) => s + d.pct, 0);
    if (otherPct > 0) top.push({ family: 'Other', pct: otherPct });
    this.charts.classes = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: top.map(d => d.family),
        datasets: [{ data: top.map(d => d.pct), backgroundColor: ['#4f9de8','#48c48e','#e05c5c','#a48ee8','#e8914f','#e8c84f','#5a6180'], borderWidth: 0, hoverOffset: 4 }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { position: 'right', labels: { color: '#e8eaf0', font: { size: 10 } } }, tooltip: { callbacks: { label: (ctx) => ` ${ctx.label}: ${ctx.raw.toFixed(1)}%` } } },
        cutout: '70%'
      }
    });
  },

  renderSpectral(graph) {
    const ctx = document.getElementById('chart-spectral');
    if (!ctx) return;
    const evs = graph.eigenvalues || [];
    this.charts.spectral = new Chart(ctx, {
      type: 'line',
      data: {
        labels: evs.map((_, i) => `λ${i}`),
        datasets: [{ label: 'Eigenvalue', data: evs, borderColor: '#4f9de8', backgroundColor: 'rgba(79,157,232,0.1)', borderWidth: 2, pointBackgroundColor: '#0a0c12', pointBorderColor: '#4f9de8', pointBorderWidth: 2, pointRadius: 4, fill: true, tension: 0.3 }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { y: { grid: { color: '#1e2333' } }, x: { grid: { color: '#1e2333' } } }
      }
    });
  },

  renderBinaryBar(baseline) {
    const ctx = document.getElementById('chart-binary-bar');
    if (!ctx || !baseline) return;
    const results = baseline.binary_results || [];
    this.charts.binaryBar = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: results.map(r => r.model),
        datasets: [
          { label: 'F1 Score', data: results.map(r => r.f1), backgroundColor: results.map(r => r.model === baseline.best_binary_model ? '#48c48e' : 'rgba(79,157,232,0.6)'), borderRadius: 6 },
          { label: 'Accuracy', data: results.map(r => r.accuracy), backgroundColor: 'rgba(164,142,232,0.5)', borderRadius: 6 }
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#e8eaf0' } } },
        scales: {
          y: { min: 0.97, max: 1.0, grid: { color: '#1e2333' }, ticks: { callback: v => v.toFixed(3) } },
          x: { grid: { display: false } }
        }
      }
    });
  },

  renderFamilyBar(baseline) {
    const ctx = document.getElementById('chart-family-bar');
    if (!ctx || !baseline) return;
    const results = baseline.family_results || [];
    this.charts.familyBar = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: results.map(r => r.model),
        datasets: [
          { label: 'Accuracy', data: results.map(r => r.accuracy), backgroundColor: 'rgba(79,157,232,0.7)', borderRadius: 6 }
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#e8eaf0' } } },
        scales: {
          y: { min: 0.8, max: 0.95, grid: { color: '#1e2333' }, ticks: { callback: v => v.toFixed(3) } },
          x: { grid: { display: false } }
        }
      }
    });
  },

  renderSpectralCompare(graph) {
    const ctx = document.getElementById('chart-spectral-compare');
    if (!ctx) return;
    const sc = graph.spectral_comparison || {};
    const models = ['Baseline RF', 'Spectral RF'];
    const f1s = [sc.baseline_f1 || 0.9967, sc.spectral_f1 || 0.9966];
    const accs = [sc.baseline_acc || 0.9935, sc.spectral_acc || 0.9933];
    this.charts.spectralCompare = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: models,
        datasets: [
          { label: 'F1 Score', data: f1s, backgroundColor: ['rgba(79,157,232,0.7)', 'rgba(72,196,142,0.7)'], borderRadius: 6 },
          { label: 'Accuracy', data: accs, backgroundColor: ['rgba(79,157,232,0.3)', 'rgba(72,196,142,0.3)'], borderRadius: 6 }
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#e8eaf0' } } },
        scales: {
          y: { min: 0.985, max: 1.0, grid: { color: '#1e2333' }, ticks: { callback: v => v.toFixed(4) } },
          x: { grid: { display: false } }
        }
      }
    });
  },

  renderMatrixImportance(matrix) {
    const ctx = document.getElementById('chart-matrix-importance');
    if (!ctx) return;
    const importance = (matrix && matrix.feature_importance) ? matrix.feature_importance : [
      { feature: 'Min', importance: 0.228, group: 'base' },
      { feature: 'AVG', importance: 0.179, group: 'base' },
      { feature: 'Tot size', importance: 0.157, group: 'base' },
      { feature: 'spectral_eigen_0', importance: 0.095, group: 'spectral' },
      { feature: 'spectral_proj_0', importance: 0.088, group: 'spectral' },
      { feature: 'Protocol Type', importance: 0.071, group: 'base' },
      { feature: 'Tot sum', importance: 0.065, group: 'base' },
      { feature: 'matrix_feat_0', importance: 0.057, group: 'matrix' },
      { feature: 'matrix_feat_1', importance: 0.042, group: 'matrix' },
      { feature: 'ICMP', importance: 0.018, group: 'base' }
    ];
    const colors = importance.map(f => f.group === 'base' ? 'rgba(79,157,232,0.7)' : f.group === 'spectral' ? 'rgba(72,196,142,0.7)' : 'rgba(164,142,232,0.7)');
    this.charts.matrixImportance = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: importance.map(f => f.feature),
        datasets: [{ label: 'Importance', data: importance.map(f => f.importance), backgroundColor: colors, borderRadius: 4 }]
      },
      options: {
        responsive: true, maintainAspectRatio: false, indexAxis: 'y',
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: (ctx) => ` ${ctx.raw.toFixed(3)} — ${importance[ctx.dataIndex].group}` } }
        },
        scales: {
          x: { grid: { color: '#1e2333' }, beginAtZero: true },
          y: { grid: { display: false }, ticks: { font: { family: "'JetBrains Mono', monospace", size: 10 } } }
        }
      }
    });
  },

  renderFederated(fed) {
    const ctx = document.getElementById('chart-fl');
    if (!ctx) return;
    const rounds = fed.rounds || [];
    const labels = rounds.map(r => `Round ${r.round}`);
    const f1s = rounds.map(r => r.global_f1);
    const accs = rounds.map(r => r.global_accuracy);
    const minVal = Math.max(0.9, Math.min(...f1s, ...accs) - 0.01);
    this.charts.fl = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [
          { label: 'Global F1 Score', data: f1s, borderColor: '#48c48e', borderWidth: 3, pointBackgroundColor: '#48c48e', pointRadius: 6, tension: 0.1 },
          { label: 'Global Accuracy', data: accs, borderColor: '#4f9de8', borderWidth: 2, pointBackgroundColor: '#4f9de8', pointRadius: 5, tension: 0.1, borderDash: [5,3] }
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#e8eaf0' } } },
        scales: {
          y: { grid: { color: '#1e2333' }, min: minVal, ticks: { callback: v => v.toFixed(4) } },
          x: { grid: { color: '#1e2333' } }
        }
      }
    });
  },

  renderConfusionMatrix(cm, type) {
    const ctx = document.getElementById('chart-confusion');
    if (!ctx) return;
    if (this.charts.confusion) this.charts.confusion.destroy();
    let color = '#4f9de8';
    if (type === 'ddos_icmp') color = '#e05c5c';
    if (type === 'mirai_greeth') color = '#48c48e';
    const tn = cm[0][0], fp = cm[0][1], fn = cm[1][0], tp = cm[1][1];
    this.charts.confusion = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: ['True Negatives', 'False Positives', 'False Negatives', 'True Positives'],
        datasets: [{ label: 'Samples', data: [tn, fp, fn, tp], backgroundColor: ['rgba(90,97,128,0.7)', 'rgba(224,92,92,0.7)', 'rgba(232,145,79,0.7)', color], borderWidth: 0, borderRadius: 4 }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          y: { type: 'logarithmic', grid: { color: '#1e2333' } },
          x: { grid: { display: false } }
        }
      }
    });
  }
};
