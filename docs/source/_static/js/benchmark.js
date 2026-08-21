// Animated benchmark bar chart.
// Each bar grows on scroll into view; its time label fades in at the tip
// once the width transition ends.
// Data shape (injected by conf.py at build time):
//   window.BENCHMARK_DATA = {
//     subtitle: "192 CPUs, 749 GB RAM",  // chart-level default
//     groups: {
//       "Basic workflow": {
//         hardware: "cpu",
//         bars: { brisc, scanpy, seurat }
//       },
//       "Basic workflow (GPU)": {
//         hardware: "gpu",
//         note: "96 CPUs, 4× H100 GPU, ...",
//         bars: { brisc, rapids }
//       },
//       ...
//     }
//   };
(function() {
  const LIB_STYLE = {
    brisc:  { label: 'brisc',             cls: 'bar-brisc'  },
    scanpy: { label: 'Scanpy',            cls: 'bar-scanpy' },
    seurat: { label: 'Seurat + BPCells',  cls: 'bar-seurat' },
    rapids: { label: 'rapids-singlecell', cls: 'bar-rapids' },
  };

  function fmt(s) {
    if (s >= 3600) return (s / 3600).toFixed(1) + 'h';
    if (s >= 60) return (s / 60).toFixed(1) + 'm';
    return s.toFixed(0) + 's';
  }

  // How many times slower than brisc, e.g. "93.8×" or "1057×".
  function fmtMult(r) {
    return (r >= 100 ? Math.round(r) : r.toFixed(1)) + '×';
  }

  function build() {
    const container = document.getElementById('benchmark-chart');
    if (!container) return;
    const data = window.BENCHMARK_DATA;
    if (!data || !data.groups) {
      container.textContent = 'benchmark data unavailable';
      return;
    }

    const groups = data.groups;
    const entries = Object.entries(groups);

    entries.forEach(([workflow, group]) => {
      const groupEl = document.createElement('div');
      groupEl.className = 'bench-group';
      if (group.hardware === 'gpu') groupEl.classList.add('bench-group--gpu');

      const title = document.createElement('div');
      title.className = 'bench-title';
      title.textContent = workflow;
      groupEl.appendChild(title);

      if (group.note) {
        const note = document.createElement('div');
        note.className = 'bench-note';
        note.textContent = group.note;
        groupEl.appendChild(note);
      }

      const libs = Object.entries(group.bars)
        .filter(([k, t]) => typeof t === 'number' && LIB_STYLE[k])
        .map(([k, t]) => ({
          key: k,
          name: LIB_STYLE[k].label,
          cls: LIB_STYLE[k].cls,
          time: t,
        }));
      const maxTime = Math.max(...libs.map(l => l.time));
      // brisc is the baseline; other bars are labeled with how much slower.
      const base = group.bars.brisc;

      libs.forEach(lib => {
        const row = document.createElement('div');
        row.className = 'bench-row';

        const label = document.createElement('div');
        label.className = 'bench-label';
        label.textContent = lib.name;

        const track = document.createElement('div');
        track.className = 'bench-track';

        const bar = document.createElement('div');
        bar.className = 'bench-bar ' + lib.cls;
        const pct = ((lib.time / maxTime) * 100).toFixed(1);
        bar.dataset.pct = pct;
        bar.style.width = '0%';

        const val = document.createElement('span');
        val.className = 'bench-val';
        const mult = (lib.key !== 'brisc' && base)
          ? ' (' + fmtMult(lib.time / base) + ')' : '';
        val.textContent = fmt(lib.time) + mult;
        val.style.left = '0%';

        bar.addEventListener('transitionend', () => {
          val.style.left = pct + '%';
          val.classList.add('is-done');
        }, { once: true });

        track.appendChild(bar);
        track.appendChild(val);
        row.appendChild(label);
        row.appendChild(track);
        groupEl.appendChild(row);
      });

      container.appendChild(groupEl);
    });

    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const bars = container.querySelectorAll('.bench-bar');
          bars.forEach((bar, i) => {
            setTimeout(() => {
              bar.style.width = bar.dataset.pct + '%';
            }, i * 80);
          });
          observer.disconnect();
        }
      });
    }, { threshold: 0.2 });

    observer.observe(container);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', build);
  } else {
    build();
  }
})();
