// TopoPyc - Snappy Interaction Script (Zero Bloat)

// 1. One-Click Install Command Copy (Bun Style)
const installBar = document.getElementById('installCommand');
const copyBtn = installBar?.querySelector('.copy-btn');
const installCode = installBar?.querySelector('.install-code');

if (copyBtn && installCode) {
  copyBtn.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(installCode.textContent.trim());
      copyBtn.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2.5">
          <polyline points="20 6 9 17 4 12"></polyline>
        </svg>
      `;
      setTimeout(() => {
        copyBtn.innerHTML = `
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
            <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"></path>
          </svg>
        `;
      }, 2000);
    } catch (err) {
      console.error('Failed to copy to clipboard', err);
    }
  });
}

// 2. Interactive Kernel Switcher Tabs
const KERNEL_DATA = {
  mandelbrot: {
    code: `@topojit(opt_level="-O3")
def kernel_mandelbrot(c_re: float, c_im: float, max_iter: int) -> int:
    z_re = 0.0
    z_im = 0.0
    for i in range(max_iter):
        z_re2 = z_re * z_re
        z_im2 = z_im * z_im
        if (z_re2 + z_im2) > 4.0:
            return i
        z_im = 2.0 * z_re * z_im + c_im
        z_re = z_re2 - z_im2 + c_re
    return max_iter`,
    v: "11 cells",
    e: "14 cells",
    t: "6 cells",
    h0: "1",
    h1: "0 (Monomorphic)",
    h2: "0 (Lock-Free)"
  },
  attention: {
    code: `@topojit(opt_level="-O3")
def attention_reduction(dim: int, query_val: float) -> float:
    acc_score = 0.0
    inv_sqrt_d = 1.0 / math.sqrt(float(dim))
    for i in range(dim):
        key_val = math.sin(float(i) * 0.1) * 0.5 + 0.5
        val_val = math.cos(float(i) * 0.05)
        score = (query_val * key_val) * inv_sqrt_d
        weight = math.exp(score * 0.1)
        acc_score += weight * val_val
    return acc_score`,
    v: "14 cells",
    e: "18 cells",
    t: "8 cells",
    h0: "1",
    h1: "0 (Monomorphic)",
    h2: "0 (Lock-Free)"
  },
  gelu: {
    code: `@topojit(opt_level="-Ofast")
def gelu_poly(x_val: float, steps: int) -> float:
    acc = x_val
    for i in range(steps):
        x3 = acc * acc * acc
        inner = 0.79788456 * (acc + 0.044715 * x3)
        e2 = math.exp(inner if inner < 15.0 else 15.0)
        tanh_val = (e2 - 1.0) / (e2 + 1.0)
        acc = 0.5 * acc * (1.0 + tanh_val)
    return acc`,
    v: "16 cells",
    e: "22 cells",
    t: "9 cells",
    h0: "1",
    h1: "0 (Monomorphic)",
    h2: "0 (Lock-Free)"
  }
};

const codeViewer = document.getElementById('codeViewer');
const metricV = document.getElementById('metricV');
const metricE = document.getElementById('metricE');
const metricT = document.getElementById('metricT');
const metricH0 = document.getElementById('metricH0');
const metricH1 = document.getElementById('metricH1');
const metricH2 = document.getElementById('metricH2');

const tabButtons = document.querySelectorAll('.tab-btn');

tabButtons.forEach(btn => {
  btn.addEventListener('click', () => {
    tabButtons.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    const kernelKey = btn.getAttribute('data-kernel');
    const data = KERNEL_DATA[kernelKey];
    if (data && codeViewer) {
      codeViewer.textContent = data.code;
      if (metricV) metricV.textContent = data.v;
      if (metricE) metricE.textContent = data.e;
      if (metricT) metricT.textContent = data.t;
      if (metricH0) metricH0.textContent = data.h0;
      if (metricH1) metricH1.textContent = data.h1;
      if (metricH2) metricH2.textContent = data.h2;
    }
  });
});

// 3. Smooth Anchor Scrolling
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function(e) {
    const target = document.querySelector(this.getAttribute('href'));
    if (target) {
      e.preventDefault();
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  });
});
