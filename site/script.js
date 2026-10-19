// TopoPyc - Interactive Holographic Simplicial Canvas & Kernel Switcher

// 1. Simplicial Dynamic Canvas Animation
const canvas = document.getElementById('topologyCanvas');
const ctx = canvas.getContext('2d');

let width, height;
function resize() {
  width = canvas.width = window.innerWidth;
  height = canvas.height = window.innerHeight;
}
window.addEventListener('resize', resize);
resize();

// Topological Simplex Vertices
const NUM_VERTICES = 45;
const vertices = [];

for (let i = 0; i < NUM_VERTICES; i++) {
  vertices.push({
    x: Math.random() * width,
    y: Math.random() * height,
    vx: (Math.random() - 0.5) * 0.8,
    vy: (Math.random() - 0.5) * 0.8,
    radius: Math.random() * 2.5 + 1.5,
    type: Math.random() > 0.5 ? 'float64' : 'int64'
  });
}

function drawSimplicialComplex() {
  ctx.clearRect(0, 0, width, height);

  // Update positions
  for (let v of vertices) {
    v.x += v.vx;
    v.y += v.vy;

    if (v.x < 0 || v.x > width) v.vx *= -1;
    if (v.y < 0 || v.y > height) v.vy *= -1;
  }

  // Draw 2-simplexes (triangles) and 1-simplexes (edges)
  for (let i = 0; i < NUM_VERTICES; i++) {
    for (let j = i + 1; j < NUM_VERTICES; j++) {
      const d1 = Math.hypot(vertices[i].x - vertices[j].x, vertices[i].y - vertices[j].y);
      if (d1 < 140) {
        // Draw 1-Simplex Edge
        ctx.strokeStyle = `rgba(0, 240, 255, ${0.35 * (1 - d1 / 140)})`;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(vertices[i].x, vertices[i].y);
        ctx.lineTo(vertices[j].x, vertices[j].y);
        ctx.stroke();

        // Check for 2-simplex triangle closure
        for (let k = j + 1; k < NUM_VERTICES; k++) {
          const d2 = Math.hypot(vertices[j].x - vertices[k].x, vertices[j].y - vertices[k].y);
          const d3 = Math.hypot(vertices[i].x - vertices[k].x, vertices[i].y - vertices[k].y);
          if (d2 < 140 && d3 < 140) {
            // Draw 2-Simplex Face
            ctx.fillStyle = `rgba(121, 40, 202, ${0.08 * (1 - (d1 + d2 + d3) / 420)})`;
            ctx.beginPath();
            ctx.moveTo(vertices[i].x, vertices[i].y);
            ctx.lineTo(vertices[j].x, vertices[j].y);
            ctx.lineTo(vertices[k].x, vertices[k].y);
            ctx.closePath();
            ctx.fill();
          }
        }
      }
    }
  }

  // Draw 0-Simplex Vertices
  for (let v of vertices) {
    ctx.beginPath();
    ctx.arc(v.x, v.y, v.radius, 0, Math.PI * 2);
    ctx.fillStyle = v.type === 'float64' ? '#00f0ff' : '#a855f7';
    ctx.shadowBlur = 8;
    ctx.shadowColor = v.type === 'float64' ? '#00f0ff' : '#a855f7';
    ctx.fill();
    ctx.shadowBlur = 0;
  }

  requestAnimationFrame(drawSimplicialComplex);
}
drawSimplicialComplex();

// 2. Interactive Simulator Kernel Presets
const KERNELS = {
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
    v: 11,
    e: 14,
    t: 6,
    h0: "1 (Связен)",
    h1: "0 (Мономорфен)",
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
    v: 14,
    e: 18,
    t: 8,
    h0: "1 (Связен)",
    h1: "0 (Мономорфен)",
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
    v: 16,
    e: 22,
    t: 9,
    h0: "1 (Связен)",
    h1: "0 (Мономорфен)",
    h2: "0 (Lock-Free)"
  }
};

const codeView = document.getElementById('kernelCodeView');
const statV = document.getElementById('statV');
const statE = document.getElementById('statE');
const statT = document.getElementById('statT');
const statH0 = document.getElementById('statH0');
const statH1 = document.getElementById('statH1');
const statH2 = document.getElementById('statH2');
const simButtons = document.querySelectorAll('.sim-btn');

simButtons.forEach(btn => {
  btn.addEventListener('click', () => {
    simButtons.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    const kernelKey = btn.getAttribute('data-kernel');
    const data = KERNELS[kernelKey];
    if (data) {
      codeView.textContent = data.code;
      statV.textContent = data.v;
      statE.textContent = data.e;
      statT.textContent = data.t;
      statH0.textContent = data.h0;
      statH1.textContent = data.h1;
      statH2.textContent = data.h2;
    }
  });
});

// Smooth scroll for nav links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function (e) {
    e.preventDefault();
    const target = document.querySelector(this.getAttribute('href'));
    if (target) {
      target.scrollIntoView({
        behavior: 'smooth',
        block: 'start'
      });
    }
  });
});
