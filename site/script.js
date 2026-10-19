// TopoPyc - Snappy Interaction Script with Theme & Multilingual i18n Engine

// ============================================================================
// 1. Multilingual Translation Dictionary (EN / RU)
// ============================================================================
const TRANSLATIONS = {
  en: {
    nav_benchmarks: "Benchmarks",
    nav_architecture: "Architecture",
    nav_interactive: "Interactive",
    nav_docs: "Docs",
    star_github: "GitHub",
    github_repo: "GitHub Repository ↗",

    hero_pill: "Discrete Exterior Calculus on Cellular Sheaves",
    hero_title_1: "Python syntax.",
    hero_title_2: "C bare-metal speed.",
    hero_desc: "TopoPyc transforms Python code into an n-dimensional simplicial complex with typed sheaves. Monomorphism is mathematically proven by vanishing 1st cohomology ker(Δ₁) = 0, compiling functions into unboxed register C99 with native AVX-512 vectorization.",

    metric_cpython: "Faster than CPython 3.13",
    metric_torch: "Faster than torch.compile",
    metric_gil: "GIL & Boxed Object Overhead",
    metric_proof: "Monomorphic Invariant Proof",

    bench_badge: "Benchmarks",
    bench_title: "Tested against production runtimes",
    bench_sub: "Real-world deep learning kernels measured across 10,000,000 continuous iterations on a single CPU core.",
    gelu_name: "Fused GELU Polynomial Kernel",
    gelu_iters: "10,000,000 iterations • Scalar Fused",
    attn_name: "Attention Dot-Product Score Reduction",
    attn_iters: "20,000,000 elements • Vector Loop",

    sim_badge: "Live Interactive Preview",
    sim_title: "See the topological reduction live",
    sim_sub: "Select a compute kernel to inspect how AST nodes translate into simplices, coboundary chains, and Hodge invariants.",
    tab_mandelbrot_title: "Mandelbrot Fractal",
    tab_mandelbrot_desc: "Dynamic branch & escape orbit",
    tab_attn_title: "Attention Reduction",
    tab_attn_desc: "Scaled dot-product sequence",
    tab_gelu_title: "Fused GELU Poly",
    tab_gelu_desc: "Hyperbolic polynomial activation",

    diag_title: "Hodge Sheaf Invariants",
    diag_status: "Verified",
    v_label: "0-Simplices (Vertices X⁽⁰⁾)",
    e_label: "1-Simplices (Def-Use Edges X⁽¹⁾)",
    t_label: "2-Simplices (Operator Faces X⁽²⁾)",
    h0_label: "dim H⁰(Δ₀) Connected Fiber",
    h1_label: "dim H¹(Δ₁) Type Defect Cohomology",
    h2_label: "dim H²(Δ₂) Concurrency Race Kernel",
    diag_sub: "100% register unboxing proven without speculative deoptimization guards.",

    arch_badge: "Architecture",
    arch_title: "Built on geometric invariant theory",
    arch_sub: "Why traditional tracing and tier-2 JITs plateau, and how simplicial sheaves bypass them.",
    arch_card1_title: "Cellular Sheaf Type Inference",
    arch_card1_desc: "Standard CFGs are 1D flat graphs. TopoPyc maps dataflow into higher-dimensional cell complexes with vector space stalks and restriction morphisms. Types propagate along algebraic boundaries, eliminating polymorphic guards.",
    arch_card2_title: "Zero-Cost Monomorphism",
    arch_card2_desc: "When dim H¹ = 0, every variable is proven to hold a strictly static type across all execution paths. Values never touch heap PyObject* descriptors.",
    arch_card3_title: "Persistent SHA-256 Caching",
    arch_card3_desc: "Compiled .so extensions are cached in ~/.cache/topopyc by deterministic hash of AST semantics and compiler flags. Subsequent test invocations run in 0.26 seconds.",
    arch_card4_title: "Direct AVX / NEON Code Synthesis",
    arch_card4_desc: "Translates monomorphic kernels into raw C99 compiled via detected native toolchains (GCC / Clang). Auto-vectorization flags (-O3 -march=native -ffast-math) unlock raw CPU compute saturation.",

    usage_badge: "Usage",
    usage_title: "Drop-in decorator. CLI compiler.",
    usage_sub: "No rewrite, no Cython annotations, no external language boundaries.",
    usage_jit_title: "In-Python JIT Kernel",
    usage_cli_title: "Standalone AOT CLI",

    footer_desc: "Topological Sheaf Python Compiler. Created at Tao Science & TopoCompil Engine. Released under the MIT License.",
    footer_head_proj: "Project",
    footer_paper: "Research Paper",
    footer_audit: "UX & DX Audit",
    footer_head_eco: "Ecosystem"
  },

  ru: {
    nav_benchmarks: "Бенчмарки",
    nav_architecture: "Архитектура",
    nav_interactive: "Интерактив",
    nav_docs: "Документация",
    star_github: "GitHub",
    github_repo: "Репозиторий GitHub ↗",

    hero_pill: "Дискретное внешнее исчисление на клеточных пучках",
    hero_title_1: "Синтаксис Python.",
    hero_title_2: "Скорость чистого Си.",
    hero_desc: "TopoPyc переводит Python код в n-мерный симплициальный комплекс с пучками векторных пространств. Мономорфизм строго доказывается занулением 1-х когомологий ker(Δ₁) = 0, компилируя функции в регистровый C99 с AVX-512 векторизацией.",

    metric_cpython: "Быстрее CPython 3.13",
    metric_torch: "Быстрее torch.compile",
    metric_gil: "Накладные расходы GIL и Boxing",
    metric_proof: "Доказанный мономорфизм",

    bench_badge: "Бенчмарки",
    bench_title: "Сравнение с production средами",
    bench_sub: "Реальные deep learning ядра измерены на 10 000 000 непрерывных итераций на одном ядре CPU.",
    gelu_name: "Ядро Fused GELU Poly",
    gelu_iters: "10 000 000 итераций • Скалярный цикл",
    attn_name: "Attention Dot-Product Score Reduction",
    attn_iters: "20 000 000 элементов • Векторный цикл",

    sim_badge: "Интерактивный симулятор",
    sim_title: "Топологическое сжатие в реальном времени",
    sim_sub: "Выберите вычислительное ядро, чтобы увидеть разбор AST узлов в симплексы, кограничные цепи и инварианты Ходжа.",
    tab_mandelbrot_title: "Фрактал Мандельброта",
    tab_mandelbrot_desc: "Динамические ветвления и цикл выхода",
    tab_attn_title: "Attention Reduction",
    tab_attn_desc: "Векторное скалярное произведение",
    tab_gelu_title: "Fused GELU Poly",
    tab_gelu_desc: "Полиномиальная активация",

    diag_title: "Пучковые инварианты Ходжа",
    diag_status: "Доказано",
    v_label: "0-Симплексы (Вершины X⁽⁰⁾)",
    e_label: "1-Симплексы (Ребра Def-Use X⁽¹⁾)",
    t_label: "2-Симплексы (Грани операторов X⁽²⁾)",
    h0_label: "dim H⁰(Δ₀) Связность слоя",
    h1_label: "dim H¹(Δ₁) Когомологии дефекта типов",
    h2_label: "dim H²(Δ₂) Ядро гонок параллелизма",
    diag_sub: "100% регистровый unboxing гарантирован без спекулятивных проверок деоптимизации.",

    arch_badge: "Архитектура",
    arch_title: "Построено на теории инвариантов",
    arch_sub: "Почему классические JIT упираются в потолок, и как пучковые комплексы решают проблему.",
    arch_card1_title: "Пучковый вывод типов",
    arch_card1_desc: "Стандартный CFG — плоский 1D граф. TopoPyc строит многомерный комплекс с векторными стеблями и отображениями рестрикций. Типы распространяются вдоль алгебраических границ без guard-ов.",
    arch_card2_title: "Zero-Cost Мономорфизм",
    arch_card2_desc: "При dim H¹ = 0 каждая переменная строго мономорфна по всем путям выполнения. Значения никогда не упаковываются в кучу PyObject*.",
    arch_card3_title: "Персистентный SHA-256 кэш",
    arch_card3_desc: "Скомпилированные .so модули кэшируются в ~/.cache/topopyc по детерминированному хэшу AST и флагов компиляции. Повторные вызовы занимают 0.26 секунды.",
    arch_card4_title: "Прямой синтез AVX / NEON кода",
    arch_card4_desc: "Трансляция мономорфных ядер в чистый C99 через системный компилятор (GCC / Clang). Флаги (-O3 -march=native -ffast-math) раскрывают предельную пропускную способность CPU.",

    usage_badge: "Использование",
    usage_title: "Один декоратор. Консольный компилятор.",
    usage_sub: "Без переписывания кода, без аннотаций Cython и сторонних языков.",
    usage_jit_title: "JIT ядро прямо в Python",
    usage_cli_title: "Автономный AOT CLI",

    footer_desc: "Топологический пучковый компилятор Python. Разработано в Tao Science & TopoCompil Engine. Лицензия MIT.",
    footer_head_proj: "Проект",
    footer_paper: "Исследовательская статья",
    footer_audit: "Аудит UX и DX",
    footer_head_eco: "Экосистема"
  }
};

let currentLang = localStorage.getItem('topopyc_lang') || 'en';

function applyLanguage(lang) {
  currentLang = lang;
  localStorage.setItem('topopyc_lang', lang);
  document.documentElement.lang = lang;

  const langLabel = document.getElementById('langLabel');
  if (langLabel) {
    langLabel.textContent = lang.toUpperCase();
  }

  const dict = TRANSLATIONS[lang] || TRANSLATIONS.en;
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    if (dict[key]) {
      el.textContent = dict[key];
    }
  });
}

const langToggle = document.getElementById('langToggle');
if (langToggle) {
  langToggle.addEventListener('click', () => {
    const nextLang = currentLang === 'en' ? 'ru' : 'en';
    applyLanguage(nextLang);
  });
}

// ============================================================================
// 2. Multi-Theme Engine (Dark / Light / OLED)
// ============================================================================
const THEMES = ['dark', 'light', 'oled'];
let currentTheme = localStorage.getItem('topopyc_theme') || 'dark';

function applyTheme(theme) {
  currentTheme = theme;
  localStorage.setItem('topopyc_theme', theme);
  document.documentElement.setAttribute('data-theme', theme);

  const themeIcon = document.getElementById('themeIcon');
  if (themeIcon) {
    if (theme === 'light') {
      // Sun icon
      themeIcon.innerHTML = `
        <circle cx="12" cy="12" r="5"></circle>
        <line x1="12" y1="1" x2="12" y2="3"></line>
        <line x1="12" y1="21" x2="12" y2="23"></line>
        <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
        <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
        <line x1="1" y1="12" x2="3" y2="12"></line>
        <line x1="21" y1="12" x2="23" y2="12"></line>
        <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
        <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
      `;
    } else if (theme === 'oled') {
      // Pure black diamond icon
      themeIcon.innerHTML = `
        <polygon points="12 2 22 12 12 22 2 12"></polygon>
      `;
    } else {
      // Moon icon (dark)
      themeIcon.innerHTML = `
        <path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"></path>
      `;
    }
  }
}

const themeToggle = document.getElementById('themeToggle');
if (themeToggle) {
  themeToggle.addEventListener('click', () => {
    const nextIdx = (THEMES.indexOf(currentTheme) + 1) % THEMES.length;
    applyTheme(THEMES[nextIdx]);
  });
}

// ============================================================================
// 3. Mobile Drawer Menu
// ============================================================================
const mobileMenuBtn = document.getElementById('mobileMenuBtn');
const mobileDrawer = document.getElementById('mobileDrawer');

if (mobileMenuBtn && mobileDrawer) {
  mobileMenuBtn.addEventListener('click', () => {
    mobileDrawer.classList.toggle('open');
  });

  // Close drawer when link clicked
  mobileDrawer.querySelectorAll('.drawer-item').forEach(item => {
    item.addEventListener('click', () => {
      mobileDrawer.classList.remove('open');
    });
  });
}

// ============================================================================
// 4. One-Click Install Command Copy (Bun Style)
// ============================================================================
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

// ============================================================================
// 5. Interactive Kernel Switcher Tabs
// ============================================================================
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

// 6. Smooth Scroll
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function(e) {
    const target = document.querySelector(this.getAttribute('href'));
    if (target) {
      e.preventDefault();
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  });
});

// Initialize on page load
applyTheme(currentTheme);
applyLanguage(currentLang);
