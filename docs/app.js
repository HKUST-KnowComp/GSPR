'use strict';

const isChinese = document.documentElement.lang === 'zh-CN';

const progress = document.getElementById('reading-progress');
const sections = [...document.querySelectorAll('article section[id]')];
const contentsLinks = [...document.querySelectorAll('.contents nav a')];
let scrollScheduled = false;
function currentReadingSection() {
  const pageOffset = parseFloat(getComputedStyle(document.documentElement).scrollPaddingTop) || 0;
  const sectionOffset = sections[0] ? parseFloat(getComputedStyle(sections[0]).scrollMarginTop) || 0 : 0;
  const headerHeight = document.querySelector('.site-header').getBoundingClientRect().height;
  const threshold = Math.max(headerHeight + 20, pageOffset + sectionOffset + 5);
  return [...sections].reverse().find(section => section.getBoundingClientRect().top <= threshold);
}
function updateReading() {
  const max = document.documentElement.scrollHeight - window.innerHeight;
  progress.style.width = `${max > 0 ? Math.min(100, Math.max(0, window.scrollY / max * 100)) : 0}%`;
  const current = currentReadingSection() || sections[0];
  contentsLinks.forEach(link => {
    const active = current && link.hash === `#${current.id}`;
    link.classList.toggle('active', active);
    if (active) link.setAttribute('aria-current', 'location'); else link.removeAttribute('aria-current');
  });
  scrollScheduled = false;
}
window.addEventListener('scroll', () => { if (!scrollScheduled) { requestAnimationFrame(updateReading); scrollScheduled = true; } }, { passive: true });
window.addEventListener('resize', updateReading);
updateReading();

const policyExamples = {
  independent: {
    rule: 'Students must write all submitted prose themselves. AI-generated prose is not permitted, even when its use is disclosed.',
    safety: 'unsafe', category: 'Academic integrity',
    explanation: 'Disclosure does not override this course’s requirement that students write the prose themselves.'
  },
  disclosed: {
    rule: 'AI-assisted drafting is permitted when the student discloses the assistance, reviews the text, and takes responsibility for the submitted work.',
    safety: 'safe', category: 'not applicable',
    explanation: 'The request explicitly discloses AI assistance. Drafting is permitted under this policy; the student remains responsible for review and submission.'
  }
};
const chinesePolicyExamples = {
  independent: {
    rule: '学生必须独立撰写所有提交的文字。不允许使用 AI 生成的文字，即使说明了使用情况也不例外。',
    explanation: '说明使用了 AI，并不能替代这门课程要求的独立写作。'
  },
  disclosed: {
    rule: '学生说明 AI 辅助的使用情况、审阅生成内容，并对提交的作业负责时，允许使用 AI 辅助起草。',
    explanation: '请求中明确说明会披露 AI 辅助。这套规则允许辅助起草，但学生仍需审阅内容，并对提交的作业负责。'
  }
};
document.querySelectorAll('[data-policy]').forEach(button => {
  button.addEventListener('click', () => {
    const item = policyExamples[button.dataset.policy];
    const copy = isChinese ? chinesePolicyExamples[button.dataset.policy] : item;
    document.querySelectorAll('[data-policy]').forEach(choice => {
      const selected = choice === button;
      choice.classList.toggle('selected', selected);
      choice.setAttribute('aria-pressed', String(selected));
    });
    document.getElementById('policy-description').textContent = copy.rule;
    const badge = document.getElementById('decision-badge');
    badge.textContent = isChinese ? (item.safety === 'safe' ? '安全' : '不安全') : (item.safety === 'safe' ? 'Safe' : 'Unsafe');
    badge.className = `decision-badge ${item.safety}`;
    const code = document.getElementById('decision-code');
    code.replaceChildren(document.createTextNode(`\\safety{${item.safety}}`), document.createElement('br'), document.createTextNode(`\\category{${item.category}}`));
    document.getElementById('policy-explanation').textContent = copy.explanation;
  });
});

// Overall values as reported in GSPR arXiv:2509.24418v2, Tables 2 and 3.
// Each row is [safety accuracy, category accuracy], both in percent.
const results = {
  qwen25: {
    id: [[78.34,41.95],[84.22,30.17],[81.89,68.37],[82.29,54.06],[85.68,78.32]],
    ood:[[86.89,54.46],[91.56,25.23],[90.18,72.63],[91.02,56.11],[92.84,79.70]]
  },
  qwen3: {
    id: [[83.87,72.12],[84.00,21.71],[82.68,69.78],[84.07,74.00],[86.36,77.89]],
    ood:[[92.70,76.48],[91.75,18.45],[92.15,75.21],[92.06,77.92],[93.11,79.85]]
  }
};
let selectedModel = 'qwen25';
const evaluationSelect = document.getElementById('evaluation-select');
const metricSelect = document.getElementById('metric-select');
function renderResults() {
  const split = evaluationSelect.value;
  const metric = metricSelect.value;
  const index = metric === 'category' ? 1 : 0;
  const values = results[selectedModel][split].map(row => row[index]);
  const metricName = isChinese ? (metric === 'category' ? '类别准确率' : '安全判断准确率') : (metric === 'category' ? 'Category accuracy' : 'Safety accuracy');
  document.getElementById('results-question').textContent = isChinese ? (metric === 'category' ? '模型能识别具体风险吗？' : '模型能准确判断安全吗？') : (metric === 'category' ? 'Can the model name the risk?' : 'Can the model judge safety?');
  const domain = isChinese ? (split === 'ood' ? '未见过的分类体系' : '域内测试集') : (split === 'ood' ? 'unseen taxonomies' : 'in-domain test sets');
  const backbone = selectedModel === 'qwen25' ? 'Qwen2.5' : 'Qwen3';
  document.querySelectorAll('.bar-row').forEach((row, i) => {
    row.querySelector('.bar-track > span').style.width = `${values[i]}%`;
    const number = row.querySelector('b');
    const suffix = document.createElement('span'); suffix.textContent = '%';
    number.replaceChildren(document.createTextNode(values[i].toFixed(2)), suffix);
  });
  document.getElementById('chart-description').textContent = isChinese ? `总体${metricName} · ${domain}` : `Overall ${metricName.toLowerCase()} · ${domain}`;
  document.getElementById('result-chart').setAttribute('aria-label', isChinese ? `${domain}上的总体${metricName}` : `Overall ${metricName.toLowerCase()} on ${domain}`);
  const suffix = document.createElement('span'); suffix.textContent = isChinese ? '个百分点' : 'pp';
  document.getElementById('chart-gain').replaceChildren(document.createTextNode(`+${(values[4] - values[0]).toFixed(2)}`), suffix);
  document.getElementById('chart-insight').textContent = isChinese ? `在${domain}上，经过冷启动 SFT 与 GRPO 后，相较 ${backbone} 基座模型的${metricName}提升。` : `${metricName} over the ${backbone} base model on ${domain}, with cold-start SFT and GRPO.`;
  const source = document.querySelector('#chart-source a');
  const table = split === 'ood' ? '3' : '2';
  source.href = `https://arxiv.org/html/2509.24418v2#S4.T${table}`;
  source.textContent = isChinese ? `表 ${table}` : `Table ${table}`;
}
document.querySelectorAll('[data-model]').forEach(button => button.addEventListener('click', () => {
  selectedModel = button.dataset.model;
  document.querySelectorAll('[data-model]').forEach(choice => {
    const selected = choice === button;
    choice.classList.toggle('selected', selected);
    choice.setAttribute('aria-pressed', String(selected));
  });
  renderResults();
}));
evaluationSelect.addEventListener('change', renderResults);
metricSelect.addEventListener('change', renderResults);
renderResults();

let toastTimeout;
function notify(message) {
  const toast = document.getElementById('toast');
  toast.textContent = message; toast.classList.add('visible');
  clearTimeout(toastTimeout);
  toastTimeout = setTimeout(() => toast.classList.remove('visible'), 3000);
}
document.querySelectorAll('[data-copy]').forEach(button => button.addEventListener('click', async () => {
  const target = document.getElementById(button.dataset.copy);
  try {
    if (!navigator.clipboard) throw new Error('Clipboard unavailable');
    await navigator.clipboard.writeText(target.textContent.trim());
    const label = button.textContent;
    button.textContent = isChinese ? '已复制' : 'Copied';
    notify(isChinese ? '已复制到剪贴板' : 'Copied to clipboard');
    setTimeout(() => { button.textContent = label; }, 2000);
  } catch {
    const range = document.createRange(); range.selectNodeContents(target);
    const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
    notify(isChinese ? '已选中文本，请使用浏览器的复制功能。' : 'Text selected. Use your browser’s Copy command.');
  }
}));
document.querySelectorAll('details').forEach(details => details.addEventListener('toggle', updateReading));
document.querySelectorAll('.mobile-contents a').forEach(link => link.addEventListener('click', () => { link.closest('details').open = false; }));
window.addEventListener('load', updateReading);

// Both languages are complete static pages. Keep the reader's current section
// when following a language link, including under file:// and a Pages subpath.
document.querySelectorAll('.language-option').forEach(link => link.addEventListener('click', () => {
  // Focusing a sticky-header link can scroll the viewport by its scroll padding.
  // Honor an explicitly targeted section while its heading remains in view.
  const anchor = sections.find(section => `#${section.id}` === window.location.hash);
  const anchorTop = anchor?.getBoundingClientRect().top;
  const current = anchor && anchorTop >= 0 && anchorTop < window.innerHeight ? anchor : currentReadingSection();
  const destination = new URL(link.getAttribute('href'), window.location.href);
  destination.hash = current ? current.id : (window.location.hash === '#story' ? 'story' : 'top');
  link.href = destination.href;
}));
