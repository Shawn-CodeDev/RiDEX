'use strict';
const examples = window.RIDEX_EXAMPLES || [];
const methods = ['Base', 'DiffusionNFT', 'RiDEx'];
const tabs = [...document.querySelectorAll('[data-task]')];
const grid = document.getElementById('comparison-grid');
const panel = document.getElementById('comparison-panel');
const dialog = document.getElementById('image-dialog');
let task = 'geneval', index = 0, lastFocus;
function currentRows() { return examples.filter(row => row.reward === task); }
function currentExample() { return currentRows()[index]; }
function showLarge(method) {
  const example = currentExample();
  document.getElementById('dialog-title').textContent = method === 'Base' ? 'Base · SD3.5 Medium' : method;
  document.getElementById('dialog-prompt').textContent = example.prompt;
  const image = document.getElementById('dialog-image');
  image.src = example.images[method]; image.alt = `${method}: ${example.prompt}`;
  image.width = example.resolution; image.height = example.resolution;
  document.querySelectorAll('.dialog-methods button').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.method === method)));
  if (!dialog.open) { lastFocus = document.activeElement; dialog.showModal(); document.body.style.overflow = 'hidden'; }
}
methods.forEach(method => {
  const button = document.createElement('button'); button.type = 'button'; button.textContent = method; button.dataset.method = method;
  button.addEventListener('click', () => showLarge(method)); document.querySelector('.dialog-methods').append(button);
});
function render() {
  const rows = currentRows(), example = rows[index]; if (!example) return;
  document.getElementById('example-prompt').textContent = example.prompt;
  document.getElementById('example-meta').textContent = `${example.resolution} × ${example.resolution} · Seed ${example.seed}`;
  document.getElementById('example-count').textContent = `${String(index + 1).padStart(2, '0')} / ${String(rows.length).padStart(2, '0')}`;
  grid.replaceChildren();
  methods.forEach(method => {
    const figure = document.createElement('figure'), label = document.createElement('figcaption');
    label.className = 'method-label'; label.textContent = method === 'Base' ? 'Base · SD3.5' : method;
    if (method === 'RiDEx') { const badge = document.createElement('small'); badge.textContent = 'Ours'; label.append(badge); }
    const button = document.createElement('button'); button.className = 'image-button'; button.type = 'button';
    button.setAttribute('aria-label', `Enlarge ${method}: ${example.prompt}`); button.title = `View ${method} at full resolution`;
    const image = document.createElement('img'); image.src = example.images[method]; image.alt = `${method}: ${example.prompt}`;
    image.width = example.resolution; image.height = example.resolution; image.decoding = 'async';
    const icon = document.createElement('span'); icon.className = 'zoom-icon'; icon.textContent = '↗'; icon.setAttribute('aria-hidden', 'true');
    button.append(image, icon); button.addEventListener('click', () => showLarge(method)); figure.append(label, button); grid.append(figure);
  });
  const thumbnails = document.getElementById('example-thumbnails'); thumbnails.replaceChildren();
  rows.forEach((row, number) => {
    const button = document.createElement('button'); button.type = 'button'; button.setAttribute('aria-pressed', String(index === number)); button.setAttribute('aria-label', `Show example: ${row.title}`);
    const image = document.createElement('img'); image.src = row.images.RiDEx; image.alt = ''; image.width = 48; image.height = 48;
    const label = document.createElement('span'); label.textContent = row.title; button.append(image, label);
    button.addEventListener('click', () => { index = number; render(); thumbnails.children[number].focus({preventScroll: true}); }); thumbnails.append(button);
  });
}
function selectTask(button) {
  task = button.dataset.task; index = 0;
  tabs.forEach(tab => { const selected = tab === button; tab.setAttribute('aria-selected', String(selected)); tab.tabIndex = selected ? 0 : -1; });
  panel.setAttribute('aria-labelledby', button.id); render();
}
tabs.forEach((button, number) => {
  button.addEventListener('click', () => selectTask(button));
  button.addEventListener('keydown', event => {
    let next;
    if (event.key === 'ArrowRight') next = (number + 1) % tabs.length;
    if (event.key === 'ArrowLeft') next = (number + tabs.length - 1) % tabs.length;
    if (event.key === 'Home') next = 0;
    if (event.key === 'End') next = tabs.length - 1;
    if (next !== undefined) { event.preventDefault(); tabs[next].focus(); selectTask(tabs[next]); }
  });
});
document.getElementById('previous-example').addEventListener('click', () => { index = (index + currentRows().length - 1) % currentRows().length; render(); });
document.getElementById('next-example').addEventListener('click', () => { index = (index + 1) % currentRows().length; render(); });
document.getElementById('close-dialog').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => { const box = dialog.getBoundingClientRect(); if (event.target === dialog && (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom)) dialog.close(); });
dialog.addEventListener('close', () => { document.body.style.overflow = ''; lastFocus?.focus(); });
document.getElementById('copy-citation').addEventListener('click', async () => {
  const status = document.getElementById('copy-status');
  try { await navigator.clipboard.writeText(document.getElementById('bibtex').textContent); status.textContent = 'BibTeX copied.'; document.getElementById('copy-citation').textContent = 'Copied ✓'; setTimeout(() => { document.getElementById('copy-citation').textContent = 'Copy BibTeX ⧉'; }, 2200); }
  catch { const range = document.createRange(); range.selectNodeContents(document.getElementById('bibtex')); const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range); status.textContent = 'Citation selected. Use your browser copy command.'; }
});
render();
