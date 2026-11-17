'use strict';
(() => {
  const header = document.getElementById('site-header');
  const menuButton = document.getElementById('menu-button');
  const mobileNav = document.getElementById('mobile-nav');
  let queued = false;
  function updateHeader() { header.classList.toggle('scrolled', window.scrollY > 28); queued = false; }
  updateHeader();
  window.addEventListener('scroll', () => { if (!queued) { window.requestAnimationFrame(updateHeader); queued = true; } }, {passive: true});
  function closeMenu() { document.body.classList.remove('menu-open'); mobileNav.hidden = true; menuButton.setAttribute('aria-expanded', 'false'); menuButton.setAttribute('aria-label','Open navigation'); }
  menuButton.addEventListener('click', () => {
    const isOpen = menuButton.getAttribute('aria-expanded') === 'true';
    if (isOpen) closeMenu(); else { mobileNav.hidden = false; document.body.classList.add('menu-open'); menuButton.setAttribute('aria-expanded','true'); menuButton.setAttribute('aria-label','Close navigation'); }
  });
  mobileNav.querySelectorAll('a').forEach(a => a.addEventListener('click', closeMenu));
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeMenu(); });
  window.matchMedia('(min-width:801px)').addEventListener('change', e => { if(e.matches) closeMenu(); });

  const navLinks = [...document.querySelectorAll('.desktop-nav a')];
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(entries => { entries.forEach(entry => { if (entry.isIntersecting) navLinks.forEach(a => a.classList.toggle('active', a.getAttribute('href') === '#'+entry.target.id)); }); }, {rootMargin:'-18% 0px -55% 0px'});
    ['method','results','gallery','paper'].forEach(id => observer.observe(document.getElementById(id)));
  }
  const motionButton = document.getElementById('motion-toggle');
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  function setMotion(paused) { document.body.classList.toggle('motion-paused',paused); motionButton.setAttribute('aria-pressed',String(paused)); motionButton.innerHTML = paused ? 'Resume motion <span aria-hidden="true">▷</span>' : 'Pause motion <span aria-hidden="true">Ⅱ</span>'; }
  setMotion(reducedMotion.matches);
  motionButton.addEventListener('click', () => setMotion(!document.body.classList.contains('motion-paused')));

  // Exact target budgets from the manuscript. Speedup uses the paper's rounded report.
  const tasks = {
    hps: {label:'HPSv2',variant:'RiDEx-Grad',nft:114.63,ridex:7.64,speed:'15',target:'34.2538'},
    pick:{label:'PickScore',variant:'RiDEx-Grad',nft:222.52,ridex:7.67,speed:'29',target:'22.6407'},
    gen: {label:'GenEval',variant:'RiDEx-CFG',nft:31.39,ridex:12.07,speed:'2.6',target:'0.9501'},
    ocr: {label:'OCR',variant:'RiDEx-CFG',nft:16.44,ridex:8.22,speed:'2.0',target:'0.9285'}
  };
  const tabs = [...document.querySelectorAll('[data-task]')];
  function selectTask(tab, focus = false) {
    const d = tasks[tab.dataset.task]; if (!d) return;
    tabs.forEach(t => { const on = t===tab; t.setAttribute('aria-selected',String(on)); t.tabIndex = on ? 0 : -1; });
    document.getElementById('efficiency-chart').setAttribute('aria-labelledby',tab.id);
    document.getElementById('speed-value').textContent = d.speed;
    document.getElementById('speed-description').textContent = d.label+' · '+d.variant;
    document.getElementById('baseline-hours').innerHTML = d.nft.toFixed(2)+' <small>GPU h</small>';
    document.getElementById('ridex-hours').innerHTML = d.ridex.toFixed(2)+' <small>GPU h</small>';
    document.getElementById('ours-bar').style.width = (100*d.ridex/d.nft)+'%';
    document.getElementById('axis-max').textContent = d.nft.toFixed(2)+' h';
    document.getElementById('target-value').textContent = d.target;
    if (focus) tab.focus();
  }
  tabs.forEach((tab,index) => {
    tab.addEventListener('click', () => selectTask(tab));
    tab.addEventListener('keydown', e => {
      let to = null;
      if(e.key==='ArrowRight') to=(index+1)%tabs.length;
      if(e.key==='ArrowLeft') to=(index-1+tabs.length)%tabs.length;
      if(e.key==='Home') to=0;
      if(e.key==='End') to=tabs.length-1;
      if(to!==null) {e.preventDefault();selectTask(tabs[to],true);}
    });
  });

  const table = document.getElementById('leaderboard');
  document.querySelectorAll('.sort-button').forEach(button => button.addEventListener('click', () => {
    const key = button.dataset.sort;
    const th = button.closest('th');
    const desc = th.getAttribute('aria-sort') !== 'descending';
    table.querySelectorAll('thead th[aria-sort]').forEach(x=>x.setAttribute('aria-sort','none'));
    th.setAttribute('aria-sort', desc?'descending':'ascending');
    const rows = [...table.tBodies[0].rows];
    rows.sort((a,b)=>{
      const av = a.dataset[key] === undefined ? null : Number(a.dataset[key]);
      const bv = b.dataset[key] === undefined ? null : Number(b.dataset[key]);
      if(av===null) return bv===null?0:1;
      if(bv===null) return -1;
      return desc?bv-av:av-bv;
    });
    rows.forEach(row=>table.tBodies[0].appendChild(row));
  }));

  document.querySelectorAll('.comparison-image input').forEach(input => {
    function update() {input.closest('.comparison-image').style.setProperty('--split',input.value+'%');input.setAttribute('aria-valuetext',input.value+'% Base, '+(100-Number(input.value))+'% RiDEx');}
    input.addEventListener('input',update);update();
  });
  const filters = [...document.querySelectorAll('[data-filter]')];
  filters.forEach(button => button.addEventListener('click', () => {
    filters.forEach(b => {b.classList.toggle('active',b===button);b.setAttribute('aria-pressed',String(b===button));});
    document.querySelectorAll('.comparison').forEach(fig=>{fig.hidden=button.dataset.filter!=='all'&&fig.dataset.category!==button.dataset.filter;});
  }));

  const copyButton = document.getElementById('copy-bib');
  copyButton.addEventListener('click', async () => {
    const text = document.getElementById('bibtex').textContent;
    try {
      if(navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(text);
      else {const area=document.createElement('textarea');area.value=text;area.style.position='fixed';area.style.top='-10000px';document.body.appendChild(area);area.select();const ok=document.execCommand('copy');area.remove();if(!ok)throw new Error('Copy unavailable');}
      copyButton.textContent='Copied';document.getElementById('copy-status').textContent='BibTeX citation copied.';
    } catch {copyButton.textContent='Select citation to copy';document.getElementById('copy-status').textContent='Automatic copying was blocked. Select the citation text and copy it.';}
    setTimeout(()=>{copyButton.innerHTML='Copy BibTeX <span aria-hidden="true">⧉</span>';},2400);
  });
})();
