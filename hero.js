/* The image mask moves only while needed; image pixels are never re-encoded. */
(() => {
  'use strict';
  const hero = document.getElementById('top');
  const header = document.getElementById('site-header');
  const buttons = [...document.querySelectorAll('[data-hero-view]')];
  const status = document.getElementById('hero-view-status');
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  let target = {x: 0, y: 0}, current = {x: 0, y: 0};
  let frame = 0, visible = true, pointer = null;

  function paint() {
    hero.style.setProperty('--spot-x', `${current.x.toFixed(2)}px`);
    hero.style.setProperty('--spot-y', `${current.y.toFixed(2)}px`);
  }
  function stop() {
    if (frame) cancelAnimationFrame(frame);
    frame = 0;
  }
  function tick() {
    frame = 0;
    if (!visible || document.hidden || hero.dataset.view !== 'spotlight') return;
    current.x += (target.x - current.x) * 0.1;
    current.y += (target.y - current.y) * 0.1;
    const settled = Math.hypot(target.x - current.x, target.y - current.y) < 0.2;
    if (settled) current = {...target};
    paint();
    if (!settled) frame = requestAnimationFrame(tick);
  }
  function move() {
    if (hero.dataset.view !== 'spotlight' || !visible || document.hidden) return;
    if (reducedMotion.matches) {
      stop(); current = {...target}; paint();
    } else if (!frame) frame = requestAnimationFrame(tick);
  }
  function center(animate = false) {
    const rect = hero.getBoundingClientRect();
    target = {x: rect.width * 0.68, y: rect.height * 0.68};
    if (animate) move();
    else { stop(); current = {...target}; paint(); }
  }
  function selectView(view) {
    stop();
    hero.dataset.view = view;
    buttons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.heroView === view)));
    status.textContent = {
      base: 'Base image: Stable Diffusion 3.5 Medium.',
      spotlight: 'Spotlight: RiDEx is revealed over the base image.',
      ridex: 'RiDEx image.'
    }[view];
    if (view === 'spotlight') center();
  }
  buttons.forEach(button => {
    button.addEventListener('click', () => selectView(button.dataset.heroView));
    button.addEventListener('keydown', event => {
      const available = buttons.filter(item => !item.disabled);
      const index = available.indexOf(button);
      let next;
      if (event.key === 'ArrowRight') next = (index + 1) % available.length;
      if (event.key === 'ArrowLeft') next = (index + available.length - 1) % available.length;
      if (event.key === 'Home') next = 0;
      if (event.key === 'End') next = available.length - 1;
      if (next !== undefined) {
        event.preventDefault(); available[next].focus(); selectView(available[next].dataset.heroView);
      }
    });
  });
  function trackPointer() {
    if (!pointer) return;
    const rect = hero.getBoundingClientRect();
    if (pointer.x < rect.left || pointer.x > rect.right || pointer.y < rect.top || pointer.y > rect.bottom) {
      pointer = null; center(); return;
    }
    target = {x: pointer.x - rect.left, y: pointer.y - rect.top};
    move();
  }
  hero.addEventListener('pointermove', event => {
    if (event.pointerType === 'touch') return;
    pointer = {x: event.clientX, y: event.clientY};
    trackPointer();
  }, {passive: true});
  hero.addEventListener('pointerleave', () => { pointer = null; center(true); });
  const visibility = new IntersectionObserver(([entry]) => {
    visible = entry.isIntersecting;
    if (!visible) stop();
    else center();
  });
  visibility.observe(hero);
  document.addEventListener('visibilitychange', () => document.hidden ? stop() : center());
  reducedMotion.addEventListener('change', () => center());
  addEventListener('pagehide', stop);
  addEventListener('pageshow', () => center());

  const menu = document.getElementById('mobile-nav');
  const toggle = document.getElementById('menu-toggle');
  function closeMenu(restoreFocus = false) {
    menu.hidden = true;
    toggle.setAttribute('aria-expanded', 'false');
    toggle.setAttribute('aria-label', 'Open navigation');
    toggle.title = 'Open navigation';
    if (restoreFocus) toggle.focus();
  }
  toggle.addEventListener('click', () => {
    const open = menu.hidden;
    menu.hidden = !open;
    toggle.setAttribute('aria-expanded', String(open));
    toggle.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
    toggle.title = open ? 'Close navigation' : 'Open navigation';
  });
  menu.addEventListener('click', event => {
    const link = event.target.closest('a');
    if (!link) return;
    closeMenu();
    if (link.hash && link.origin === location.origin) {
      const destination = document.querySelector(link.hash);
      destination.setAttribute('tabindex', '-1');
      destination.focus({preventScroll: true});
    }
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && !menu.hidden) closeMenu(true);
  });
  document.addEventListener('pointerdown', event => {
    if (!menu.hidden && !header.contains(event.target)) closeMenu();
  });
  function syncHeader() {
    header.classList.toggle('is-scrolled', hero.getBoundingClientRect().bottom <= 101);
  }
  addEventListener('scroll', () => {
    syncHeader(); trackPointer();
    if (!menu.hidden) closeMenu();
  }, {passive: true});
  addEventListener('resize', () => {
    center(); syncHeader(); trackPointer();
    if (innerWidth >= 768 && !menu.hidden) {
      const activeMenu = menu.contains(document.activeElement);
      closeMenu();
      if (activeMenu) header.querySelector('.desktop-nav a').focus({preventScroll: true});
    }
  }, {passive: true});
  const navLinks = [...header.querySelectorAll('.desktop-nav a')];
  const sectionObserver = new IntersectionObserver(entries => {
    entries.filter(entry => entry.isIntersecting).forEach(entry => {
      navLinks.forEach(link => {
        if (link.hash === `#${entry.target.id}`) link.setAttribute('aria-current', 'location');
        else link.removeAttribute('aria-current');
      });
    });
  }, {rootMargin: '-12% 0px -72% 0px'});
  document.querySelectorAll('main > section').forEach(section => sectionObserver.observe(section));

  function imageFailed(which) {
    buttons.forEach(button => { if (button.dataset.heroView === 'spotlight' || button.dataset.heroView === which) button.disabled = true; });
    const remaining = buttons.find(button => !button.disabled);
    if (remaining) selectView(remaining.dataset.heroView);
    else hero.dataset.view = 'base';
    status.textContent = 'One or more comparison images could not load.';
  }
  hero.querySelectorAll('.hero-image').forEach((image, index) => {
    const which = index === 0 ? 'base' : 'ridex';
    image.addEventListener('error', () => imageFailed(which), {once: true});
    if (image.complete && image.naturalWidth === 0) imageFailed(which);
  });
  center(); syncHeader();
})();
