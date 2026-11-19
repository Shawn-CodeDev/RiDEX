(() => {
  'use strict';
  const header = document.getElementById('site-header');
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
    header.classList.toggle('is-scrolled', window.scrollY > 48);
  }
  window.addEventListener('scroll', () => {
    syncHeader();
    if (!menu.hidden) closeMenu();
  }, {passive: true});
  window.addEventListener('resize', () => {
    syncHeader();
    if (window.innerWidth >= 768 && !menu.hidden) {
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
  syncHeader();
})();
