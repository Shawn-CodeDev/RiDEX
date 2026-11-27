export function createImageViewer(dialog, methods) {
  const title = dialog.querySelector('#dialog-title');
  const prompt = dialog.querySelector('#dialog-prompt');
  const media = dialog.querySelector('.dialog-media');
  const images = dialog.querySelector('.dialog-images');
  const status = dialog.querySelector('.dialog-status');
  const motion = matchMedia('(prefers-reduced-motion: no-preference)');
  let row, displayed, requested, previousFocus, generation = 0;

  const buttons = methods.map(method => {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = method;
    button.dataset.method = method;
    button.setAttribute('aria-pressed', 'false');
    button.addEventListener('click', () => select(method));
    return button;
  });
  dialog.querySelector('.dialog-methods').replaceChildren(...buttons);

  function updateState(state, message = '') {
    dialog.dataset.state = state;
    media.setAttribute('aria-busy', String(state === 'loading'));
    status.textContent = message;
    status.hidden = !message;
    buttons.forEach(button => {
      button.setAttribute('aria-pressed', String(button.dataset.method === displayed));
      button.toggleAttribute('data-pending', state === 'loading' && button.dataset.method === requested);
    });
  }

  async function select(method) {
    if (!row || !dialog.open || method === requested && dialog.dataset.state !== 'error') return;
    const revision = ++generation;
    requested = method;
    if (method === displayed) {
      updateState('ready');
      return;
    }
    updateState('loading', `Loading ${method}...`);
    const incoming = new Image(row.width, row.height);
    incoming.decoding = 'async';
    incoming.alt = `${method}: ${row.prompt}`;
    incoming.src = row.images[method];
    try {
      await incoming.decode();
    } catch {
      if (revision === generation && dialog.open) updateState('error', `Could not load ${method}. Select it again to retry.`);
      return;
    }
    // Only the most recent selection can change the visible model.
    if (revision !== generation || !dialog.open) return;
    const outgoing = images.querySelector('#dialog-image');
    if (outgoing && motion.matches) {
      outgoing.removeAttribute('id');
      outgoing.setAttribute('aria-hidden', 'true');
      images.replaceChildren(outgoing, incoming);
      const transition = incoming.animate([{ opacity: 0 }, { opacity: 1 }], {
        duration: 320, easing: 'cubic-bezier(.22, .75, .28, 1)',
      });
      transition.onfinish = () => outgoing.remove();
      transition.oncancel = () => outgoing.remove();
    } else {
      images.replaceChildren(incoming);
    }
    incoming.id = 'dialog-image';
    displayed = method;
    dialog.dataset.method = method;
    updateState('ready');
  }

  dialog.querySelector('#close-dialog').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => {
    if (event.target !== dialog) return;
    const box = dialog.getBoundingClientRect();
    if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) dialog.close();
  });
  dialog.addEventListener('close', () => {
    if (dialog.open) return;
    ++generation;
    row = null;
    images.replaceChildren();
    previousFocus?.focus({ preventScroll: true });
  });
  dialog.addEventListener('keydown', event => {
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
    const step = event.key === 'ArrowRight' ? 1 : -1;
    const index = (methods.indexOf(requested) + step + methods.length) % methods.length;
    event.preventDefault();
    buttons[index].focus({ preventScroll: true });
    select(methods[index]);
  });

  return (example, method, origin) => {
    if (dialog.open) return;
    row = example;
    displayed = requested = null;
    delete dialog.dataset.method;
    images.replaceChildren();
    title.textContent = row.title;
    prompt.textContent = row.prompt;
    dialog.style.setProperty('--media-ratio', row.width / row.height);
    previousFocus = document.activeElement;
    dialog.showModal();
    if (motion.matches) {
      const target = media.getBoundingClientRect();
      const from = origin && target.width
        ? {
            x: origin.left + origin.width / 2 - target.left - target.width / 2,
            y: origin.top + origin.height / 2 - target.top - target.height / 2,
            scale: Math.max(0.28, Math.min(0.82, origin.width / target.width)),
          }
        : { x: 0, y: 18, scale: 0.96 };
      dialog.animate([
        { opacity: 0, transform: `translate(${from.x}px, ${from.y}px) scale(${from.scale})` },
        { opacity: 1, transform: 'none' },
      ], { duration: 480, easing: 'cubic-bezier(.18, .85, .26, 1)' });
    }
    select(method);
  };
}
