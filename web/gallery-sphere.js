import * as THREE from './vendor/three.module.min.js';
import { OrbitControls } from './vendor/OrbitControls.js';

export function createGallerySphere(stage, examples, onOpen) {
  const select = document.getElementById('sphere-select');
  const open = document.getElementById('sphere-open');
  const toggle = document.getElementById('sphere-toggle');
  const reset = document.getElementById('sphere-reset');
  const status = document.getElementById('sphere-status');
  const tooltip = document.getElementById('sphere-tooltip');
  const title = document.getElementById('sphere-title');
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  let selected = 0;
  select.replaceChildren(...examples.map(row => new Option(row.title, row.id)));
  select.disabled = open.disabled = examples.length === 0;
  function selectItem(index) {
    selected = index;
    select.value = examples[index].id;
    stage.dataset.selectedId = examples[index].id;
    title.textContent = examples[index].title;
  }
  if (!examples.length) {
    status.textContent = 'No images are available.';
    stage.dataset.state = 'empty';
    stage.setAttribute('aria-busy', 'false');
    return;
  }
  selectItem(0);
  open.addEventListener('click', () => onOpen(examples[selected]));
  select.addEventListener('change', () => selectItem(examples.findIndex(row => row.id === select.value)));

  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'low-power' });
  } catch {
    status.textContent = '3D is unavailable in this browser. You can still select and view every image below.';
    stage.dataset.state = 'error';
    stage.setAttribute('aria-busy', 'false');
    return;
  }
  const canvas = renderer.domElement;
  canvas.tabIndex = 0;
  canvas.setAttribute('aria-label', 'Rotate image sphere. Arrow keys rotate; Enter opens the selected image.');
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  stage.prepend(canvas);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 100);
  const controls = new OrbitControls(camera, canvas);
  controls.enableZoom = false;
  controls.enablePan = false;
  controls.enableDamping = true;
  controls.dampingFactor = 0.075;
  controls.rotateSpeed = 0.48;
  controls.autoRotateSpeed = 0.28;
  controls.minPolarAngle = 0.12;
  controls.maxPolarAngle = Math.PI - 0.12;
  controls.touches.ONE = THREE.TOUCH.ROTATE;
  const radius = 3.2;
  const cards = [];
  const borders = [];
  const back = new THREE.Mesh(
    new THREE.SphereGeometry(radius - 0.18, 48, 32),
    new THREE.MeshBasicMaterial({ color: '#101512' }),
  );
  scene.add(back);
  const borderGeometry = new THREE.PlaneGeometry(1, 1);
  const imageGeometry = new THREE.PlaneGeometry(1, 1);
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  // Equal-area placement gives every image its own point on the sphere.
  examples.forEach((row, index) => {
    const y = 1 - 2 * (index + 0.5) / examples.length;
    const ring = Math.sqrt(1 - y * y);
    const azimuth = index * goldenAngle;
    const normal = new THREE.Vector3(Math.cos(azimuth) * ring, y, Math.sin(azimuth) * ring);
    const group = new THREE.Group();
    group.position.copy(normal).multiplyScalar(radius);
    group.lookAt(normal.clone().multiplyScalar(radius + 1));
    const ratio = row.width / row.height;
    const width = Math.sqrt(1.75 * ratio);
    const height = width / ratio;
    const border = new THREE.Mesh(borderGeometry, new THREE.MeshBasicMaterial({ color: '#3b4b40' }));
    border.scale.set(width + 0.035, height + 0.035, 1);
    const image = new THREE.Mesh(imageGeometry, new THREE.MeshBasicMaterial({ color: '#202a24' }));
    image.scale.set(width, height, 1);
    image.position.z = 0.008;
    image.userData.index = index;
    group.add(border, image);
    scene.add(group);
    cards.push(image);
    borders.push(border);
  });

  let visible = false, suspended = false, disposed = false;
  let rotating = !reducedMotion.matches, frame = 0, lastTime = 0, frames = 0;
  let focusTween = null, settleUntil = 0, distance = 12;
  let dragging = false, down = null, moved = 0, activePointer = null, hovered = -1;
  const startDirection = new THREE.Vector3(0.35, 0.16, 1).normalize();
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();

  function updateToggle() {
    toggle.setAttribute('aria-pressed', String(rotating));
    const label = rotating ? 'Pause rotation' : 'Resume rotation';
    toggle.setAttribute('aria-label', label);
    toggle.title = label;
  }
  function requestFrame() {
    if (!frame && visible && !document.hidden && !suspended && !disposed) frame = requestAnimationFrame(draw);
  }
  function pause() {
    rotating = false;
    controls.autoRotate = false;
    updateToggle();
  }
  function draw(time) {
    frame = 0;
    if (!visible || document.hidden || suspended || disposed) return;
    const delta = Math.min((time - (lastTime || time)) / 1000, 0.05);
    lastTime = time;
    controls.autoRotate = rotating && !dragging && !focusTween;
    if (focusTween) {
      const fraction = Math.min((time - focusTween.start) / 650, 1);
      const amount = fraction * fraction * (3 - 2 * fraction);
      const rotation = new THREE.Quaternion().slerpQuaternions(new THREE.Quaternion(), focusTween.rotation, amount);
      camera.position.copy(focusTween.from).applyQuaternion(rotation).setLength(distance);
      if (fraction === 1) focusTween = null;
    }
    controls.update(delta);
    renderer.render(scene, camera);
    stage.dataset.frames = String(++frames);
    if (rotating || dragging || focusTween || time < settleUntil) requestFrame();
  }
  function resize() {
    const width = stage.clientWidth, height = stage.clientHeight;
    if (!width || !height) return;
    camera.aspect = width / height;
    const vertical = THREE.MathUtils.degToRad(camera.fov / 2);
    const horizontal = Math.atan(Math.tan(vertical) * camera.aspect);
    distance = (radius + 0.34) / Math.sin(Math.min(vertical, horizontal)) * 1.06;
    if (camera.position.length() < 2) camera.position.copy(startDirection);
    camera.position.setLength(distance);
    camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
    requestFrame();
  }
  function face(index) {
    pause();
    controls.update();
    const target = cards[index].parent.position.clone().normalize();
    const from = camera.position.clone().normalize();
    if (reducedMotion.matches) camera.position.copy(target).multiplyScalar(distance);
    else focusTween = { from, rotation: new THREE.Quaternion().setFromUnitVectors(from, target), start: performance.now() };
    requestFrame();
  }
  function hit(event) {
    const box = canvas.getBoundingClientRect();
    pointer.set((event.clientX - box.left) / box.width * 2 - 1, -(event.clientY - box.top) / box.height * 2 + 1);
    raycaster.setFromCamera(pointer, camera);
    const hits = raycaster.intersectObjects([...cards, back], false);
    return hits.length && hits[0].object !== back ? hits[0].object.userData.index : -1;
  }
  function highlight(index) {
    if (hovered === index) return;
    if (hovered !== -1) borders[hovered].material.color.set('#3b4b40');
    hovered = index;
    if (index !== -1) borders[index].material.color.set('#d9e5d2');
    requestFrame();
  }
  function stop() {
    if (frame) cancelAnimationFrame(frame);
    frame = 0; lastTime = 0;
  }

  updateToggle();
  toggle.disabled = reset.disabled = false;
  toggle.addEventListener('click', () => { rotating = !rotating; updateToggle(); requestFrame(); });
  reset.addEventListener('click', () => {
    pause(); focusTween = null; controls.reset(); camera.position.copy(startDirection).multiplyScalar(distance); requestFrame();
  });
  select.addEventListener('change', () => face(selected));
  controls.addEventListener('start', () => { pause(); focusTween = null; dragging = true; tooltip.hidden = true; requestFrame(); });
  controls.addEventListener('end', () => { dragging = false; settleUntil = performance.now() + 700; requestFrame(); });
  canvas.addEventListener('pointerdown', event => {
    if (activePointer !== null) { down = null; moved = Infinity; return; }
    if (event.button !== 0) return;
    activePointer = event.pointerId; down = { x: event.clientX, y: event.clientY }; moved = 0;
  });
  canvas.addEventListener('pointermove', event => {
    if (down) moved = Math.max(moved, Math.hypot(event.clientX - down.x, event.clientY - down.y));
    if (dragging || event.pointerType !== 'mouse') return;
    const index = hit(event);
    highlight(index);
    canvas.style.cursor = index === -1 ? 'grab' : 'pointer';
    tooltip.hidden = index === -1;
    if (index === -1) return;
    tooltip.textContent = examples[index].title;
    const box = stage.getBoundingClientRect();
    tooltip.style.left = `${Math.max(8, Math.min(event.clientX - box.left + 16, box.width - tooltip.offsetWidth - 8))}px`;
    tooltip.style.top = `${Math.max(8, Math.min(event.clientY - box.top + 18, box.height - tooltip.offsetHeight - 8))}px`;
  });
  canvas.addEventListener('pointerup', event => {
    if (event.pointerId !== activePointer) return;
    if (down && moved < 7) {
      const index = hit(event);
      if (index !== -1) { selectItem(index); canvas.focus({ preventScroll: true }); onOpen(examples[index]); }
    }
    down = null; activePointer = null;
  });
  canvas.addEventListener('pointercancel', () => { down = null; activePointer = null; dragging = false; });
  canvas.addEventListener('pointerleave', () => { tooltip.hidden = true; highlight(-1); });
  canvas.addEventListener('keydown', event => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault(); pause(); onOpen(examples[selected]); return;
    }
    if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) return;
    event.preventDefault(); pause(); focusTween = null;
    const position = new THREE.Spherical().setFromVector3(camera.position);
    if (event.key === 'ArrowLeft') position.theta -= 0.18;
    if (event.key === 'ArrowRight') position.theta += 0.18;
    if (event.key === 'ArrowUp') position.phi = Math.max(0.12, position.phi - 0.18);
    if (event.key === 'ArrowDown') position.phi = Math.min(Math.PI - 0.12, position.phi + 0.18);
    camera.position.setFromSpherical(position); requestFrame();
  });
  canvas.addEventListener('webglcontextlost', event => {
    event.preventDefault(); suspended = true; stop();
    status.hidden = false; status.textContent = '3D rendering was interrupted. Reload the page to restart it.';
    stage.dataset.state = 'error';
  });
  const visibilityObserver = new IntersectionObserver(entries => {
    visible = entries[0].isIntersecting;
    if (visible) requestFrame(); else stop();
  });
  visibilityObserver.observe(stage);
  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(stage);
  const modalObserver = new MutationObserver(() => {
    suspended = document.getElementById('image-dialog').open;
    if (suspended) stop(); else requestFrame();
  });
  modalObserver.observe(document.getElementById('image-dialog'), { attributes: true, attributeFilter: ['open'] });
  document.addEventListener('visibilitychange', () => document.hidden ? stop() : requestFrame());
  reducedMotion.addEventListener('change', () => { if (reducedMotion.matches) pause(); requestFrame(); });
  resize();
  controls.saveState();

  let nextTexture = 0, loaded = 0, failures = 0;
  const loader = new THREE.TextureLoader();
  async function loadTextures() {
    while (nextTexture < examples.length && !disposed) {
      const index = nextTexture++;
      try {
        const texture = await loader.loadAsync(new URL(`../assets/globe/${examples[index].id}.webp`, import.meta.url).href);
        if (disposed) { texture.dispose(); return; }
        texture.colorSpace = THREE.SRGBColorSpace;
        texture.anisotropy = Math.min(renderer.capabilities.getMaxAnisotropy(), 4);
        cards[index].material.map = texture;
        cards[index].material.color.set('#ffffff');
        cards[index].material.needsUpdate = true;
        stage.dataset.loadedCount = String(++loaded);
        if (!failures) status.textContent = `Loading gallery... ${loaded} / ${examples.length}`;
        requestFrame();
      } catch {
        failures++;
        status.textContent = 'Some images could not load. Please reload the page.';
      }
    }
  }
  Promise.all(Array.from({ length: 4 }, loadTextures)).then(() => {
    if (disposed) return;
    status.hidden = !failures;
    stage.dataset.state = failures ? 'error' : 'ready';
    stage.setAttribute('aria-busy', 'false');
    requestFrame();
  });
  window.addEventListener('pagehide', event => {
    stop();
    if (event.persisted) return;
    disposed = true;
    visibilityObserver.disconnect(); resizeObserver.disconnect(); modalObserver.disconnect();
    controls.dispose();
    for (const card of cards) { card.material.map?.dispose(); card.material.dispose(); }
    for (const border of borders) border.material.dispose();
    imageGeometry.dispose(); borderGeometry.dispose(); back.geometry.dispose(); back.material.dispose(); renderer.dispose();
  });
  window.addEventListener('pageshow', requestFrame);
}
