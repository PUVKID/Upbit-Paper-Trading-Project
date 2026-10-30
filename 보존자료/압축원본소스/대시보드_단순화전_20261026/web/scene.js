'use strict';
(() => {
  const host = document.getElementById('scene-stage');
  const canvas = document.getElementById('prediction-scene');
  const status = document.getElementById('scene-status');
  const counts = document.getElementById('scene-counts');
  const tooltip = document.getElementById('scene-tooltip');
  const coins = ['BTC', 'SOL', 'ETH', 'XRP'];
  const classes = ['하락', '중립', '상승'];
  const classNames = ['down', 'flat', 'up'];
  const colors = [0xe47767, 0x7396a6, 0x08ad91];
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  let motion = !reduced.matches;
  let visible = true;
  let disposed = false;
  let ready = false;
  let renderframes = 0;
  let timer = null;
  let signature = '';
  let rows = coins.map(coin => ({coin, values: null, source: '미기록'}));
  let renderer, scene, camera, group, raycaster, pointer, environment;
  let resourceGeometries = [], resourceMaterials = [], meshes = [];
  let targetView = {x: 0, y: 0};
  let currentView = {x: 0, y: 0};
  let mobile = false;
  const labels = [];
  const format = value => value.toLocaleString('ko-KR');
  const valid = value => typeof value === 'number' && Number.isSafeInteger(value) && value >= 0;

  function parseCounts(report) {
    // 라벨 분포·확률 분위수와 혼합하지 않고 예측 표본 수만 사용합니다.
    for (const kind of ['diagnostics', 'research']) {
      for (const key of ['argmax_counts', 'prediction_distribution', 'predicted_distribution']) {
        const raw = report?.[kind]?.[key];
        const values = Array.isArray(raw) ? raw : raw && ['down', 'flat', 'up'].map(name => raw[name]);
        if (values?.length === 3 && values.every(valid)) return {values, source: `${kind}.${key}`};
      }
    }
    return {values: null, source: '예측 표본 미기록'};
  }

  function renderCounts() {
    counts.replaceChildren();
    for (const row of rows) {
      const item = document.createElement('div');
      item.className = 'count-group';
      const title = document.createElement('b');
      title.textContent = row.coin;
      const source = document.createElement('small');
      source.textContent = row.values ? `후속 평가 예측 ${format(row.values.reduce((a, b) => a + b, 0))}개` : '예측 표본 대기';
      source.title = row.source;
      const values = document.createElement('div');
      values.className = 'count-values';
      classes.forEach((label, index) => {
        const value = document.createElement('span');
        value.className = classNames[index];
        value.textContent = `${label} ${row.values ? format(row.values[index]) : '—'}`;
        values.append(value);
      });
      item.append(title, source, values);
      if (!ready) {
        const bars = document.createElement('div');
        bars.className = 'fallback-bars';
        bars.setAttribute('aria-hidden', 'true');
        const max = Math.max(1, ...(row.values || []));
        (row.values || [0, 0, 0]).forEach((value, index) => {
          const bar = document.createElement('div');
          bar.className = `fallback-bar ${classNames[index]}`;
          const fill = document.createElement('span');
          fill.style.height = `${value / max * 100}%`;
          bar.append(fill);
          bars.append(bar);
        });
        item.append(bars);
      }
      counts.append(item);
    }
  }

  function stopTimer() {
    if (timer !== null) clearTimeout(timer);
    timer = null;
  }

  function active() { return motion && !reduced.matches && !document.hidden && visible && !disposed; }

  function frame() {
    timer = null;
    if (!ready || disposed || document.hidden || !visible) return;
    let changing = false;
    for (const mesh of meshes) {
      const goal = mesh.userData.target;
      const next = active() ? mesh.scale.y + (goal - mesh.scale.y) * .18 : goal;
      mesh.scale.y = Math.abs(goal - next) < .002 ? goal : next;
      mesh.position.y = .11 + mesh.scale.y / 2;
      if (Math.abs(mesh.scale.y - goal) > .002) changing = true;
    }
    for (const key of ['x', 'y']) {
      currentView[key] = active() ? currentView[key] + (targetView[key] - currentView[key]) * .14 : 0;
      if (active() && Math.abs(currentView[key] - targetView[key]) > .0005) changing = true;
    }
    group.rotation.y = currentView.x;
    group.rotation.x = currentView.y;
    renderer.render(scene, camera);
    for (const label of labels) {
      const anchor = meshes[label.index * 3 + 1].position.clone();
      anchor.y = .1;
      anchor.z += .8;
      anchor.applyMatrix4(group.matrixWorld).project(camera);
      label.element.style.left = `${(anchor.x * .5 + .5) * host.clientWidth}px`;
      label.element.style.top = `${(-anchor.y * .5 + .5) * host.clientHeight}px`;
    }
    renderframes++;
    if (changing && active()) schedule();
  }

  function schedule() {
    if (timer === null && ready && !document.hidden && visible && !disposed) timer = setTimeout(frame, Math.ceil(1000 / 24));
  }

  function resize() {
    if (!ready || disposed) return;
    const width = host.clientWidth, height = host.clientHeight;
    if (!width || !height) return;
    mobile = width < 730;
    renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 1.25));
    renderer.setSize(width, height, false);
    const aspect = width / height;
    const viewWidth = mobile ? 8.8 : Math.max(15, 5 * aspect);
    const viewHeight = Math.max(mobile ? 8.3 : 5.1, viewWidth / aspect);
    camera.left = -viewWidth / 2;
    camera.right = viewWidth / 2;
    camera.top = viewHeight / 2;
    camera.bottom = -viewHeight / 2;
    camera.position.set(0, mobile ? 8.4 : 6.4, mobile ? 12 : 13);
    camera.lookAt(0, .9, 0);
    camera.updateProjectionMatrix();
    for (const mesh of meshes) {
      const {coinIndex, classIndex} = mesh.userData;
      mesh.position.x = mobile ? (coinIndex % 2 ? 2 : -2) + (classIndex - 1) * .66 : (coinIndex - 1.5) * viewWidth * .235 + (classIndex - 1) * .9;
      mesh.position.z = mobile ? (coinIndex < 2 ? -2 : 2) : 0;
      mesh.scale.x = mobile ? 1 : 1.6;
      mesh.scale.z = mobile ? 1 : 1.38;
    }
    const platform = group.getObjectByName('platform');
    platform.scale.set(mobile ? 7.6 : viewWidth * .9, 1, mobile ? 7 : 2.8);
    setTargets();
  }

  function setTargets() {
    const max = Math.max(1, ...rows.flatMap(row => row.values || []));
    for (const mesh of meshes) {
      const row = rows[mesh.userData.coinIndex];
      const value = row.values?.[mesh.userData.classIndex];
      mesh.visible = row.values !== null;
      mesh.userData.value = value;
      mesh.userData.target = value === undefined ? .001 : Math.max(.008, value / max * (mobile ? 2.45 : 3.2));
    }
    schedule();
  }

  function cleanupResources() {
    stopTimer();
    resourceGeometries.forEach(value => value.dispose());
    resourceMaterials.forEach(value => value.dispose());
    environment?.dispose();
    renderer?.dispose();
    resourceGeometries = [];
    resourceMaterials = [];
  }

  function fallback(reason) {
    ready = false;
    cleanupResources();
    canvas.hidden = true;
    labels.forEach(label => { label.element.hidden = true; });
    host.classList.add('fallback');
    status.textContent = `${reason} · 아래 2D 막대와 실측 표본 수로 표시합니다.`;
    renderCounts();
  }

  function init() {
    const vendor = window.UpbitVendors;
    if (!vendor?.THREE) return fallback('3D 라이브러리를 불러오지 못했습니다');
    const T = vendor.THREE;
    try {
      renderer = new T.WebGLRenderer({canvas, antialias: true, alpha: true, powerPreference: 'low-power'});
      renderer.setClearColor(0xf2f7f8, 0);
      renderer.toneMapping = T.ACESFilmicToneMapping;
      renderer.toneMappingExposure = 1.1;
      scene = new T.Scene();
      camera = new T.OrthographicCamera(-8, 8, 4, -4, .1, 80);
      group = new T.Group();
      scene.add(group);
      scene.add(new T.HemisphereLight(0xffffff, 0xb7ccce, 1.8));
      const light = new T.DirectionalLight(0xffffff, 2);
      light.position.set(-3, 7, 6);
      scene.add(light);
      if (vendor.RoomEnvironment) {
        const pmrem = new T.PMREMGenerator(renderer);
        const room = new vendor.RoomEnvironment();
        environment = pmrem.fromScene(room, .04);
        scene.environment = environment.texture;
        room.dispose();
        pmrem.dispose();
      }
      const geometry = new T.BoxGeometry(.48, 1, .58);
      resourceGeometries.push(geometry);
      const materials = colors.map(color => new T.MeshPhysicalMaterial({color, metalness: .12, roughness: .06, transmission: .38, thickness: 1.1, transparent: true, opacity: .98, ior: 1.48, clearcoat: 1, clearcoatRoughness: .04}));
      resourceMaterials.push(...materials);
      const edgesGeometry = new T.EdgesGeometry(geometry);
      const edgeMaterial = new T.LineBasicMaterial({color: 0xadd5d9, transparent: true, opacity: .55});
      resourceGeometries.push(edgesGeometry);
      resourceMaterials.push(edgeMaterial);
      for (let coinIndex = 0; coinIndex < 4; coinIndex++) {
        const label = document.createElement('span');
        label.className = 'scene-label';
        label.textContent = coins[coinIndex];
        label.setAttribute('aria-hidden', 'true');
        host.append(label);
        labels.push({index: coinIndex, element: label});
        for (let classIndex = 0; classIndex < 3; classIndex++) {
          const mesh = new T.Mesh(geometry, materials[classIndex]);
          mesh.rotation.y = .22;
          mesh.userData = {coinIndex, classIndex, target: .001};
          mesh.scale.y = .001;
          mesh.add(new T.LineSegments(edgesGeometry, edgeMaterial));
          group.add(mesh);
          meshes.push(mesh);
        }
      }
      const platformGeometry = new T.BoxGeometry(1, .1, 1);
      const platformMaterial = new T.MeshPhysicalMaterial({color: 0xddeff1, roughness: .18, transmission: .65, thickness: .2, transparent: true, opacity: .65, clearcoat: 1});
      resourceGeometries.push(platformGeometry);
      resourceMaterials.push(platformMaterial);
      const platform = new T.Mesh(platformGeometry, platformMaterial);
      platform.name = 'platform';
      platform.position.y = .025;
      group.add(platform);
      raycaster = new T.Raycaster();
      pointer = new T.Vector2();
      ready = true;
      resize();
      setTargets();
    } catch (error) {
      fallback('WebGL을 사용할 수 없습니다');
    }
  }

  function hover(event) {
    if (!ready || disposed) return;
    const rect = canvas.getBoundingClientRect();
    pointer.set((event.clientX - rect.left) / rect.width * 2 - 1, -(event.clientY - rect.top) / rect.height * 2 + 1);
    if (active()) {
      targetView = {x: pointer.x * .045, y: -pointer.y * .018};
      schedule();
    }
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObjects(meshes, false).find(item => item.object.visible);
    tooltip.hidden = !hit;
    if (hit) {
      const {coinIndex, classIndex, value} = hit.object.userData;
      tooltip.textContent = `${coins[coinIndex]} · ${classes[classIndex]} 예측 ${format(value)}개`;
      tooltip.style.left = `${Math.max(4, Math.min(rect.width - 210, event.clientX - rect.left + 12))}px`;
      tooltip.style.top = `${Math.max(4, Math.min(rect.height - 45, event.clientY - rect.top - 38))}px`;
    }
  }

  canvas.addEventListener('pointermove', hover);
  canvas.addEventListener('pointerleave', () => { tooltip.hidden = true; targetView = {x: 0, y: 0}; schedule(); });
  canvas.addEventListener('webglcontextlost', event => { event.preventDefault(); fallback('3D 그래픽 연결이 중단되었습니다'); });
  document.addEventListener('visibilitychange', () => { if (document.hidden) stopTimer(); else schedule(); });
  reduced.addEventListener('change', () => { stopTimer(); schedule(); });
  const observer = new ResizeObserver(resize);
  observer.observe(host);
  const intersection = new IntersectionObserver(entries => { visible = entries[0].isIntersecting; if (visible) schedule(); else stopTimer(); });
  intersection.observe(host);
  window.addEventListener('pagehide', () => { disposed = true; cleanupResources(); observer.disconnect(); intersection.disconnect(); }, {once: true});
  window.UpbitScene = {
    update(snapshot) {
      rows = coins.map(coin => ({coin, ...parseCounts(snapshot.markets?.[`KRW-${coin}`])}));
      const next = JSON.stringify(rows);
      if (next === signature) return;
      signature = next;
      renderCounts();
      const available = rows.filter(row => row.values).length;
      if (ready) status.textContent = available ? `후속 평가 최대 확률 예측 · 실측 표본 수 · ${available}/4개 시장 · 막대 높이 공통 척도` : '실측 예측 표본 대기 · 빈 막대는 생성하지 않습니다';
      if (ready) setTargets();
    },
    setMotion(value) { motion = Boolean(value); stopTimer(); targetView = {x: 0, y: 0}; schedule(); },
    stats() { return {renderframes, paused: !active(), ready, mobile, pending: !rows.some(row => row.values), counts: rows.map(row => ({coin: row.coin, values: row.values, source: row.source})), meshes: meshes.length, pixelRatio: ready ? renderer.getPixelRatio() : null}; }
  };
  init();
  renderCounts();
})();
