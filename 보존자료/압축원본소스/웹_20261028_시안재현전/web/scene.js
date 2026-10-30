'use strict';
(() => {
  const host = document.getElementById('scene-stage');
  const canvas = document.getElementById('prediction-scene');
  const status = document.getElementById('scene-status');
  const counts = document.getElementById('scene-counts');
  const tooltip = document.getElementById('scene-tooltip');
  const coins = ['BTC', 'SOL', 'ETH', 'XRP', 'DOGE'];
  const classes = ['하락', '관망', '상승'];
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
  let allocation = null;
  let trainingScene = null;
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
    for (const [index, row] of rows.entries()) {
      const heading = document.getElementById('coin-headings')?.children[index];
      const symbol = heading?.querySelector('small');
      if (symbol) symbol.textContent = row.values ? row.coin : `${row.coin} · 평가 대기`;
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

  function inViewport(element) {
    if (element.closest('[hidden],details:not([open])')) return false;
    const rect = element.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0 && rect.bottom > 0 && rect.top < innerHeight && rect.right > 0 && rect.left < innerWidth;
  }

  function syncVisibility() {
    visible = inViewport(host);
    allocation?.syncVisibility();
    if (visible || allocation?.visible) schedule();
    else stopTimer();
  }

  function active() { return motion && !reduced.matches && !document.hidden && (visible || allocation?.visible) && !disposed; }

  function frame() {
    timer = null;
    if (!ready || disposed || document.hidden || !(visible || allocation?.visible)) return;
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
    if (visible) { renderer.render(scene, camera); renderframes++; }
    for (const label of labels) {
      const anchor = meshes[label.index].position.clone();
      anchor.y = .1;
      anchor.z += .8;
      anchor.applyMatrix4(group.matrixWorld).project(camera);
      label.element.style.left = `${(anchor.x * .5 + .5) * host.clientWidth}px`;
      label.element.style.top = `${(-anchor.y * .5 + .5) * host.clientHeight}px`;
    }
    const headingContainer = document.getElementById('coin-headings');
    if (headingContainer) {
      if (mobile) {
        headingContainer.style.inset = '0';
        headingContainer.style.display = 'block';
      } else headingContainer.removeAttribute('style');
      [...headingContainer.children].forEach((heading, index) => {
        if (!mobile) { heading.removeAttribute('style'); return; }
        const anchor = meshes[index * 3 + 1].position.clone();
        anchor.y = 3.1;
        anchor.project(camera);
        heading.style.position = 'absolute';
        heading.style.width = '45%';
        heading.style.justifyContent = 'center';
        heading.style.transform = 'translate(-50%,-100%)';
        heading.style.left = `${(anchor.x * .5 + .5) * host.clientWidth}px`;
        heading.style.top = `${host.offsetTop + (-anchor.y * .5 + .5) * host.clientHeight}px`;
      });
    }
    if (allocation?.draw(active())) changing = true;
    if (changing && active()) schedule();
  }

  function schedule() {
    if (timer === null && ready && !document.hidden && (visible || allocation?.visible) && !disposed) timer = setTimeout(frame, Math.ceil(1000 / 24));
  }

  function resize() {
    if (!ready || disposed) return;
    visible = inViewport(host);
    const width = host.clientWidth;
    if (!width) return;
    mobile = width < 730;
    // 장면 행 수와 CSS 높이는 같은 컨테이너 폭 판정을 사용합니다.
    document.getElementById('prediction-workspace').classList.toggle('stacked-scene', mobile);
    const height = host.clientHeight;
    if (!height) return;
    renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 1.25));
    renderer.setSize(width, height, false);
    const aspect = width / height;
    const viewWidth = mobile ? 8.8 : Math.max(15, 5 * aspect);
    const viewHeight = Math.max(mobile ? 12 : 5.1, viewWidth / aspect);
    camera.left = -viewWidth / 2;
    camera.right = viewWidth / 2;
    camera.top = viewHeight / 2;
    camera.bottom = -viewHeight / 2;
    camera.position.set(0, mobile ? 12 : 6.4, mobile ? 8 : 13);
    camera.lookAt(0, .9, 0);
    camera.updateProjectionMatrix();
    for (const mesh of meshes) {
      const {coinIndex, classIndex} = mesh.userData;
      const column = coinIndex === coins.length - 1 ? 0 : coinIndex % 2 ? 2 : -2;
      mesh.position.x = mobile ? column + (classIndex - 1) * .66 : (coinIndex - (coins.length - 1) / 2) * viewWidth * .18 + (classIndex - 1) * .9;
      mesh.position.z = mobile ? (Math.floor(coinIndex / 2) - 1) * 5.4 : 0;
      mesh.scale.x = mobile ? 1 : 1.6;
      mesh.scale.z = mobile ? 1 : 1.38;
    }
    const platform = group.getObjectByName('platform');
    platform.scale.set(mobile ? 7.6 : viewWidth * .95, 1, mobile ? 13.8 : 2.8);
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
    host.style.height = '0px';
    host.style.marginTop = '0';
    document.getElementById('prediction-detail').open = true;
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
      for (let coinIndex = 0; coinIndex < coins.length; coinIndex++) {
        for (let classIndex = 0; classIndex < 3; classIndex++) {
          const label = document.createElement('span');
          label.className = `scene-label ${classNames[classIndex]}`;
          label.textContent = classes[classIndex];
          label.setAttribute('aria-hidden', 'true');
          host.append(label);
          labels.push({index: coinIndex * 3 + classIndex, element: label});
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
  const intersection = new IntersectionObserver(syncVisibility);
  intersection.observe(host);
  window.addEventListener('resize', () => { resize(); allocation?.resize(); syncVisibility(); });
  window.addEventListener('scroll', syncVisibility, {passive: true});
  window.addEventListener('pagehide', () => { disposed = true; allocation?.dispose(); cleanupResources(); observer.disconnect(); intersection.disconnect(); }, {once: true});
  window.UpbitScene = {
    update(snapshot) {
      syncVisibility();
      allocation?.update(snapshot);
      trainingScene?.update(snapshot);
      rows = coins.map(coin => ({coin, ...parseCounts(snapshot.markets?.[`KRW-${coin}`])}));
      const next = JSON.stringify(rows);
      if (next === signature) return;
      signature = next;
      renderCounts();
      const available = rows.filter(row => row.values).length;
      if (ready) status.textContent = available ? `과거 후속 평가 최대 확률 예측 · 표본 수는 확률이 아닙니다 · ${available}/${coins.length}개 시장 기록 · 막대 높이 공통 척도` : '실측 예측 표본 대기 · 빈 막대는 생성하지 않습니다';
      if (ready) setTargets();
    },
    setMotion(value) { motion = Boolean(value); stopTimer(); targetView = {x: 0, y: 0}; schedule(); trainingScene?.refresh(); },
    stats() { return {renderframes, paused: !ready || !active(), ready, mobile, pending: !rows.some(row => row.values), counts: rows.map(row => ({coin: row.coin, values: row.values, source: row.source})), meshes: meshes.length, pixelRatio: ready ? renderer.getPixelRatio() : null, allocation: allocation?.stats(), training: trainingScene?.stats()}; }
  };
  function createTrainingScene() {
    const stage = document.getElementById('training-stage');
    const element = document.getElementById('training-scene');
    const fallbackText = document.getElementById('training-scene-fallback');
    const state = {ready: false, renderframes: 0, visible: false, stage: null, meshes: 0, pixelRatio: null};
    const geometries = [], materials = [], modules = [], links = [];
    const captions = document.getElementById('workflow-steps');
    stage.append(captions);
    stage.classList.add('projected-workflow');
    let renderer3, scene3, camera3, timer3 = null, stopped = false;
    const canAnimate = () => state.ready && state.visible && !document.hidden && motion && !reduced.matches && !stopped;
    function draw() {
      timer3 = null;
      if (!state.ready || !state.visible || document.hidden || stopped) return;
      modules.forEach((module, index) => {
        module.rotation.y = canAnimate() && state.stage === index ? Math.sin(Date.now() / 1400) * .12 : 0;
      });
      renderer3.render(scene3, camera3);
      state.renderframes++;
      if (canAnimate()) timer3 = setTimeout(draw, Math.ceil(1000 / 24));
    }
    function refresh() {
      if (timer3 !== null) clearTimeout(timer3);
      timer3 = null;
      state.visible = inViewport(stage);
      if (state.visible && state.ready && !document.hidden) draw();
    }
    function resize3() {
      if (!state.ready || !stage.clientWidth || !stage.clientHeight) return;
      const width = stage.clientWidth, height = stage.clientHeight, aspect = width / height;
      renderer3.setPixelRatio(Math.min(devicePixelRatio || 1, 1.25));
      state.pixelRatio = renderer3.getPixelRatio();
      renderer3.setSize(width, height, false);
      const narrow = matchMedia('(max-width:760px)').matches;
      const view = narrow ? 6.4 / aspect : Math.max(2.9, 11.8 / aspect);
      camera3.left = -view * aspect / 2; camera3.right = view * aspect / 2;
      camera3.top = view / 2; camera3.bottom = -view / 2;
      camera3.updateProjectionMatrix();
      modules.forEach((module, index) => {
        module.position.set(narrow ? (index % 2 - .5) * 3.2 : (index - 1.5) * 2.8,
          narrow ? (index < 2 ? 2.6 : -2.1) : 0, 0);
      });
      links.forEach(link => { link.visible = !narrow; });
      // 모듈의 실제 화면 좌표에 설명을 연결해 반응형에서도 대응을 유지합니다.
      scene3.updateMatrixWorld(true);
      camera3.updateMatrixWorld(true);
      state.anchors = modules.map((module, index) => {
        const point = module.position.clone();
        point.y -= .12;
        point.z += .8;
        point.project(camera3);
        const x = (point.x + 1) * width / 2;
        const y = (1 - point.y) * height / 2 + 9;
        const caption = captions.children[index];
        caption.style.left = `${x}px`;
        caption.style.top = `${y}px`;
        caption.style.width = `${width / (narrow ? 2 : 4) - 18}px`;
        return {index, x, y};
      });
      refresh();
    }
    try {
      const T = window.UpbitVendors?.THREE;
      if (!T) throw new Error('3D 라이브러리 미기록');
      renderer3 = new T.WebGLRenderer({canvas: element, alpha: true, antialias: true, powerPreference: 'low-power'});
      renderer3.setClearColor(0xffffff, 0);
      scene3 = new T.Scene();
      camera3 = new T.OrthographicCamera(-6, 6, 3, -3, .1, 50);
      camera3.position.set(0, 7, 12); camera3.lookAt(0, .5, 0);
      scene3.add(new T.HemisphereLight(0xffffff, 0xa8c4ca, 2));
      const light = new T.DirectionalLight(0xffffff, 2); light.position.set(-4, 8, 5); scene3.add(light);
      const baseGeo = new T.BoxGeometry(2.15, .22, 1.55);
      const connectorGeo = new T.BoxGeometry(.8, .09, .14);
      geometries.push(baseGeo, connectorGeo);
      const baseMat = new T.MeshStandardMaterial({color: 0xb8cbd0, metalness: .22, roughness: .45});
      const linkMat = new T.MeshStandardMaterial({color: 0x80a6a8, metalness: .25, roughness: .4});
      materials.push(baseMat, linkMat);
      for (let i = 0; i < 4; i++) {
        const module = new T.Group(); module.position.x = (i - 1.5) * 2.8; scene3.add(module); modules.push(module);
        const base = new T.Mesh(baseGeo, baseMat); module.add(base);
        const material = new T.MeshStandardMaterial({color: 0x768d95, metalness: .18, roughness: .3, transparent: true, opacity: .95});
        materials.push(material);
        module.userData.stageMaterial = material;
        const add = (geometry, x, y, z, rotation = 0) => {
          geometries.push(geometry);
          const part = new T.Mesh(geometry, material);
          part.position.set(x, y, z); part.rotation.z = rotation; module.add(part); state.meshes++;
        };
        if (i === 0) {
          for (let sheet = 0; sheet < 5; sheet++) add(new T.BoxGeometry(.23, 1.15, .8), (sheet - 2) * .29, .75, 0);
        } else if (i === 1) {
          add(new T.BoxGeometry(1.25, .36, 1.05), 0, .5, 0);
          for (let pin = 0; pin < 5; pin++) {
            add(new T.BoxGeometry(.1, .12, .22), (pin - 2) * .23, .27, -.62);
            add(new T.BoxGeometry(.1, .12, .22), (pin - 2) * .23, .27, .62);
          }
        } else if (i === 2) {
          add(new T.BoxGeometry(1.25, .12, .9), 0, .27, 0);
          add(new T.TorusGeometry(.45, .07, 8, 32), 0, .94, 0);
          add(new T.BoxGeometry(.13, .6, .13), .46, .48, 0, .75);
        } else {
          add(new T.BoxGeometry(1.1, 1.25, .25), 0, .85, 0);
          for (let line = 0; line < 3; line++) add(new T.BoxGeometry(.65, .06, .035), 0, .55 + line * .24, .15);
        }
        if (i < 3) { const link = new T.Mesh(connectorGeo, linkMat); link.position.set(module.position.x + 1.4, .02, 0); scene3.add(link); links.push(link); }
      }
      state.ready = true;
      resize3();
    } catch (error) {
      renderer3?.dispose(); element.hidden = true; fallbackText.hidden = false; stage.classList.add('workflow-fallback');
    }
    const sizeObserver = new ResizeObserver(resize3); sizeObserver.observe(stage);
    const visibilityObserver = new IntersectionObserver(refresh); visibilityObserver.observe(stage);
    document.addEventListener('visibilitychange', refresh);
    reduced.addEventListener('change', refresh);
    window.addEventListener('scroll', refresh, {passive: true});
    element.addEventListener('webglcontextlost', event => { event.preventDefault(); state.ready = false; element.hidden = true; fallbackText.hidden = false; stage.classList.add('workflow-fallback'); refresh(); });
    window.addEventListener('pagehide', () => { stopped = true; if (timer3 !== null) clearTimeout(timer3); sizeObserver.disconnect(); visibilityObserver.disconnect(); geometries.forEach(item => item.dispose()); materials.forEach(item => item.dispose()); renderer3?.dispose(); }, {once: true});
    return {
      refresh,
      update(snapshot) {
        const t = snapshot.training || {};
        state.stage = t.state === 'running' ? ({features: 0, fit: 1, calibration: 2, locked_test: 2, complete: 3})[t.stage] ?? null : null;
        modules.forEach((module, index) => { module.userData.stageMaterial.color.setHex(index === state.stage ? 0x008c83 : 0x768d95); });
        refresh();
      },
      stats() { return {...state, paused: !canAnimate()}; }
    };
  }
  function createAllocation() {
    const element = document.getElementById('allocation-scene');
    const stage = document.getElementById('allocation-stage');
    const note = document.getElementById('allocation-note');
    const state = {ready: false, cashRatio: null, coinRatio: null, pending: true, meshes: 0, renderframes: 0};
    let allocationRenderer, allocationScene, allocationCamera, allocationGroup, cashMesh, coinMesh;
    let lastInput = '', currentCash = 0, currentCoins = 0, targetAngle = 0, currentAngle = 0;
    let dataSource = null;
    const geometries = [], materials = [];
    const api = {visible: true};
    api.syncVisibility = () => { api.visible = inViewport(stage); };
    const isNumber = value => typeof value === 'number' && Number.isFinite(value);
    const ratioText = value => `${(value * 100).toLocaleString('ko-KR', {maximumFractionDigits: 1})}%`;

    function fallbackBars() {
      stage.querySelector('.allocation-fallback')?.remove();
      if (state.ready || state.pending) return;
      const bar = document.createElement('div');
      bar.className = 'allocation-fallback';
      bar.setAttribute('aria-hidden', 'true');
      for (const ratio of [state.cashRatio, state.coinRatio]) {
        const fill = document.createElement('span');
        fill.style.width = `${ratio * 100}%`;
        bar.append(fill);
      }
      stage.append(bar);
    }

    function resizeAllocation() {
      if (!state.ready || !stage.clientWidth || !stage.clientHeight) return;
      const aspect = stage.clientWidth / stage.clientHeight;
      allocationRenderer.setPixelRatio(Math.min(devicePixelRatio || 1, 1.25));
      allocationRenderer.setSize(stage.clientWidth, stage.clientHeight, false);
      const height = Math.max(4.4, 3.5 / aspect);
      allocationCamera.left = -height * aspect / 2;
      allocationCamera.right = height * aspect / 2;
      allocationCamera.top = height / 2;
      allocationCamera.bottom = -height / 2;
      allocationCamera.updateProjectionMatrix();
      schedule();
    }

    function disposeAllocation() {
      geometries.forEach(geometry => geometry.dispose());
      materials.forEach(material => material.dispose());
      allocationRenderer?.dispose();
    }

    try {
      const T = window.UpbitVendors?.THREE;
      if (!T) throw new Error('3D 라이브러리 미기록');
      allocationRenderer = new T.WebGLRenderer({canvas: element, alpha: true, antialias: true, powerPreference: 'low-power'});
      allocationRenderer.setClearColor(0xffffff, 0);
      allocationRenderer.toneMapping = T.ACESFilmicToneMapping;
      allocationRenderer.toneMappingExposure = 1.05;
      allocationScene = new T.Scene();
      if (environment) allocationScene.environment = environment.texture;
      allocationCamera = new T.OrthographicCamera(-3, 3, 2.2, -2.2, .1, 50);
      allocationCamera.position.set(4, 3.2, 6);
      allocationCamera.lookAt(0, 1.55, 0);
      allocationScene.add(new T.HemisphereLight(0xffffff, 0xbed6d7, 2));
      const light = new T.DirectionalLight(0xffffff, 2.5);
      light.position.set(-3, 5, 5);
      allocationScene.add(light);
      allocationGroup = new T.Group();
      allocationScene.add(allocationGroup);
      const shellGeometry = new T.BoxGeometry(1.6, 3, 1.2);
      const fillGeometry = new T.BoxGeometry(1.51, 1, 1.11);
      const edges = new T.EdgesGeometry(shellGeometry);
      const platformGeometry = new T.BoxGeometry(2.2, .07, 1.75);
      geometries.push(shellGeometry, fillGeometry, edges, platformGeometry);
      const shellMaterial = new T.MeshPhysicalMaterial({color: 0xbce7e4, roughness: .07, transmission: .9, thickness: .18, transparent: true, opacity: .32, clearcoat: 1, depthWrite: false});
      const cashMaterial = new T.MeshPhysicalMaterial({color: 0x12a99c, roughness: .08, transmission: .5, thickness: 1.1, transparent: true, opacity: .93, clearcoat: 1});
      const coinMaterial = new T.MeshPhysicalMaterial({color: 0x8298a5, roughness: .08, transmission: .45, thickness: 1.1, transparent: true, opacity: .93, clearcoat: 1});
      const edgeMaterial = new T.LineBasicMaterial({color: 0x5caaa8, transparent: true, opacity: .65});
      const platformMaterial = new T.MeshPhysicalMaterial({color: 0xe4f3f1, transparent: true, opacity: .55, transmission: .6, roughness: .12});
      materials.push(shellMaterial, cashMaterial, coinMaterial, edgeMaterial, platformMaterial);
      const shell = new T.Mesh(shellGeometry, shellMaterial);
      shell.position.y = 1.6;
      shell.add(new T.LineSegments(edges, edgeMaterial));
      cashMesh = new T.Mesh(fillGeometry, cashMaterial);
      coinMesh = new T.Mesh(fillGeometry, coinMaterial);
      cashMesh.visible = false;
      coinMesh.visible = false;
      const platform = new T.Mesh(platformGeometry, platformMaterial);
      platform.position.y = .035;
      allocationGroup.add(shell, cashMesh, coinMesh, platform);
      state.meshes = 4;
      state.ready = true;
      resizeAllocation();
    } catch (error) {
      disposeAllocation();
      element.hidden = true;
      state.ready = false;
    }

    const sizeObserver = new ResizeObserver(resizeAllocation);
    sizeObserver.observe(stage);
    const visibilityObserver = new IntersectionObserver(() => {
      api.syncVisibility();
      if (api.visible || visible) schedule();
      else stopTimer();
    });
    visibilityObserver.observe(stage);
    element.addEventListener('pointermove', event => {
      if (!active()) return;
      const rect = element.getBoundingClientRect();
      targetAngle = ((event.clientX - rect.left) / rect.width - .5) * .14;
      schedule();
    });
    element.addEventListener('pointerleave', () => { targetAngle = 0; schedule(); });
    element.addEventListener('webglcontextlost', event => {
      event.preventDefault();
      state.ready = false;
      element.hidden = true;
      disposeAllocation();
      note.textContent = '3D 연결 중단 · 아래 숫자와 2D 비중으로 표시합니다.';
      fallbackBars();
    });

    api.update = snapshot => {
      const account = snapshot.account || {};
      const assets = snapshot.assets;
      const input = JSON.stringify([account.cash, account.equity, assets?.map(asset => asset.market_value)]);
      if (input === lastInput) return;
      lastInput = input;
      const cash = account.cash, equity = account.equity;
      const hasAssets = Array.isArray(assets) && assets.every(asset => isNumber(asset.market_value) && asset.market_value >= 0);
      const coinValue = hasAssets ? assets.reduce((sum, asset) => sum + asset.market_value, 0) : null;
      const complete = isNumber(cash) && cash >= 0 && isNumber(equity) && equity >= 0 && hasAssets;
      const consistent = complete && Math.abs(cash + coinValue - equity) <= Math.max(.01, equity * 1e-6);
      state.pending = !consistent;
      state.cashRatio = consistent ? equity > 0 ? cash / equity : 0 : null;
      state.coinRatio = consistent ? equity > 0 ? coinValue / equity : 0 : null;
      dataSource = {cash: isNumber(cash) ? cash : null, equity: isNumber(equity) ? equity : null, coinValue, assetRows: Array.isArray(assets) ? assets.length : null};
      document.getElementById('allocation-cash').textContent = consistent ? ratioText(state.cashRatio) : '미기록';
      document.getElementById('allocation-coins').textContent = consistent ? ratioText(state.coinRatio) : '미기록';
      note.textContent = !complete ? '자산 기록 확인 중' : !consistent ? '자산 평가 합계 확인 중' : equity === 0 ? '기록된 자산 0원' : '현재 자금의 실제 비중';
      if (!state.ready) note.textContent += ' · 3D 사용 불가, 2D 비중 표시';
      element.setAttribute('aria-label', consistent ? `실제 자산 비중: 현금 ${ratioText(state.cashRatio)}, 코인 ${ratioText(state.coinRatio)}` : '자산 비중 미기록');
      if (state.pending) currentCash = currentCoins = 0;
      fallbackBars();
      schedule();
    };
    api.draw = animate => {
      if (!state.ready || !api.visible || document.hidden) return false;
      let changing = false;
      for (const kind of ['cash', 'coins']) {
        const target = state.pending ? 0 : (kind === 'cash' ? state.cashRatio : state.coinRatio) * 2.88;
        const current = kind === 'cash' ? currentCash : currentCoins;
        let next = animate ? current + (target - current) * .18 : target;
        if (Math.abs(target - next) < .002) next = target;
        else changing = true;
        if (kind === 'cash') currentCash = next;
        else currentCoins = next;
      }
      cashMesh.visible = !state.pending && currentCash > 0;
      coinMesh.visible = !state.pending && currentCoins > 0;
      cashMesh.scale.y = Math.max(.001, currentCash);
      coinMesh.scale.y = Math.max(.001, currentCoins);
      cashMesh.position.y = .15 + currentCash / 2;
      coinMesh.position.y = .15 + currentCash + currentCoins / 2;
      currentAngle = animate ? currentAngle + (targetAngle - currentAngle) * .14 : 0;
      if (animate && Math.abs(targetAngle - currentAngle) > .0005) changing = true;
      allocationGroup.rotation.y = currentAngle;
      allocationRenderer.render(allocationScene, allocationCamera);
      state.renderframes++;
      return changing;
    };
    api.stats = () => ({...state, paused: !state.ready || !active() || !api.visible, source: dataSource});
    api.resize = resizeAllocation;
    api.dispose = () => { disposeAllocation(); sizeObserver.disconnect(); visibilityObserver.disconnect(); };
    return api;
  }
  init();
  allocation = createAllocation();
  trainingScene = createTrainingScene();
  renderCounts();
})();
