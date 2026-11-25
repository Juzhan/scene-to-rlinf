// The scene editor's interface (tools/editor_ui/app.css): a top bar, the scene tree on the left, the selection's
// properties on the right (the scene's when nothing is selected), view tools over the canvas. Injected into
// viser's page by Editor._start_app; viser keeps the 3D view. Data and edits go to the editor through
// /api/outliner/ (tree), /api/ui/ (top bar, properties) and /api/import/ (files brought in); only the panels'
// arrangement is kept here.
(() => {
  if (document.getElementById('rl-app')) return;
  const API = document.getElementById('rl-app-js').dataset.api;
  const esc = text => String(text ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
  const svg = body => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">${body}</svg>`;
  const ICON = {
    robot: '<rect x="6" y="5" width="12" height="9" rx="2"/><path d="M9 18h6M12 14v4M9.5 9h.01M14.5 9h.01M12 2v3"/>',
    box: '<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z"/><path d="M12 12l8-4.5M12 12v9M12 12L4 7.5"/>',
    mountain: '<path d="M3 20l6-10 4 6 3-4 5 8z"/>', radar: '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3"/><path d="M12 12l6-6"/>',
    bulb: '<path d="M9 17h6M10 21h4M12 3a6 6 0 0 0-3.5 10.9V17h7v-3.1A6 6 0 0 0 12 3z"/>',
    'stack-2': '<path d="M12 4l8 4-8 4-8-4z"/><path d="M4 12l8 4 8-4M4 16l8 4 8-4"/>',
    'map-pin': '<circle cx="12" cy="10" r="3"/><path d="M12 21s-7-6.2-7-11a7 7 0 0 1 14 0c0 4.8-7 11-7 11z"/>',
    camera: '<path d="M4 8h3l2-3h6l2 3h3v11H4z"/><circle cx="12" cy="13" r="3.2"/>',
    folder: '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
    sim: '<circle cx="12" cy="12" r="8.5"/><path d="M3.5 12h17M12 3.5c3 3.2 3 13.8 0 17M12 3.5c-3 3.2-3 13.8 0 17"/>',
    joint: '<circle cx="12" cy="12" r="3"/><path d="M12 3v6M12 15v6"/>', slide: '<path d="M12 4v16M8 8l4-4 4 4M8 16l4 4 4-4"/>',
  };
  const I = {
    caret: svg('<path d="M9 6l6 6-6 6"/>'),
    magnet: svg('<path d="M6 3v8a6 6 0 0 0 12 0V3h-4v8a2 2 0 0 1-4 0V3zM6 7h4M14 7h4"/>'),
    drop: svg('<path d="M12 3v11M7 9l5 5 5-5M4 20h16"/>'), open: svg('<path d="M14 4h6v6M20 4l-9 9M18 14v6H4V6h6"/>'), left: svg('<path d="M15 6l-6 6 6 6"/>'), right: svg('<path d="M9 6l6 6-6 6"/>'),
    eye: svg('<path d="M2.5 12s3.5-6.5 9.5-6.5S21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z"/><circle cx="12" cy="12" r="2.6"/>'),
    eyeOff: svg('<path d="M3 3l18 18M10.6 6.1A9.7 9.7 0 0 1 12 6c6 0 9.5 6 9.5 6a17 17 0 0 1-3 3.7M6.6 6.6C4 8.3 2.5 12 2.5 12S6 18 12 18c1.6 0 3-.4 4.3-1"/>'),
    lock: svg('<rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/>'),
    pointer: svg('<path d="M6 3l12 9-5.5 1.2L16 20l-2.6 1.2-3.3-6.7L6 18z"/>'),
    render: svg('<path d="M4 8h3l2-3h6l2 3h3v11H4z"/><circle cx="12" cy="13" r="3.2"/>'),
    renderOff: svg('<path d="M3 3l18 18M8 5h7l2 3h3v10M17 19H4V8h3"/><path d="M9.8 10.9a3.2 3.2 0 0 0 4.3 4.3"/>'),
    dynamic: svg('<circle cx="12" cy="7.5" r="4"/><path d="M12 14v7M8.5 17.5L12 21l3.5-3.5"/>'),
    kinematic: svg('<rect x="3" y="8" width="10" height="9" rx="1.5"/><path d="M16 12.5h5M18.5 10l2.5 2.5-2.5 2.5"/>'),
    fixed: svg('<circle cx="12" cy="4.5" r="1.6"/><path d="M12 6v14M8 9h8M5 13a7 7 0 0 0 14 0"/>'),
    static: svg('<rect x="3" y="6" width="18" height="12" rx="1"/><path d="M3 12h18M9 6v6M15 12v6"/>'),
    link: svg('<path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1"/>'),
    motion: svg('<path d="M4 17c4 0 4-10 8-10s4 10 8 10"/>'),
    move: svg('<path d="M12 3v18M3 12h18M12 3l-3 3M12 3l3 3M12 21l-3-3M12 21l3-3M3 12l3-3M3 12l3 3M21 12l-3-3M21 12l-3 3"/>'),
    rotate: svg('<path d="M20 12a8 8 0 1 1-2.3-5.7M20 4v4h-4"/>'),
    scale: svg('<path d="M4 20V14M4 20h6M4 20l7-7M20 4h-6M20 4v6M20 4l-7 7"/>'),
    undo: svg('<path d="M9 14L4 9l5-5"/><path d="M4 9h10a6 6 0 0 1 0 12h-3"/>'), redo: svg('<path d="M15 14l5-5-5-5"/><path d="M20 9H10a6 6 0 0 0 0 12h3"/>'),
    play: svg('<path d="M7 4l13 8-13 8z"/>'), focus: svg('<circle cx="12" cy="12" r="3"/><path d="M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3"/>'),
    trash: svg('<path d="M4 7h16M10 11v6M14 11v6M5 7l1 13h12l1-13M9 7V4h6v3"/>'), x: svg('<path d="M6 6l12 12M18 6L6 18"/>'),
    chev: svg('<path d="M6 9l6 6 6-6"/>'), scene: svg('<path d="M3 20h18M6 20V10l6-5 6 5v10"/>'),
    target: svg('<circle cx="12" cy="12" r="7"/><circle cx="12" cy="12" r="2"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3"/>'),
    plus: svg('<path d="M12 5v14M5 12h14"/>'), upload: svg('<path d="M12 16V4M7 9l5-5 5 5M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/>'),
    search: svg('<circle cx="11" cy="11" r="6"/><path d="M20 20l-4.5-4.5"/>'), check: svg('<path d="M5 12.5l4.5 4.5L19 7.5"/>'),
    image: svg('<rect x="3" y="5" width="18" height="14" rx="2"/><circle cx="9" cy="10" r="1.6"/><path d="M4 18l5-5 4 4 3-3 4 4"/>'),
    sun: svg('<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>'),
    cube: svg('<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z"/><path d="M4 7.5l8 4.5 8-4.5M12 12v9"/>'),
    panel: svg('<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M15 4v16"/>'),
    panelLeft: svg('<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M9 4v16"/>'),
    log: svg('<path d="M5 4h14a1 1 0 0 1 1 1v11a1 1 0 0 1-1 1h-8l-4 3v-3H5a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1z"/><path d="M8 9h8M8 12.5h5"/>'),
    more: svg('<circle cx="5" cy="12" r="1.3"/><circle cx="12" cy="12" r="1.3"/><circle cx="19" cy="12" r="1.3"/>'),
  };
  const MODES = [['joint_position', '位置'], ['joint_velocity', '速度'], ['joint_effort', '力矩'], ['none', '不控制']];
  const ROLE = {planner: 'Planner', executor: 'Executor'};
  // Policy layers as the editor names and colours them (role_labels): {role: {label, slot}}; slot picks the colour.
  let ROLES = {};
  const roleName = role => (ROLES[role] && ROLES[role].label) || ROLE[role] || role;
  const roleVar = role => `var(--rl-${(ROLES[role] && ROLES[role].slot) || role}, #8a919c)`;
  const roleMark = role => (ROLES[role] && ROLES[role].mark) || roleName(role)[0].toUpperCase();
  const KINDS = [['object', '物体'], ['robot', '机器人'], ['scene', '场景']];
  const short = label => String(label || '').replace(/^根部/, '').replace(/（绝对值）/, '').replace(/^坐标系(位置|朝向)（(.+)）$/, '$2 $1');
  // "宽度（像素）" -> name, the unit shown in the field, the whole text for the tooltip.
  const UNITS = {'像素': 'px', '米': 'm', '秒': 's', '度': '°'};
  function splitLabel(label) {
    const m = /^(.*?)（(.*)）$/.exec(label || '');
    if (!m) return [label || '', '', label || ''];
    const first = m[2].split(/[，,；]/)[0].trim(), unit = UNITS[first] || first;
    return [m[1].trim(), unit.length <= 4 ? unit : '', label];
  }
  const baseName = path => String(path || '').replace(/^world\//, '');
  const KEY_LABELS = {resolution: '间距', size: '范围'};  // keys inside a parameter table (a scan grid)

  document.body.insertAdjacentHTML('beforeend', `<div id="rl-app">
    <header id="rl-top">
      <span class="brand">RLinf</span><button class="project" id="rl-project"></button><span id="rl-cmds"></span>
      <button class="rl-tb" id="rl-more" title="更多" hidden>${I.more}</button>
      <button id="rl-problem" hidden>⚠ 后端不接受当前场景</button><span class="grow"></span>
      <span class="pick" id="rl-physics" hidden><em>物理</em><span class="slot"></span></span>
      <span class="pick" id="rl-renderer" hidden><em>渲染</em><span class="slot"></span></span>
      <button class="rl-icon rl-side" id="rl-side-left" title="显示 / 收起左栏（T）">${I.panelLeft}</button>
      <button class="rl-icon rl-side" id="rl-side-right" title="显示 / 收起右栏（N）">${I.panel}</button>
      <button class="rl-icon" id="rl-preview" title="复制仿真预览命令（带当前视角；选中相机时改为看该相机的真实画面）">${I.play}</button>
      <button class="rl-icon" id="rl-log" title="消息：提示和错误都留在这里，可展开看详细输出">${I.log}</button>
      <button class="rl-icon" id="rl-undo" title="撤回 Ctrl+Z">${I.undo}</button>
      <button class="rl-icon" id="rl-redo" title="重做 Ctrl+Shift+Z">${I.redo}</button>
      <button id="rl-save" title="保存到 scene.yaml（Ctrl+S）">保存</button>
    </header>
    <div id="rl-nav" title="点轴：沿该轴看；拖动：绕视点旋转"></div>
    <div id="rl-tools">
      <button data-gizmo="move" title="移动 G">${I.move}</button><button data-gizmo="rotate" title="旋转 R">${I.rotate}</button>
      <button data-gizmo="scale" title="缩放 S">${I.scale}</button><hr><button data-tool="snap" title="吸附表面：拖动物体时贴着下方表面（桌面、地形）走，松手落稳">${I.magnet}</button>
      <button data-tool="drop" title="落到下方表面 End">${I.drop}</button><button data-tool="focus" title="聚焦 F">${I.focus}</button>
    </div>
    <aside id="rl-left">
      <input id="rl-filter" type="search" placeholder="筛选" autocomplete="off">
      <div id="rl-reads"></div>
      <div id="rl-tree"></div>
      <div id="rl-adds">
        <button id="rl-add" title="资产架 Shift+A">${I.plus}添加</button>
        <button id="rl-import" title="导入自己的文件：URDF、MJCF、USD/USDZ、OBJ、FBX、glTF/GLB、STL、PLY、DAE（或打包的 zip / 文件夹）；也可以直接拖进窗口">${I.upload}导入</button>
      </div>
      <div id="rl-drawer" hidden><div class="bar">添加到场景<span style="flex:1"></span><button class="rl-icon" id="rl-drawer-close">${I.x}</button></div></div>
    </aside>
    <aside id="rl-right"></aside>
    <div class="rl-resize" data-side="left" title="拖动调宽，双击收起"></div>
    <div class="rl-resize" data-side="right" title="拖动调宽，双击收起"></div>
    <div id="rl-menu" hidden></div>
    <div id="rl-pop" hidden></div>
    <div id="rl-dialog" hidden><div class="box"></div></div>
    <div id="rl-dropzone" hidden><div>${I.upload}<b>松开导入</b><span>URDF · MJCF · USD · OBJ · FBX · glTF · STL · PLY · DAE · zip</span></div></div>
    <div id="rl-toast" title="点击查看全部消息"></div>
    <div id="rl-log-panel" hidden></div>
    <div id="rl-jobs" hidden></div>
    <input type="file" id="rl-files" multiple hidden><input type="file" id="rl-folder" webkitdirectory hidden>
  </div><button id="rl-back">返回新界面</button>`);
  const $ = id => document.getElementById(id);

  async function api(path, body) {
    const response = await fetch(API + 'api/' + path, body === undefined ? {} :
      {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || response.statusText);
    return data;
  }
  async function act(path, body) {
    try { const data = await api(path, body); kick(); return data; } catch (error) { toast(error.message, true); }
  }
  // Continuous edits (a drag): one request in flight, the latest value next.
  const live = {};
  function sendLive(path, body) {
    const slot = live[path] || (live[path] = {busy: false, next: null});
    if (slot.busy) { slot.next = body; return; }
    slot.busy = true;
    api(path, body).catch(() => {}).finally(() => { slot.busy = false; if (slot.next) { const next = slot.next; slot.next = null; sendLive(path, next); } });
  }
  // The release of a drag: drop the queued preview, wait for the one in flight, then commit (one undo step).
  function finish(path, body) {
    const slot = live[path];
    if (slot) slot.next = null;
    return new Promise(resolve => { const wait = () => (slot && slot.busy ? setTimeout(wait, 15) : resolve()); wait(); }).then(() => act(path, body));
  }
  let toastTimer = null, lastStatus = 0;
  // A toast goes away; every message stays in the list behind the top bar's message button: the editor's own
  // reports (ui/log, with a tool's output as detail) and the page's errors (kept here, for this page).
  const pageLog = [];
  function toast(text, error, fromEditor = false) {
    if (!text) return;
    const t = $('rl-toast'); t.textContent = text; t.className = 'on' + (error ? ' error' : '');
    clearTimeout(toastTimer); toastTimer = setTimeout(() => { t.className = ''; }, error ? 6000 : 3200);
    if (error && !fromEditor) { pageLog.push({id: 'p' + pageLog.length, time: Date.now() / 1000, text, error: true, detail: ''}); markLog(); }
  }
  let logSeen = 0, logCleared = 0, pageSeen = 0;
  function markLog() {
    const unseen = (ui && ui.last_error > logSeen) || pageLog.length > pageSeen;
    $('rl-log').classList.toggle('alert', !!unseen && $('rl-log-panel').hidden);
  }
  async function openLog() {
    const panel = $('rl-log-panel');
    let messages = [];
    try { messages = (await api('ui/log')).messages; } catch (error) { messages = []; }
    const all = [...messages.filter(m => m.id > logCleared), ...pageLog.filter(m => !m.cleared)]
      .sort((a, b) => b.time - a.time);
    const time = t => new Date(t * 1000).toLocaleTimeString('zh-CN', {hour12: false});
    panel.innerHTML = `<div class="bar"><b>消息</b><span class="grow"></span>
        <button class="rl-link" data-log="clear"${all.length ? '' : ' disabled'}>清空</button><button class="rl-icon sm" data-log="close">${I.x}</button></div>
      <div class="list">${all.length ? all.map((m, i) => `<div class="msg${m.error ? ' error' : ''}">
        <div class="head"><span class="t">${time(m.time)}</span><span class="text">${esc(m.text.replace(/^⚠️\s*/, ''))}</span></div>
        ${m.detail ? `<details><summary>详细输出</summary><pre>${esc(m.detail)}</pre>
          <button class="rl-link" data-log="copy" data-i="${i}">复制</button></details>` : ''}</div>`).join('')
        : '<div class="empty">还没有消息</div>'}</div>`;
    panel.hidden = false;
    panel.onclick = async event => {
      const b = event.target.closest('[data-log]'); if (!b) return;
      if (b.dataset.log === 'close') { panel.hidden = true; }
      else if (b.dataset.log === 'copy') {
        const m = all[+b.dataset.i];
        try { await navigator.clipboard.writeText(`${m.text}\n${m.detail}`); b.textContent = '已复制'; } catch { b.textContent = '复制失败'; }
      } else if (b.dataset.log === 'clear') {
        logCleared = Math.max(logCleared, ...messages.map(m => m.id));
        pageLog.forEach(m => { m.cleared = true; }); openLog();
      }
    };
    logSeen = Math.max(logSeen, (ui && ui.last_error) || 0); pageSeen = pageLog.length;
    markLog();
  }
  $('rl-log').onclick = () => ($('rl-log-panel').hidden ? openLog() : ($('rl-log-panel').hidden = true));
  $('rl-toast').onclick = () => openLog();

  // ------------------------------------------------------------------ dropdowns (one look for every choice)
  // dd() renders a button; the options live here until the panel is drawn again. A long list gets a search box.
  const dds = {};
  let ddCount = 0;
  function dd(options, value, pick, {placeholder = '', scope = 'i', search = options.length > 8, disabled = false} = {}) {
    const id = scope + (++ddCount), current = options.find(o => o[0] === value);
    dds[id] = {options, value, pick, search};
    return `<button class="rl-dd${current ? '' : ' empty'}" data-dd="${id}"${disabled ? ' disabled' : ''} title="${esc(current ? current[2] || current[1] : placeholder)}">
      <span>${esc(current ? current[1] : placeholder)}</span>${I.chev}</button>`;
  }
  const dropScope = scope => { for (const id of Object.keys(dds)) if (id.startsWith(scope)) delete dds[id]; };
  function openDd(button) {
    const entry = dds[button.dataset.dd]; if (!entry) return;
    const pop = $('rl-pop'), rect = button.getBoundingClientRect();
    let active = Math.max(0, entry.options.findIndex(o => o[0] === entry.value));
    const draw = words => {
      const list = entry.options.filter(o => !words || (o[1] + ' ' + o[0]).toLowerCase().includes(words));
      active = Math.min(active, Math.max(0, list.length - 1));
      pop.querySelector('.list').innerHTML = list.map((o, i) => `<button data-v="${esc(o[0])}" class="${o[0] === entry.value ? 'on' : ''}${i === active ? ' hot' : ''}"
        title="${esc(o[2] || '')}"><span>${esc(o[1])}</span>${o[0] === entry.value ? I.check : ''}</button>`).join('') || '<div class="none">没有匹配项</div>';
      return list;
    };
    pop.innerHTML = (entry.search ? `<label class="find">${I.search}<input placeholder="搜索"></label>` : '') + '<div class="list"></div>';
    pop.hidden = false;
    pop.style.minWidth = rect.width + 'px';
    let list = draw('');
    const place = () => {
      const below = innerHeight - rect.bottom - 8, height = Math.min(pop.scrollHeight, 340);
      pop.style.left = Math.min(rect.left, innerWidth - pop.offsetWidth - 8) + 'px';
      pop.style.top = (below >= height || below > rect.top ? rect.bottom + 4 : rect.top - height - 4) + 'px';
    };
    place();
    pop.querySelector('.on, .hot')?.scrollIntoView({block: 'nearest'});
    const input = pop.querySelector('input');
    const choose = value => { pop.hidden = true; button.focus(); if (value !== undefined && value !== entry.value) entry.pick(value); };
    pop.onclick = e => { const b = e.target.closest('button[data-v]'); if (b) choose(b.dataset.v); };
    pop.onkeydown = e => {
      if (e.key === 'Escape') { e.preventDefault(); choose(); }
      else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault(); active = (active + (e.key === 'ArrowDown' ? 1 : -1) + list.length) % Math.max(1, list.length);
        list = draw(input ? input.value.trim().toLowerCase() : ''); pop.querySelector('.hot')?.scrollIntoView({block: 'nearest'});
      } else if (e.key === 'Enter') { e.preventDefault(); if (list[active]) choose(list[active][0]); }
    };
    if (input) { input.oninput = () => { active = 0; list = draw(input.value.trim().toLowerCase()); }; input.focus(); }
    else pop.querySelector('.list button')?.focus();
  }
  document.addEventListener('click', event => {
    const button = event.target.closest('#rl-app .rl-dd');
    if (button && !button.disabled) { event.stopPropagation(); $('rl-pop').hidden && openDd(button); }
  }, true);
  document.addEventListener('mousedown', event => {
    if (!event.target.closest('#rl-pop') && !event.target.closest('.rl-dd')) $('rl-pop').hidden = true;
    if (!event.target.closest('#rl-menu')) $('rl-menu').hidden = true;
    if (!event.target.closest('#rl-log-panel, #rl-log, #rl-toast')) $('rl-log-panel').hidden = true;
  }, true);

  // ------------------------------------------------------------------ top bar
  let ui = null, shownCamera = null;
  function renderTop() {
    if (!ui) return;
    // Another editor process (restarted): viser's reconnect does not bring its scene back, so the page loads
    // afresh, the 3D view and the lists from the new process.
    if (renderTop.boot && ui.boot !== renderTop.boot) { location.reload(); return; }
    renderTop.boot = ui.boot;
    renderJobs();
    $('rl-project').innerHTML = `${esc(ui.project)}${I.chev}`;
    $('rl-save').classList.toggle('dirty', ui.dirty);
    $('rl-undo').disabled = !ui.undo; $('rl-redo').disabled = !ui.redo;
    $('rl-preview').hidden = !ui.command;
    $('rl-problem').hidden = !ui.problem; $('rl-problem').title = ui.problem || '';
    if ($('rl-pop').hidden) {
      dropScope('t');
      for (const name of ['physics', 'renderer']) {
        const box = $('rl-' + name), choice = ui.simulation[name];
        box.hidden = !choice;
        if (choice) box.querySelector('.slot').innerHTML = dd(choice.options.map(([key, label]) => [key, label.replace(/（.*）/, ''), label]),
          choice.value, value => act('ui/simulation', {name, value}), {scope: 't', search: false});
      }
    }
    document.querySelectorAll('#rl-tools [data-gizmo]').forEach(b => b.classList.toggle('on', b.dataset.gizmo === ui.gizmo));
    document.querySelector('#rl-tools [data-tool="snap"]')?.classList.toggle('on', !!ui.snap);
    document.body.classList.toggle('rl-picking', !!ui.picking);
    // An editor restarted under an open page builds viser's panels again: they stay off while the app shows.
    if (ui.classic && !document.body.classList.contains('rl-classic') && !renderTop.classicSent) {
      renderTop.classicSent = true;
      api('ui/classic', {on: false}).finally(() => { renderTop.classicSent = false; });
    }
    if (ui.status && ui.status.id !== lastStatus) {
      if (lastStatus) toast(ui.status.text, ui.status.error, true);
      lastStatus = ui.status.id;
      if (!$('rl-log-panel').hidden) openLog();
    }
    markLog();
    showCamera();
    fitTop();
    window.__rlinfOutline?.(ui.outline || []);  // the patched viser client (the editor page) draws it
    if (ui.parent_choice && ui.parent_choice.id !== renderTop.choice && $('rl-dialog').hidden) {
      renderTop.choice = ui.parent_choice.id;
      refreshInspector().then(() => parentDialog({target: ui.parent_choice.target, body: ui.parent_choice.body}));
    }
  }
  // Additions running in the background (an import converting, a SimReady download): what each does, for how
  // long, and a way to cancel a converter. The same place in the view shows where it will appear.
  function renderJobs() {
    const box = $('rl-jobs'), jobs = ui.jobs || [];
    box.hidden = !jobs.length;
    const clock = t => `${Math.floor(t / 60)}:${String(t % 60).padStart(2, '0')}`;
    box.innerHTML = jobs.map(j => `<div class="job"><span class="spin"></span><div class="what"><b>${esc(j.name)}</b>
        <span>${esc(j.message)}</span><div class="rl-bar${j.fraction === null ? ' busy' : ''}"><i style="width:${j.fraction === null ? 100 : Math.round(100 * j.fraction)}%"></i></div></div>
        <span class="t" title="已用时间">${clock(j.seconds)}</span>${j.cancel ? `<button class="rl-link" data-cancel="${esc(j.key)}">取消</button>` : ''}</div>`).join('');
  }
  $('rl-jobs').onclick = event => { const b = event.target.closest('[data-cancel]'); if (b) act('ui/cancel_job', {key: b.dataset.cancel}); };
  function showCamera() {
    const img = $('rl-cam'); if (!img || !ui) return;
    if (shownCamera !== ui.camera) { shownCamera = ui.camera; img.src = `${API}api/ui/camera.jpg?r=${ui.camera}`; }
  }
  $('rl-problem').onclick = () => toast(ui.problem, true);
  $('rl-save').onclick = () => act('ui/save', {});
  $('rl-undo').onclick = () => act('ui/undo', {});
  $('rl-redo').onclick = () => act('ui/redo', {});
  $('rl-preview').onclick = async () => {
    try { await navigator.clipboard.writeText(ui.command); } catch {
      const area = document.createElement('textarea'); area.value = ui.command; document.body.appendChild(area); area.select(); document.execCommand('copy'); area.remove();
    }
    toast(`已复制：${ui.command}`);
  };
  function classic(on) { document.body.classList.toggle('rl-classic', on); act('ui/classic', {on}); }
  $('rl-back').onclick = () => classic(false);
  $('rl-project').onclick = event => menu(event, [['新建项目…', newProject], ['打开项目…', openProject]], true);
  // Commands in the bar; when it runs out of room, labels go first, then commands (highest `p` first) move
  // into the ⋯ menu. `on` marks a switch.
  const CMDS = [
    {id: 'import', label: '导入', icon: I.upload, p: 1, title: '导入文件或文件夹（也可以直接拖进窗口）',
     run: event => menu(event, [['导入文件…', () => $('rl-files').click()], ['导入文件夹…', () => $('rl-folder').click()]], true)},
    {id: 'reload', label: '重新加载', icon: I.rotate, p: 5, title: '放弃未保存的修改，重新读取 scene.yaml', run: () => act('ui/reload', {})},
    {id: 'refresh', label: '刷新网格', icon: I.cube, p: 6, title: '在后台重新导出原生网格', run: () => act('ui/refresh_native', {})},
    {id: 'textures', label: '贴图', icon: I.image, p: 3, gap: true, title: '显示导出的贴图与表面颜色',
     on: () => ui.textures, run: () => act('ui/textures', {on: !ui.textures})},
    {id: 'style', label: '简洁风格', icon: I.sun, p: 2, title: '仿 Newton OpenGL 查看器：渐变天空、棋盘地面、主光源阴影；关闭后用 viser 默认的网格与环境光',
     on: () => ui.style === 'newton', run: () => act('ui/style', {style: ui.style === 'newton' ? 'default' : 'newton'})},
  ];
  $('rl-cmds').innerHTML = CMDS.map(c => `<button class="rl-tb${c.gap ? ' gap' : ''}" data-cmd="${c.id}" data-p="${c.p}" title="${esc(c.title)}">${c.icon}<span>${esc(c.label)}</span></button>`).join('');
  $('rl-cmds').onclick = event => { const b = event.target.closest('[data-cmd]'); if (b) CMDS.find(c => c.id === b.dataset.cmd).run(event); };
  $('rl-more').onclick = event => menu(event, CMDS.filter(c => $('rl-cmds').querySelector(`[data-cmd="${c.id}"]`).hidden)
    .map(c => [c.label, () => c.run(event), c.on ? c.on() : false]), true);
  function fitTop() {
    const top = $('rl-top'), buttons = [...$('rl-cmds').children];
    buttons.forEach(b => { b.hidden = false; b.classList.toggle('on', !!(ui && CMDS.find(c => c.id === b.dataset.cmd).on?.())); });
    $('rl-more').hidden = true; top.classList.remove('compact');
    const over = () => top.scrollWidth > top.clientWidth + 1;
    if (over()) top.classList.add('compact');
    for (const b of buttons.sort((a, b) => b.dataset.p - a.dataset.p)) {
      if (!over()) break;
      b.hidden = true; $('rl-more').hidden = false;
    }
  }
  new ResizeObserver(fitTop).observe($('rl-top'));
  // Side panels: drag the inner edge to resize, double-click it, the bar buttons or T / N to hide; kept per browser.
  const sides = JSON.parse(localStorage.getItem('rl-sides') || '{}');
  const SIDE = {left: [296, 200, 640], right: [316, 220, 680]};
  function layoutSides() {
    for (const side of ['left', 'right']) {
      const [initial, low, high] = SIDE[side], s = sides[side] || {};
      const width = Math.min(Math.max(s.width || initial, low), Math.min(high, innerWidth * .45));
      document.documentElement.style.setProperty(`--rl-${side}`, s.off ? '0px' : width + 'px');
      document.body.classList.toggle(`rl-${side}-off`, !!s.off);
      $(`rl-side-${side}`).classList.toggle('on', !s.off);
    }
    localStorage.setItem('rl-sides', JSON.stringify(sides));
  }
  function toggleSide(side) { sides[side] = {...sides[side], off: !(sides[side] || {}).off}; layoutSides(); }
  $('rl-side-left').onclick = () => toggleSide('left');
  $('rl-side-right').onclick = () => toggleSide('right');
  document.querySelectorAll('.rl-resize').forEach(handle => {
    const side = handle.dataset.side;
    handle.ondblclick = () => toggleSide(side);
    handle.onpointerdown = event => {
      event.preventDefault(); handle.setPointerCapture(event.pointerId); handle.classList.add('drag');
      const move = e => {
        sides[side] = {width: side === 'left' ? e.clientX : innerWidth - e.clientX, off: false};
        layoutSides();
      };
      const up = () => { handle.classList.remove('drag'); handle.removeEventListener('pointermove', move); handle.removeEventListener('pointerup', up); };
      handle.addEventListener('pointermove', move); handle.addEventListener('pointerup', up);
    };
  });
  addEventListener('resize', layoutSides);
  layoutSides();
  // Canvas pixels per CSS pixel: the editor sizes the fixed on-screen gizmo with it (_gizmo_unit).
  setInterval(() => {
    const canvas = [...document.querySelectorAll('canvas')].sort((a, b) => b.clientWidth * b.clientHeight - a.clientWidth * a.clientHeight)[0];
    if (!canvas || !canvas.clientHeight) return;
    const ratio = Math.round(canvas.height / canvas.clientHeight * 100) / 100;
    if (ratio !== fitTop.ratio) { fitTop.ratio = ratio; api('ui/viewport', {pixel_ratio: ratio}).catch(() => { fitTop.ratio = null; }); }
  }, 2000);
  // The axes in the view's corner (drawn by the editor) take clicks and drags: snap to an axis, or orbit.
  (() => {
    const nav = $('rl-nav');
    const nearest = (px, py) => {
      if (!ui || !ui.view) return null;
      const [w, x, y, z] = ui.view;
      const rows = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]];
      let best = null;
      rows.forEach((row, i) => {  // world axis i in the camera frame: row i of the rotation
        for (const [sign, scale] of [['+', 1.15], ['-', -1]]) {
          const d = Math.hypot(px - row[0] * 26 * scale, py - row[1] * 26 * scale);
          if (d < 14 && (!best || d < best.d)) best = {d, axis: sign + 'xyz'[i]};
        }
      });
      return best && best.axis;
    };
    nav.addEventListener('pointerdown', event => {
      event.preventDefault(); event.stopPropagation(); nav.setPointerCapture(event.pointerId);
      const box = nav.getBoundingClientRect(), start = [event.clientX, event.clientY];
      let last = start, moved = false, owed = [0, 0], busy = false;
      // The pointer moves faster than the view can be sent: what is not sent yet adds up, nothing is dropped.
      const flush = () => {
        if (busy || (!owed[0] && !owed[1])) return;
        const [dx, dy] = owed; owed = [0, 0]; busy = true;
        api('ui/view', {dx, dy}).catch(() => {}).finally(() => { busy = false; requestAnimationFrame(flush); });
      };
      const move = e => {
        if (!moved && Math.hypot(e.clientX - start[0], e.clientY - start[1]) < 3) return;
        moved = true; nav.classList.add('drag');
        owed = [owed[0] + e.clientX - last[0], owed[1] + e.clientY - last[1]]; last = [e.clientX, e.clientY];
        flush();
      };
      const up = e => {
        nav.removeEventListener('pointermove', move); nav.removeEventListener('pointerup', up); nav.classList.remove('drag');
        if (moved) return;
        const scale = 84 / box.width, axis = nearest((e.clientX - box.left) * scale - 42, (e.clientY - box.top) * scale - 42);
        if (axis) act('ui/view', {axis});
      };
      nav.addEventListener('pointermove', move); nav.addEventListener('pointerup', up);
    });
  })();
  $('rl-tools').onclick = event => {
    const b = event.target.closest('button'); if (!b) return;
    if (b.dataset.gizmo) act('ui/gizmo', {mode: b.dataset.gizmo});
    else if (b.dataset.tool === 'snap') act('ui/snap', {on: !(ui && ui.snap)});
    else if (b.dataset.tool === 'drop') act('ui/drop', {});
    else act('ui/focus', {});
  };

  // The patched viewer's scale handles (tools/viser_client): factors since the drag started, sent as they come.
  window.__rlinfScale = (factors, phase) => {
    if (phase === 'update') sendLive('ui/gizmo_scale', {factors, phase});
    else if (phase === 'end') finish('ui/gizmo_scale', {factors: null, phase});
    else act('ui/gizmo_scale', {factors, phase});
  };

  // ------------------------------------------------------------------ dialogs (import, projects)
  function dialog(title, body, actions, onOpen) {
    const box = $('rl-dialog').querySelector('.box');
    box.classList.remove('wide', 'pick');
    box.innerHTML = `<h3>${esc(title)}<button class="rl-icon" data-close>${I.x}</button></h3><div class="content">${body}</div>
      <div class="actions">${actions.map(([label, kind], i) => `<button data-a="${i}" class="${kind || ''}">${esc(label)}</button>`).join('')}</div>`;
    $('rl-dialog').hidden = false;
    box.onclick = e => {
      if (e.target.closest('[data-close]')) return closeDialog();
      const b = e.target.closest('[data-a]'); if (b && !b.disabled) actions[+b.dataset.a][2](box);
    };
    onOpen && onOpen(box);
  }
  // The server says what deleting the selection also changes (sensors, observations, control); delete on confirm.
  const inline = text => esc(text).replace(/\*\*(.+?)\*\*/g, '<b>$1</b>').replace(/`(.+?)`/g, '<code>$1</code>');
  async function confirmDelete() {
    let plan;
    try { plan = await api('ui/delete'); } catch (error) { return toast(error.message, true); }
    dialog(plan.paths.length > 1 ? `删除 ${plan.paths.length} 个实体` : '删除实体',
      `<div class="rl-notes">${plan.lines.map(line => `<p>${inline(line)}</p>`).join('')}</div>`,
      [['取消', '', closeDialog], ['删除', 'danger', () => { closeDialog(); act('ui/delete', {paths: plan.paths}); }]],
      box => box.querySelector('.danger').focus());
  }
  const closeDialog = () => {
    $('rl-dialog').hidden = true; dropScope('g');
    if (ui && ui.start) { ui.start = false; api('ui/start_done', {}).catch(() => {}); }  // the start screen, closed
    if (ui && ui.parent_choice) api('ui/parent_choice', {}).catch(() => {});  // a pick in the view, not taken
  };
  // Esc closes what is open over the page (a dialog, a list, a menu, a pick) and stops there: viser's own
  // Esc would also clear the selection.
  addEventListener('keydown', event => {
    if (event.key !== 'Escape') return;
    const open = !$('rl-dialog').hidden || !$('rl-pop').hidden || !$('rl-menu').hidden || (ui && ui.picking);
    if (!open) return;
    event.stopImmediatePropagation(); event.preventDefault();
    if (!$('rl-pop').hidden) { $('rl-pop').hidden = true; return; }
    $('rl-menu').hidden = true;
    if (!$('rl-dialog').hidden) closeDialog();
    else if (ui && ui.picking) setPicking(false);
  }, true);
  // Delete asks in the app's dialog; viser's own Delete command would open its own confirmation too.
  addEventListener('keydown', event => {
    if (!['Delete', 'Backspace'].includes(event.key) || event.target.matches?.('input, textarea, select, [contenteditable]')) return;
    event.stopImmediatePropagation(); event.preventDefault();
    if ($('rl-dialog').hidden && !event.ctrlKey && !event.metaKey && !event.altKey && !event.shiftKey) confirmDelete();
  }, true);
  $('rl-dialog').addEventListener('mousedown', e => { if (e.target.id === 'rl-dialog') closeDialog(); });

  async function newProject() {
    const data = await api('ui/projects');
    let template = data.templates[0][0];
    dialog('新建项目', `<div class="rl-field"><span class="rl-lbl">名称</span><input id="rl-np-name" value="my_scene" spellcheck="false"></div>
      <div class="rl-field"><span class="rl-lbl">模板</span><span id="rl-np-template"></span></div>`,
      [['取消', '', closeDialog], ['新建', 'primary', async () => {
        const name = $('rl-np-name').value.trim();
        if (await act('ui/new_project', {name, template})) closeDialog();
      }]], box => {
        const slot = () => { $('rl-np-template').innerHTML = dd(data.templates, template, v => { template = v; slot(); }, {scope: 'g'}); };
        slot(); box.querySelector('input').select();
      });
  }
  // Opened without a project (run.sh edit): start a new one from a template, open one, or look around the blank one.
  let startShown = false;
  async function startDialog() {
    startShown = true;
    const data = await api('ui/projects');
    let template = data.templates[0][0];
    const create = async () => {
      const name = $('rl-st-name').value.trim();
      if (await act('ui/new_project', {name, template})) { ui.start = false; closeDialog(); }
    };
    dialog('开始', `<div class="rl-start">
        <section><h4>新建项目</h4>
          <div class="rl-field"><span class="rl-lbl">名称</span><input id="rl-st-name" value="my_scene" spellcheck="false"></div>
          <div class="rl-tpls" id="rl-st-tpl">${data.templates.map(([id, label]) =>
            `<button data-t="${esc(id)}" class="${id === template ? 'on' : ''}"><b>${esc(id === 'blank' ? '空白' : id)}</b><span>${esc(label.replace(/^空白：/, ''))}</span></button>`).join('')}</div>
        </section>
        <section><h4>打开项目</h4><label class="find">${I.search}<input id="rl-st-find" placeholder="搜索项目"></label>
          <div class="rl-list" id="rl-st-list"></div></section></div>`,
      [['先在空白场景里看看', '', closeDialog], ['新建', 'primary', create]], box => {
        box.classList.add('wide');
        $('rl-st-tpl').onclick = e => { const b = e.target.closest('[data-t]'); if (!b) return; template = b.dataset.t;
          $('rl-st-tpl').querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b)); };
        const draw = words => { $('rl-st-list').innerHTML = data.projects.filter(p => !p.current && (!words || p.name.toLowerCase().includes(words)))
          .map(p => `<button data-path="${esc(p.path)}">${svg(ICON.folder)}<span>${esc(p.name)}</span></button>`).join(''); };
        draw('');
        $('rl-st-find').oninput = e => draw(e.target.value.trim().toLowerCase());
        $('rl-st-list').onclick = async e => { const b = e.target.closest('button[data-path]');
          if (b && await act('ui/open_project', {path: b.dataset.path})) { ui.start = false; closeDialog(); } };
        $('rl-st-name').onkeydown = e => { if (e.key === 'Enter') create(); };
        $('rl-st-name').select();
      });
  }
  async function openProject() {
    const data = await api('ui/projects');
    dialog('打开项目', `<label class="find">${I.search}<input id="rl-op-find" placeholder="搜索项目"></label><div class="rl-list" id="rl-op-list"></div>`,
      [['取消', '', closeDialog]], box => {
        const draw = words => { $('rl-op-list').innerHTML = data.projects.filter(p => !words || p.name.toLowerCase().includes(words))
          .map(p => `<button data-path="${esc(p.path)}" class="${p.current ? 'on' : ''}">${svg(ICON.folder)}<span>${esc(p.name)}</span>${p.current ? '<i>当前</i>' : ''}</button>`).join(''); };
        draw('');
        $('rl-op-find').oninput = e => draw(e.target.value.trim().toLowerCase());
        $('rl-op-find').focus();
        $('rl-op-list').onclick = async e => { const b = e.target.closest('button[data-path]'); if (b && !b.classList.contains('on') && await act('ui/open_project', {path: b.dataset.path})) closeDialog(); };
      });
  }

  // Import: files, a folder or a zip; dropped on the window or chosen. Uploaded file by file (a URDF keeps its
  // meshes beside it), then converted and added by the editor (shared/asset_import.py).
  function guessKind(files) {
    const names = files.map(f => f.path.toLowerCase());
    if (names.some(n => n.endsWith('.urdf') || n.endsWith('.urdf.xacro') || n.endsWith('.mjcf'))) return 'robot';
    const size = files.reduce((sum, f) => sum + f.file.size, 0);
    return size > 80e6 ? 'scene' : 'object';
  }
  const TEXT = /\.(urdf|xacro|xml|mjcf|yaml|yml|mtl|gltf)$/i, TEXT_LIMIT = 2 << 20;  // shared/asset_import.py TEXT_SUFFIXES
  const megabytes = size => size < 1e6 ? `${Math.max(1, Math.round(size / 1e3))} KB` : `${(size / 1e6).toFixed(size > 1e7 ? 0 : 1)} MB`;
  const MAIN = /\.(urdf|urdf\.xacro|xml|mjcf|usd|usda|usdc|usdz|obj|stl|ply|glb|gltf|fbx|dae|off|3mf|blend|abc|zip)$/i;
  // A xacro whose top level only defines macros, properties and arguments describes no robot by itself (lite6.urdf.xacro,
  // used by xarm_device.urdf.xacro): it is not offered as the file to import.
  async function onlyMacros(f) {
    const doc = new DOMParser().parseFromString(await f.file.text(), 'application/xml');
    const top = [...(doc.documentElement ? doc.documentElement.children : [])];
    return !doc.querySelector('parsererror') && top.every(e => /^xacro:(macro|property|include|arg)$/.test(e.tagName));
  }
  async function openImport(files) {
    files = files.filter(f => !/(^|\/)\./.test(f.path));
    if (!files.length) return;
    const macros = new Set();
    for (const f of files.filter(f => /\.xacro$/i.test(f.path) && f.file.size < (2 << 20))) if (await onlyMacros(f)) macros.add(f);
    if (macros.size) files = files.map(f => (macros.has(f) ? {...f, macro: true} : f));
    // What imports by itself from a chosen folder: robot descriptions (URDF, a top-level xacro, an .xml that is a
    // MuJoCo model) and USD files. Several of them (a repository, r1pro.urdf beside r1pro_original.urdf, the same
    // name in two folders) are picked from a tree of the folder.
    const isMjcf = async f => /\.mjcf$/i.test(f.path) || (/\.xml$/i.test(f.path) && !/(^|\/)package\.xml$/i.test(f.path)
      && f.file.size < (8 << 20) && (await f.file.slice(0, 4096).text()).includes('<mujoco'));
    const candidates = [];
    for (const f of files) if (!f.macro && (/\.(urdf|urdf\.xacro|usd|usda|usdc|usdz)$/i.test(f.path) || await isMjcf(f))) candidates.push(f);
    // A robot description first (a ROS package keeps its .urdf.xacro below meshes and launch files), else the shallowest, largest file.
    const rank = f => /\.(urdf|xacro|mjcf|xml)$/i.test(f.path) ? 0 : 1;
    const main = (candidates.length ? candidates : files.filter(f => MAIN.test(f.path) && !f.macro)).sort((a, b) => rank(a) - rank(b)
      || a.path.split('/').length - b.path.split('/').length || a.path.length - b.path.length || b.file.size - a.file.size)[0];
    if (!main) { toast('没有可导入的文件：URDF、MJCF、USD、OBJ、FBX、glTF、STL、PLY、DAE 或 zip', true); return; }
    let kind = guessKind(files), fixed = kind === 'robot';
    let chosen = candidates.length > 1 ? main.path : null;
    const size = files.reduce((sum, f) => sum + f.file.size, 0);
    const labelOf = path => path.split('/').pop().replace(/(\.urdf)?\.[^.]+$/, '');
    const label = labelOf(main.path);
    const isXacro = () => /\.xacro$/i.test(chosen || main.path);
    // From a chosen folder only what the description (or a USD) uses is sent: its text files first, then the files
    // the editor finds it names (api/import/needs). Loose meshes still go together: several make one object.
    const selective = () => files.length > 1 && /\.(urdf|xacro|xml|mjcf|usd|usda|usdc|usdz)$/i.test(chosen || main.path);
    const body = () => `<div class="rl-file">${I.upload}<div><b>${esc(main.path.split('/').pop())}</b>
        <span>${files.length > 1 ? `共 ${files.length} 个文件 · ` : ''}${megabytes(size)}${selective() ? ' · 只上传它用到的' : ''}</span></div></div>
      ${chosen ? `<div class="rl-im-pick"><label class="find">${I.search}<input id="rl-im-find" placeholder="在 ${candidates.length} 个可导入的文件中搜索" spellcheck="false"></label>
        <div class="rl-ftree" id="rl-im-tree"></div></div>` : ''}
      <div class="rl-field" id="rl-im-xacro-line"${isXacro() ? '' : ' hidden'} title="xacro 的参数，写成 名字:=值、用空格分开；不写就用文件里的默认值">
        <span class="rl-lbl">xacro 参数</span><input id="rl-im-xacro" placeholder="如 dof:=7 add_gripper:=true" spellcheck="false"></div>
      <div class="rl-field"><span class="rl-lbl">名称</span><input id="rl-im-name" value="${esc(label)}" spellcheck="false"></div>
      <div class="rl-field"><span class="rl-lbl">类型</span><div class="rl-seg" id="rl-im-kind">${KINDS.map(([k, l]) => `<button data-k="${k}" class="${k === kind ? 'on' : ''}">${l}</button>`).join('')}</div></div>
      <div class="rl-line" id="rl-im-fixed-line"${kind === 'scene' ? ' hidden' : ''}><span class="grow">固定根部（不会倒下）</span><span class="rl-toggle${fixed ? ' on' : ''}" id="rl-im-fixed"></span></div>
      <div class="rl-bar" hidden><i></i></div><p class="rl-im-note" id="rl-im-note"></p>`;
    dialog('导入到场景', body(), [['取消', '', closeDialog], ['导入', 'primary', async box => {
      const go = box.querySelector('[data-a="1"]'); go.disabled = true;
      const bar = box.querySelector('.rl-bar'), fill = bar.querySelector('i'); bar.hidden = false;
      const batch = Array.from(crypto.getRandomValues(new Uint8Array(12)), b => (b % 36).toString(36)).join('');
      const note = text => { $('rl-im-note').textContent = text; };
      const send = async list => {
        const total = list.reduce((sum, f) => sum + f.file.size, 0) || 1;
        let sent = 0;
        for (const f of list) {
          await new Promise((resolve, reject) => {
            const request = new XMLHttpRequest();
            request.open('POST', `${API}api/import/file?batch=${batch}&path=${encodeURIComponent(f.path)}`);
            request.upload.onprogress = e => { fill.style.width = `${100 * (sent + e.loaded) / total}%`; };
            request.onload = () => (request.status === 200 ? resolve() : reject(new Error(JSON.parse(request.responseText || '{}').error || request.statusText)));
            request.onerror = () => reject(new Error('上传中断'));
            request.send(f.file);
          });
          sent += f.file.size;
        }
      };
      const main_ = chosen || main.path, xacro = isXacro() ? $('rl-im-xacro').value.trim() : '';
      try {
        if (selective()) {
          const texts = files.filter(f => TEXT.test(f.path) && f.file.size <= TEXT_LIMIT), first = new Set(texts.map(f => f.path));
          note(`读取描述文件（${texts.length} 个）…`);
          await send(texts);
          note('找出它用到的文件…');
          const {needed} = await api('import/needs', {batch, main: main_, xacro, files: files.map(f => f.path)});
          const wanted = new Set(needed), rest = files.filter(f => wanted.has(f.path) && !first.has(f.path));
          note(`上传用到的 ${rest.length} 个文件（${megabytes(rest.reduce((sum, f) => sum + f.file.size, 0))}，共 ${files.length} 个）…`);
          fill.style.width = '0';
          await send(rest);
        } else await send(files);
        await api('import/finish', {batch, kind, fixed, name: $('rl-im-name').value.trim(), main: chosen || (selective() ? main_ : null), xacro});
        closeDialog(); kick();
      } catch (error) { toast(error.message, true); go.disabled = false; bar.hidden = true; note(''); }
    }]], box => {
      box.querySelector('#rl-im-kind').onclick = e => { const b = e.target.closest('[data-k]'); if (!b) return; kind = b.dataset.k;
        box.querySelectorAll('#rl-im-kind button').forEach(x => x.classList.toggle('on', x === b)); $('rl-im-fixed-line').hidden = kind === 'scene'; };
      $('rl-im-fixed').onclick = e => { fixed = !fixed; e.target.classList.toggle('on', fixed); };
      if (chosen) {
        // The folder as a tree of what imports by itself: folders fold (a chain of single folders is one row),
        // the chosen file is marked; a search lists the matching files with their folders.
        box.classList.add('pick');
        const root = {dirs: new Map(), files: []};
        for (const f of candidates) {
          let node = root;
          const parts = f.path.split('/');
          for (const part of parts.slice(0, -1)) {
            if (!node.dirs.has(part)) node.dirs.set(part, {dirs: new Map(), files: [], count: 0});
            node = node.dirs.get(part);
          }
          node.files.push(f.path);
        }
        const count = node => (node.count = node.files.length + [...node.dirs.values()].reduce((n, d) => n + count(d), 0));
        count(root);
        const open = new Set();
        const reveal = path => { const parts = path.split('/'); for (let i = 1; i < parts.length; i++) open.add(parts.slice(0, i).join('/')); };
        if (candidates.length <= 15) candidates.forEach(f => reveal(f.path)); else reveal(chosen);
        const tag = path => /\.xacro$/i.test(path) ? 'xacro' : /\.urdf$/i.test(path) ? 'URDF' : /\.usd[acz]?$/i.test(path) ? 'USD' : 'MJCF';
        const fileRow = (path, depth, shown) => `<div class="f${path === chosen ? ' on' : ''}" data-file="${esc(path)}" style="padding-left:${8 + depth * 14}px"
            title="${esc(path)}"><span class="t">${tag(path)}</span>${shown}</div>`;
        const rows = (node, prefix, depth) => {
          const out = [];
          for (const [name, child] of [...node.dirs].sort((a, b) => a[0].localeCompare(b[0]))) {
            let label = name, path = prefix + name, inner = child;
            while (inner.dirs.size === 1 && !inner.files.length) {  // a/b/c: one row for a chain of single folders
              const [next, deeper] = [...inner.dirs][0];
              label += '/' + next; path += '/' + next; inner = deeper;
            }
            const on = open.has(path);
            out.push(`<div class="d" data-dir="${esc(path)}" style="padding-left:${8 + depth * 14}px"><span class="caret${on ? ' open' : ''}">${I.caret}</span>
              ${svg(ICON.folder)}<span class="n">${esc(label)}</span><i>${inner.count}</i></div>`);
            if (on) out.push(...rows(inner, path + '/', depth + 1));
          }
          for (const path of node.files.sort()) out.push(fileRow(path, depth, esc(path.split('/').pop())));
          return out;
        };
        const draw = () => {
          const words = $('rl-im-find').value.trim().toLowerCase();
          $('rl-im-tree').innerHTML = words
            ? candidates.filter(f => f.path.toLowerCase().includes(words)).map(f => fileRow(f.path, 0,
                `${esc(f.path.split('/').pop())}<em>${esc(f.path.split('/').slice(0, -1).join('/'))}</em>`)).join('') || '<div class="none">没有匹配的文件</div>'
            : rows(root, '', 0).join('');
          $('rl-im-tree').querySelector('.f.on')?.scrollIntoView({block: 'nearest'});
        };
        const pick = path => {
          const name = $('rl-im-name');  // the name follows the file, unless typed
          if (name.value.trim() === labelOf(chosen)) name.value = labelOf(path);
          chosen = path;
          box.querySelector('.rl-file b').textContent = path.split('/').pop();
          $('rl-im-xacro-line').hidden = !isXacro();
          draw();
        };
        $('rl-im-tree').onclick = e => {
          const dir = e.target.closest('[data-dir]'), file = e.target.closest('[data-file]');
          if (dir) { open.has(dir.dataset.dir) ? open.delete(dir.dataset.dir) : open.add(dir.dataset.dir); draw(); }
          else if (file) pick(file.dataset.file);
        };
        $('rl-im-find').oninput = draw;
        draw();
      }
      $('rl-im-name').select();
    });
  }
  const picked = input => openImport([...input.files].map(file => ({file, path: file.webkitRelativePath || file.name})));
  $('rl-files').onchange = e => { picked(e.target); e.target.value = ''; };
  $('rl-folder').onchange = e => { picked(e.target); e.target.value = ''; };
  $('rl-import').onclick = event => menu(event, [['选择文件（可多选或 zip）…', () => $('rl-files').click()], ['选择文件夹（URDF 连同网格）…', () => $('rl-folder').click()]], true);
  async function entries(items) {
    const files = [];
    const walk = async (entry, prefix) => {
      if (entry.isFile) files.push({file: await new Promise((ok, no) => entry.file(ok, no)), path: prefix + entry.name});
      else if (entry.isDirectory) {
        const reader = entry.createReader();
        for (;;) { const batch = await new Promise((ok, no) => reader.readEntries(ok, no)); if (!batch.length) break;
          for (const child of batch) await walk(child, prefix + entry.name + '/'); }
      }
    };
    for (const entry of items.map(item => item.webkitGetAsEntry && item.webkitGetAsEntry()).filter(Boolean)) await walk(entry, '');
    return files;
  }
  let dragDepth = 0;
  const external = e => [...(e.dataTransfer?.types || [])].includes('Files');
  addEventListener('dragenter', e => { if (external(e)) { dragDepth++; $('rl-dropzone').hidden = false; } });
  addEventListener('dragleave', e => { if (external(e) && --dragDepth <= 0) { dragDepth = 0; $('rl-dropzone').hidden = true; } });
  addEventListener('dragover', e => { if (external(e)) { e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; } });
  addEventListener('drop', async e => {
    if (!external(e)) return;
    e.preventDefault(); dragDepth = 0; $('rl-dropzone').hidden = true;
    openImport(await entries([...e.dataTransfer.items]));
  });

  // ------------------------------------------------------------------ menus (the tree and the 3D view share them)
  function menu(event, items, below, at) {
    event?.preventDefault?.(); event?.stopPropagation?.();
    const box = $('rl-menu');
    box.innerHTML = items.map((item, i) => item === null ? '<hr>' :
      `<button data-i="${i}" class="${item[3] || ''}${item[2] ? ' check' : ''}">${esc(item[0])}${item[4] ? `<kbd>${esc(item[4])}</kbd>` : ''}</button>`).join('');
    box.hidden = false;
    const rect = at || (below ? event.currentTarget.getBoundingClientRect() : {left: event.clientX, bottom: event.clientY});
    box.style.left = Math.min(rect.left, innerWidth - box.offsetWidth - 8) + 'px';
    box.style.top = Math.min(rect.bottom + (below ? 4 : 0), innerHeight - box.offsetHeight - 8) + 'px';
    box.onclick = e => { const b = e.target.closest('button[data-i]'); if (!b) return; box.hidden = true; items[+b.dataset.i][1](); };
  }
  // Sensors added on an entity (at a link when one was clicked): camera, contact, height scan, frame.
  function sensorItems(path, body) {
    return ((ui && ui.sensors) || []).map(([factory, label]) => [`添加${label}${body ? `（${body}）` : ''}`,
      () => act('ui/add_sensor', {factory, target: path, body: body || null})]);
  }
  function entityMenu(r, body) {
    if (String(r.id).startsWith('group:'))
      return [['聚焦', () => act('ui/focus', {}), false, '', 'F'], ['解散组合（保留位置）', () => act('ui/ungroup', {})]];
    const many = tree.rows.filter(x => x.selected && x.id.startsWith('entity:')).length > 1;
    return [['聚焦', () => act('ui/focus', {}), false, '', 'F'], ['落到下方表面', () => act('ui/drop', {}), false, '', 'End'],
      [r.hidden ? '显示' : '隐藏', () => act('ui/hidden', {}), false, '', 'H'],
      ['重命名', () => { const input = $('rl-name'); if (input) { input.focus(); input.select(); } }, false, '', 'F2'], null,
      ['设父物体：点选…', () => setPicking(true), false, '', 'P'], ...(r.note ? [['解除父物体', () => act('ui/parent', {target: null}), false, '', 'Alt+P']] : []),
      ...(many ? [['组成布局组合', () => act('ui/group', {}), false, '', 'Ctrl+G']] : []), null,
      ...(['robot', 'box'].includes(r.icon) ? [...sensorItems(r.id.slice(7), body), null] : []),
      ['删除', confirmDelete, false, 'danger', 'Del']];
  }

  // ------------------------------------------------------------------ tree
  let tree = {revision: null, rows: [], lens: 'scene', role: null, roles: [], train: null, counts: {}};
  const objects = {};
  const store = (() => { try { return localStorage; } catch { return null; } })();
  const opened = new Set(JSON.parse((store && store.getItem('rl.opened')) || '[]'));
  const saveOpened = () => store && store.setItem('rl.opened', JSON.stringify([...opened]));
  const listed = {}, PAGE = 200;
  const reading = () => tree.lens === 'observe' && tree.role;

  function renderReads() {
    $('rl-reads').innerHTML = tree.roles.map(role => {
      const on = reading() === role;
      return `<button data-role="${role}" class="${on ? 'on' : ''}" style="${on ? `background:${roleVar(role)}` : ''}"
        title="在场景中标出 ${esc(roleName(role))} 读取的物体；右键设为训练"><i style="background:${roleVar(role)}"></i>${esc(roleName(role))}${tree.train === role ? ' 🔥' : ''}<b>${tree.counts[role] || 0}</b></button>`;
    }).join('');
  }
  $('rl-reads').onclick = event => { const b = event.target.closest('button'); if (!b) return;
    act('outliner/lens', reading() === b.dataset.role ? {lens: 'scene'} : {lens: 'observe', role: b.dataset.role}); };
  $('rl-reads').oncontextmenu = event => { const b = event.target.closest('button'); if (!b) return;
    menu(event, [['训练这个组件', () => act('outliner/train', {role: b.dataset.role}), tree.train === b.dataset.role]]); };

  function marks(r) {
    const uses = r.uses || {}, out = [], role = reading();
    if (role) {
      const n = uses[role] || 0;
      if (n) out.push(`<span class="rl-count" style="background:${roleVar(role)}" title="${esc(roleName(role))} 读它的 ${n} 个量">${n}</span>`);
    } else {
      // Which layer reads it, as the initial of its name (P / E, or the layers' own names).
      for (const role of tree.roles) if (uses[role]) out.push(`<span class="rl-reads" style="background:${roleVar(role)}"
        title="${esc(roleName(role))} 读它的 ${uses[role]} 个量">${esc(roleMark(role))}</span>`);
    }
    if (r.mode === 'none') out.push('<span class="rl-mini" title="不受策略控制（control.modes: none）">⊘</span>');
    if (r.note) out.push(`<span class="rl-mini" title="${esc(r.note)}">${I.link}</span>`);
    if (r.motion) out.push(`<span class="rl-mini" title="${esc(r.motion)}">${I.motion}</span>`);
    if (r.changed) out.push('<span class="rl-dot unsaved" title="未保存"></span>');
    return out.join('');
  }
  function switches(r) {
    const sw = [];
    sw.push(r.selectable === null ? '<span class="rl-sw blank"></span>' :
      `<button class="rl-sw${r.selectable ? '' : ' set'}" data-what="selectable" title="${r.selectable ? '视图中可点选；点击锁定' : '已锁定；点击解锁'}">${r.selectable ? I.pointer : I.lock}</button>`);
    sw.push(!r.physics ? '<span class="rl-sw blank"></span>' :
      `<button class="rl-sw${r.physics.mode === 'dynamic' || r.physics.mode === 'static' ? '' : ' set'}" data-what="physics" title="${esc(r.physics.label + '：' + r.physics.hint)}">${I[r.physics.mode]}</button>`);
    sw.push(r.render === null ? '<span class="rl-sw blank"></span>' :
      `<button class="rl-sw${r.render ? '' : ' set'}" data-what="render" title="${r.render ? '出现在相机画面；点击隐藏' : '不出现在相机画面'}">${r.render ? I.render : I.renderOff}</button>`);
    sw.push(`<button class="rl-sw eye${r.hidden ? ' set' : ''}" data-what="hidden" title="编辑器中显示 H">${r.hidden ? I.eyeOff : I.eye}</button>`);
    return sw.join('');
  }
  // Every row with something under it folds: a scene's objects or a robot's links (folded until opened: key
  // `id` / `j:id`), and entities placed under it (a group's members, a camera on a robot; open until folded: `!id`).
  const foldKey = r => r.objects ? r.id : (r.joints && r.joints.length) ? `j:${r.id}` : `!${r.id}`;
  const isOpen = r => { const key = foldKey(r); return key.startsWith('!') ? !opened.has(key) : opened.has(key); };
  function entityRow(r, words, kids) {
    const sub = r.objects || (r.joints && r.joints.length) || kids, key = foldKey(r);
    const caret = sub ? `<span class="caret${isOpen(r) ? ' open' : ''}" data-open="${esc(key)}">${I.caret}</span>` : '<span class="caret none"></span>';
    const dim = r.hidden || (reading() && !(r.uses || {})[reading()]);
    return `<div class="rl-row${r.selected && !r.selected_object && !r.selected_joint ? ' sel' : ''}${r.selected && !r.active ? ' also' : ''}${dim ? ' dim' : ''}"
        data-id="${esc(r.id)}" draggable="true" title="${esc(r.hint || '')}" style="padding-left:${2 + (words.length ? 0 : r.depth) * 14}px">
      ${caret}<span class="ico">${svg(ICON[r.icon] || ICON.box)}</span><span class="name">${esc(r.label)}</span>
      <span class="marks">${marks(r)}</span>${switches(r)}</div>`;
  }
  function jointRows(r, depth) {
    return r.joints.map(j => `<div class="rl-row sub${r.selected_joint === j.name ? ' sel' : ''}" data-joint="${esc(j.name)}" data-link="${esc(j.link)}" data-of="${esc(r.id)}"
        title="${esc(j.name)} → ${esc(j.link)}（可把物体拖到这里，装在这个连杆上）" style="padding-left:${2 + (depth + j.depth) * 14}px"><span class="caret none"></span>
        <span class="ico">${svg(j.type === 'prismatic' ? ICON.slide : ICON.joint)}</span><span class="name">${esc(j.link)}</span></div>`);
  }
  // A folder opens with `entity|id`; a scene's single top folder starts open and closes with `entity|!id`.
  function objectRow(entity, o, depth, open = false, key = `${entity}|${o.id}`) {
    const folder = o.kind === 'folder', removed = o.excluded || o.removed;
    return `<div class="rl-row obj${o.selected ? ' sel' : ''}${o.hidden ? ' dim' : ''}${removed ? ' removed' : ''}" data-entity="${esc(entity)}" data-object="${esc(o.id)}"
        title="${esc(o.id)}" style="padding-left:${2 + depth * 14}px">
      ${folder ? `<span class="caret${open ? ' open' : ''}" data-open="${esc(key)}">${I.caret}</span>` : '<span class="caret none"></span>'}
      <span class="ico">${svg(ICON[folder ? 'folder' : o.kind === 'light' ? 'bulb' : 'box'])}</span><span class="name">${esc(o.label)}</span>
      ${folder ? `<span class="marks" style="color:var(--rl-mute);font-size:11px">${o.count}</span>` : ''}
      <button class="rl-sw eye${o.hidden ? ' set' : ''}" data-what="hidden">${o.hidden ? I.eyeOff : I.eye}</button></div>`;
  }
  function objectRows(entity, depth, words) {
    const data = objects[entity];
    if (!data) return [];
    if (words.length) return data.rows.filter(o => o.kind !== 'folder' && words.every(w => (o.label + ' ' + o.id).toLowerCase().includes(w)))
                                      .slice(0, 300).map(o => objectRow(entity, o, 1));
    const children = {};
    for (const o of data.rows) (children[o.parent] = children[o.parent] || []).push(o);
    const roots = children[null] || [], out = [];
    const walk = (list, level, removed, parent) => {
      const key = `${entity}|${parent}`, limit = listed[key] || PAGE;
      for (const o of list.slice(0, limit)) {
        const single = level === 0 && roots.length === 1, toggle = `${entity}|${single ? '!' : ''}${o.id}`;
        const open = o.kind === 'folder' && opened.has(toggle) !== single;
        out.push(objectRow(entity, removed ? {...o, removed: true} : o, depth + level, open, toggle));
        if (open) walk(children[o.id] || [], level + 1, removed || o.excluded, o.id);
      }
      if (list.length > limit) out.push(`<div class="rl-row obj more" data-more="${esc(key)}" style="padding-left:${2 + (depth + level) * 14}px">
        <span class="caret none"></span><span class="name">再显示 ${Math.min(PAGE * 5, list.length - limit)}（共 ${list.length - limit}）</span></div>`);
    };
    walk(roots, 0, false, null);
    return out;
  }
  function renderTree() {
    renderReads();
    const words = $('rl-filter').value.trim().toLowerCase().split(/\s+/).filter(Boolean), html = [];
    let folded = null;  // the depth of a folded row: rows below it are hidden
    tree.rows.forEach((r, i) => {
      if (folded !== null && r.depth > folded && !words.length) return;
      folded = null;
      const match = !words.length || words.every(w => (r.label + ' ' + r.id).toLowerCase().includes(w));
      const open = isOpen(r), kids = (tree.rows[i + 1] || {depth: -1}).depth > r.depth;
      let inner = [];
      if (r.objects && (open || (words.length && objects[r.id]))) inner = objectRows(r.id, r.depth + 1, words);
      else if (r.joints && r.joints.length && open) inner = jointRows(r, r.depth + 1);
      if (match || inner.length) html.push(entityRow(r, words, kids), ...inner);
      if (!open && kids) folded = r.depth;
    });
    $('rl-tree').innerHTML = html.join('');
    const sel = $('rl-tree').querySelector('.rl-row.sel');
    if (sel && sel.dataset.id !== renderTree.sel) sel.scrollIntoView({block: 'nearest'});
    renderTree.sel = sel && sel.dataset.id;
    if (dragged) $('rl-tree').querySelector(`.rl-row[data-id="${CSS.escape(dragged)}"]`)?.classList.add('dragging');
  }
  const revealed = {};  // entity row -> the selected object last opened down to
  async function loadTree(force) {
    const data = await api('outliner/state' + (tree.revision === null || force ? '' : `?revision=${tree.revision}`));
    if (!data.rows) return;
    tree = {revision: data.revision, rows: data.rows, lens: data.lens, role: data.role, roles: data.roles || [], train: data.train, counts: data.counts || {}};
    if (data.role_labels) ROLES = data.role_labels;
    const wanted = tree.rows.filter(r => r.objects && (opened.has(r.id) || r.selected_object || $('rl-filter').value.trim()));
    await Promise.all(wanted.filter(r => !objects[r.id] || objects[r.id].revision !== tree.revision).map(async r => {
      try { objects[r.id] = await api('outliner/objects?entity=' + encodeURIComponent(r.id.slice(7))); } catch { delete objects[r.id]; }
    }));
    // A newly selected object (a click in the view) opens the tree down to it once; the rows can be folded after.
    for (const r of tree.rows) if (r.selected_object && objects[r.id] && revealed[r.id] !== r.selected_object) {
      revealed[r.id] = r.selected_object;
      const parent = Object.fromEntries(objects[r.id].rows.map(o => [o.id, o.parent]));
      opened.add(r.id);
      for (let id = parent[r.selected_object]; id; id = parent[id]) { opened.add(`${r.id}|${id}`); opened.delete(`${r.id}|!${id}`); }
    }
    for (const r of tree.rows) if (!r.selected_object) delete revealed[r.id];
    const chosen = tree.rows.findIndex(r => r.active);
    if (chosen >= 0 && revealed[''] !== tree.rows[chosen].id) {
      revealed[''] = tree.rows[chosen].id;
      for (let i = chosen - 1, depth = tree.rows[chosen].depth; i >= 0 && depth > 0; i--) {
        const r = tree.rows[i];
        if (r.depth >= depth) continue;
        depth = r.depth;
        const key = foldKey(r);
        key.startsWith('!') ? opened.delete(key) : opened.add(key);
      }
    }
    if (!dragged) renderTree();
  }
  function toggleOpen(key) { opened.has(key) ? opened.delete(key) : opened.add(key); saveOpened(); loadTree(true); }
  // Choosing a parent by clicking: in the tree (an entity or one of a robot's links) or in the 3D view.
  function setPicking(on) { act('ui/pick_parent', {on}); }

  $('rl-tree').addEventListener('click', event => {
    const caret = event.target.closest('.caret[data-open]');
    if (caret) { toggleOpen(caret.dataset.open); return; }
    const row = event.target.closest('.rl-row'); if (!row) return;
    if (row.dataset.more) { listed[row.dataset.more] = (listed[row.dataset.more] || PAGE) + PAGE * 5; renderTree(); return; }
    if (ui && ui.picking && ((row.dataset.id || '').startsWith('entity:') || row.dataset.joint)) {
      const target = (row.dataset.of || row.dataset.id).slice(7);
      api('ui/pick_parent', {on: false}).then(() => parentDialog({target, body: row.dataset.link || null}));
      return;
    }
    if (row.dataset.joint) { act('outliner/joint', {id: row.dataset.of, joint: row.dataset.joint}); return; }
    const button = event.target.closest('.rl-sw');
    if (row.dataset.object !== undefined) {
      const body = {id: row.dataset.entity, object: row.dataset.object};
      act(button ? 'outliner/toggle_object' : 'outliner/select_object', button ? {...body, what: 'hidden'} : body); return;
    }
    if (button) { act('outliner/toggle', {id: row.dataset.id, what: button.dataset.what}); return; }
    if (event.ctrlKey || event.metaKey) { act('outliner/select', {id: row.dataset.id, mode: 'toggle'}); return; }
    if (event.shiftKey) {
      const ids = [...$('rl-tree').querySelectorAll('.rl-row[data-id]')].map(r => r.dataset.id);
      const active = (tree.rows.find(r => r.active) || {}).id, [a, b] = [ids.indexOf(active), ids.indexOf(row.dataset.id)];
      act('outliner/select', {id: row.dataset.id, mode: 'range', ids: a < 0 ? [row.dataset.id] : ids.slice(Math.min(a, b), Math.max(a, b) + 1)});
      return;
    }
    act('outliner/select', {id: row.dataset.id});
  });
  $('rl-tree').addEventListener('dblclick', event => {
    const row = event.target.closest('.rl-row'); if (!row || event.target.closest('.rl-sw, .caret')) return;
    if (row.dataset.object !== undefined) act('outliner/select_object', {id: row.dataset.entity, object: row.dataset.object, focus: true});
    else if (row.dataset.id) act('outliner/focus', {id: row.dataset.id});
  });
  $('rl-tree').addEventListener('contextmenu', async event => {
    const row = event.target.closest('.rl-row'); if (!row) return;
    event.preventDefault();
    if (row.dataset.object !== undefined) {
      await act('outliner/select_object', {id: row.dataset.entity, object: row.dataset.object});
      const o = (objects[row.dataset.entity]?.rows || []).find(x => x.id === row.dataset.object) || {};
      menu(event, objectMenu(o, row.dataset.entity, row.dataset.object));
      return;
    }
    if (row.dataset.link) {  // a robot's link: sensors go on it
      menu(event, [...sensorItems(row.dataset.of.slice(7), row.dataset.link)]);
      return;
    }
    if (!row.dataset.id) return;
    const r = tree.rows.find(x => x.id === row.dataset.id) || {};
    if (!r.selected) await act('outliner/select', {id: row.dataset.id});
    menu(event, entityMenu(r));
  });
  function objectMenu(o, entity, id) {
    return [['聚焦', () => act('outliner/select_object', {id: entity, object: id, focus: true}), false, '', 'F'],
      [o.hidden ? '显示' : '隐藏', () => act('ui/object', {what: 'hidden'}), false, '', 'H'],
      [o.excluded ? '恢复到场景' : '从场景中移除', () => act('ui/object', {what: 'exclude'}), false, o.excluded ? '' : 'danger'],
      ...(o.kind === 'object' && !o.excluded ? [['取出为独立物体', () => act('ui/object', {what: 'extract'})]] : [])];
  }
  // Drag a row onto another (its parent: kept in place) or onto one of a robot's links (mounted / fixed there);
  // hovering a robot opens its links. Dropped on empty space: no parent.
  let dragged = null, hoverOpen = null;
  const clearDrop = () => document.querySelectorAll('.rl-row.drop').forEach(e => e.classList.remove('drop'));
  $('rl-tree').addEventListener('dragstart', event => { const row = event.target.closest('.rl-row[data-id]'); if (row && row.dataset.id.startsWith('entity:')) {
    dragged = row.dataset.id; event.dataTransfer.effectAllowed = 'link'; event.dataTransfer.setData('text/plain', dragged); row.classList.add('dragging'); } });
  $('rl-tree').addEventListener('dragover', event => {
    if (!dragged) return; event.preventDefault();
    const row = event.target.closest('.rl-row[data-id], .rl-row[data-link]');
    if (row && row.classList.contains('drop')) return;
    clearDrop(); clearTimeout(hoverOpen && hoverOpen.timer); hoverOpen = null;
    if (!row || row.dataset.id === dragged) return;
    row.classList.add('drop');
    const r = row.dataset.id && tree.rows.find(x => x.id === row.dataset.id);
    if (r && r.joints && r.joints.length && !opened.has(`j:${r.id}`))
      hoverOpen = {timer: setTimeout(() => { opened.add(`j:${r.id}`); saveOpened(); renderTree(); }, 550)};
  });
  $('rl-tree').addEventListener('drop', event => {
    if (!dragged) return; event.preventDefault();
    const row = event.target.closest('.rl-row[data-id], .rl-row[data-link]'), source = dragged; dragged = null;
    clearDrop(); clearTimeout(hoverOpen && hoverOpen.timer);
    if (row && row.dataset.id === source) return renderTree();
    if (!row) { act('outliner/bind', {id: source, target: null}); return; }  // on empty space: no parent, kept in place
    if (row.dataset.id && !row.dataset.id.startsWith('entity:')) return renderTree();
    const target = (row.dataset.of || row.dataset.id).slice(7);
    api('outliner/select', {id: source}).then(async () => { await refreshInspector(); parentDialog({target, body: row.dataset.link || null}); });
  });
  $('rl-tree').addEventListener('dragend', () => { dragged = null; clearDrop(); clearTimeout(hoverOpen && hoverOpen.timer); renderTree(); });
  $('rl-filter').addEventListener('input', () => loadTree(true));

  // Right-click in the 3D view: select what is under the pointer (a click viser sees), then the same menu.
  let rightDown = null;
  const onCanvas = e => e.target && e.target.tagName === 'CANVAS' && !document.body.classList.contains('rl-classic');
  document.addEventListener('pointerdown', e => { if (e.button === 2 && onCanvas(e)) rightDown = {x: e.clientX, y: e.clientY}; }, true);
  document.addEventListener('contextmenu', e => { if (onCanvas(e)) e.preventDefault(); }, true);
  document.addEventListener('pointerup', async e => {
    if (e.button !== 2 || !rightDown) return;
    const at = rightDown; rightDown = null;
    if (Math.hypot(e.clientX - at.x, e.clientY - at.y) > 4) return;  // a right-drag pans the view
    const canvas = e.target, init = {bubbles: true, cancelable: true, clientX: at.x, clientY: at.y, button: 0, pointerId: 1, isPrimary: true, pointerType: 'mouse'};
    canvas.dispatchEvent(new PointerEvent('pointerdown', {...init, buttons: 1}));
    canvas.dispatchEvent(new PointerEvent('pointerup', {...init, buttons: 0}));
    canvas.dispatchEvent(new MouseEvent('click', init));
    await new Promise(resolve => setTimeout(resolve, 260));
    await loadTree(true);
    const r = tree.rows.find(x => x.active);
    if (!r) return;
    const box = {left: at.x, bottom: at.y};
    if (r.selected_object) {
      const o = (objects[r.id]?.rows || []).find(x => x.id === r.selected_object) || {kind: 'object'};
      menu(null, objectMenu(o, r.id, r.selected_object), false, box);
    } else {
      const state = await api('ui/state');
      menu(null, entityMenu(r, state.hit && 'entity:' + state.hit.path === r.id ? state.hit.body : ''), false, box);
    }
  }, true);

  // ------------------------------------------------------------------ add drawer (the asset shelf)
  async function openDrawer() {
    const drawer = $('rl-drawer');
    if (!drawer.querySelector('iframe')) {
      const frame = document.createElement('iframe'); frame.srcdoc = await (await fetch(API + 'ui/shelf.html')).text(); frame.title = '资产架';
      drawer.appendChild(frame);
    }
    drawer.hidden = false;
  }
  $('rl-add').onclick = openDrawer;
  $('rl-drawer-close').onclick = () => { $('rl-drawer').hidden = true; };

  // ------------------------------------------------------------------ number fields (Blender-like)
  const fmt = (v, d = 3) => (typeof v === 'number' && isFinite(v) ? String(+v.toFixed(d)) : '');
  function nf(key, value, {step = .01, digits = 3, axis = '', off = false, unit = ''} = {}) {
    return `<div class="rl-nf${off ? ' off' : ''}" data-nf="${esc(key)}" data-step="${step}" data-digits="${digits}" data-value="${value ?? ''}">
      <button class="st" data-dir="-1" tabindex="-1">${I.left}</button>${axis ? `<b class="${axis}">${axis.toUpperCase()}</b>` : ''}
      <input value="${fmt(value, digits)}" tabindex="-1">${unit ? `<i class="u">${esc(unit)}</i>` : ''}<button class="st" data-dir="1" tabindex="-1">${I.right}</button></div>`;
  }
  let interacting = false;
  // What a field changes: `final` commits (one undo step), otherwise it previews while dragging.
  function commitField(key, value, final) {
    const d = inspector, [kind, a, b] = key.split(':');
    if (kind === 'pos' || kind === 'rot') {
      const which = kind === 'pos' ? 'position' : 'rotation', vector = [...d[which]]; vector[+a] = value; d[which] = vector;
      return final ? finish('ui/pose', {[which]: vector}) : sendLive('ui/pose', {[which]: vector});
    }
    if (kind === 'scale') { const vector = [...d.scale]; vector[+a] = value; d.scale = vector;
      return final ? finish('ui/scale', {value: vector}) : sendLive('ui/scale', {value: vector}); }
    if (kind === 'size') { const size = [...d.shape.size]; size[+a] = value; d.shape.size = size;
      return final ? finish('ui/shape', {shape: d.shape.shape, size}) : sendLive('ui/shape', {shape: d.shape.shape, size}); }
    if (kind === 'joint') return final ? finish('ui/initial', {joint: a, value, commit: true}) : sendLive('ui/initial', {joint: a, value});
    if (!final) return;
    if (kind === 'period') return act('outliner/component', {role: a, field: kind, value});
    if (kind === 'param') {
      const field = (d.parameters || []).find(f => f.name === a);
      if (field && field.type === 'vector') { const v = [...field.value]; v[+b] = value; field.value = v; return act('ui/parameter', {name: a, value: v}); }
      return act('ui/parameter', {name: a, value: field && field.type === 'integer' ? Math.round(value) : value});
    }
    if (kind === 'pdict') {
      const [, name, entry, i] = key.split(':'), field = (d.parameters || []).find(f => f.name === name);
      const table = JSON.parse(JSON.stringify(field.value));
      if (Array.isArray(table[entry])) table[entry][+i] = value; else table[entry] = value;
      field.value = table;
      return act('ui/parameter', {name, value: table});
    }
    if (kind === 'yaw') { const r = d.reset, yaw = [...r.yaw]; yaw[+a] = value * Math.PI / 180;
      return act('ui/reset', {region: r.region, yaw, interval: r.interval}); }
    if (kind === 'interval') return act('ui/reset', {region: d.reset.region, yaw: d.reset.yaw, interval: value || null});
    if (kind === 'speed' || kind === 'turn') { const m = {...d.motion.draft};
      if (kind === 'turn') m.turn = value;
      else { const s = Array.isArray(m.speed) ? [...m.speed] : [m.speed ?? .1, m.speed ?? .1]; s[+a] = value; m.speed = s[0] === s[1] ? s[0] : s; }
      return act('ui/motion', {motion: m}); }
  }
  function fieldValue(el) { return parseFloat(el.dataset.value); }
  function showField(el, value) { el.dataset.value = value; el.querySelector('input').value = fmt(value, +el.dataset.digits); }
  document.addEventListener('mousedown', event => {
    const field = event.target.closest('#rl-app .rl-nf'); if (!field || field.classList.contains('edit')) return;
    event.preventDefault();
    const step = +field.dataset.step, stepper = event.target.closest('.st');
    if (stepper) { const value = (fieldValue(field) || 0) + step * +stepper.dataset.dir; showField(field, value); commitField(field.dataset.nf, value, true); return; }
    const start = event.clientX, base = fieldValue(field) || 0;
    let moved = false; interacting = true;
    const move = e => {
      if (!moved && Math.abs(e.clientX - start) < 3) return;
      moved = true; document.body.style.cursor = 'ew-resize';
      const value = base + Math.round((e.clientX - start) / 3) * step * (e.shiftKey ? .1 : 1);
      showField(field, value); commitField(field.dataset.nf, value, false);
    };
    const up = () => {
      removeEventListener('mousemove', move); removeEventListener('mouseup', up); document.body.style.cursor = '';
      if (moved) { interacting = false; commitField(field.dataset.nf, fieldValue(field), true); return; }
      field.classList.add('edit'); const input = field.querySelector('input'); input.focus(); input.select();
    };
    addEventListener('mousemove', move); addEventListener('mouseup', up);
  }, true);
  function endEdit(input, keep) {
    const field = input.closest('.rl-nf'); if (!field || !field.classList.contains('edit')) return;
    field.classList.remove('edit'); interacting = false;
    const value = parseFloat(input.value);
    if (keep && isFinite(value) && value !== fieldValue(field)) { showField(field, value); commitField(field.dataset.nf, value, true); }
    else showField(field, fieldValue(field));
  }
  document.addEventListener('keydown', e => { const input = e.target.closest?.('.rl-nf input'); if (!input) return;
    if (e.key === 'Enter') { e.preventDefault(); endEdit(input, true); input.blur(); }
    if (e.key === 'Escape') { e.preventDefault(); endEdit(input, false); input.blur(); } }, true);
  document.addEventListener('focusout', e => { if (e.target.closest?.('.rl-nf input')) endEdit(e.target, true); }, true);

  // ------------------------------------------------------------------ properties of the selection (or of the scene)
  let inspector = {revision: null};
  async function refreshInspector() {  // the selection as it is now (before a dialog reads it)
    const data = await api('ui/inspector');
    if (data.owner !== undefined) { inspector = data; renderInspector(); }
  }
  const SECTIONS = ['policy', 'robots', 'transform', 'view', 'params', 'observe', 'control', 'shape', 'reset', 'motion', 'advanced'];
  const order = JSON.parse((store && store.getItem('rl.order')) || '[]');
  const folded = new Set(JSON.parse((store && store.getItem('rl.folded')) || '["advanced"]'));
  const saveSections = () => { if (store) { store.setItem('rl.order', JSON.stringify(order)); store.setItem('rl.folded', JSON.stringify([...folded])); } };
  const vec3 = (label, kind, values, opts) => `<div class="rl-row3"><span>${label}</span>${values.map((v, i) => nf(`${kind}:${i}`, v, {...opts, axis: 'xyz'[i]})).join('')}</div>`;

  function sceneSections(s) {
    // The policy as one card: what it reads, then (an arrow) the robots it drives. With two layers the card
    // holds both in order, an arrow between them (its tooltip: what the first hands the second, from policy.py).
    const out = [], layers = s.components, two = layers.length > 1;
    const arrow = dashed => `<svg viewBox="0 0 12 30" width="12" height="30"><path d="M6 2v24" stroke="currentColor" stroke-width="2"
        stroke-linecap="round"${dashed ? ' stroke-dasharray="3 4"' : ''}/><path d="M1.5 21.5 6 27l4.5-5.5" fill="none" stroke="currentColor"
        stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
    const flow = (c, tip, on = true) => `<div class="rl-flow${on ? '' : ' off'}" style="--c:${roleVar(c.role)}" title="${esc(tip)}">${arrow(!on)}</div>`;
    const between = (c, next) => {
      if (!next.inputs.some(g => g.key === c.role && g.read))
        return flow(c, `${roleName(next.role)} 没有读 ${roleName(c.role)} 的结果：在它的“添加输入…”中加上 ${roleName(c.role)}`, false);
      const r = c.returns || {}, parts = r.parts || [];
      const tip = parts.length
        ? `${roleName(c.role)} 的 forward 返回字典：` + parts.map(p => `${p}${r.notes && r.notes[p] ? `（${r.notes[p]}）` : ''}`).join('、')
          + `；${roleName(next.role)} 读 obs.${c.role}.<名字>`
        : `${roleName(c.role)} 的 forward 返回的张量；${roleName(next.role)} 读 obs.${c.role}.output。让它返回字典（每项后面写 # 说明），这里就会显示各部分的名字和说明`;
      return flow(c, tip);
    };
    const card = c => {
      const marked = s.lens === 'observe' && s.role === c.role;
      const eye = `<button class="rl-icon sm${marked ? ' on' : ''}" data-reads="${c.role}" title="在场景中标出它读取的物体"
          style="${marked ? `background:${roleVar(c.role)}` : ''}">${I.eye}</button>`;
      const tag = `<span class="tag${c.trained ? ' hot' : ''}">${c.trained ? '🔥 训练' : '固定'}</span>`;
      const period = `<span class="rl-period" title="多久算一次（秒）"><span class="rl-lbl">每</span>${nf(`period:${c.role}`, +c.period, {step: .01, digits: 3, unit: 's'})}</span>`;
      const head = two
        ? `<div class="top"><i style="background:${roleVar(c.role)}"></i><strong>${esc(roleName(c.role))}</strong>${tag}<span class="grow"></span>${period}${eye}</div>`
        : `<div class="top">${tag}<span class="grow"></span>${period}${eye}</div>`;
      return `<div class="rl-comp">${head}${inputList(c)}</div>`;
    };
    const last = layers[layers.length - 1];
    const driven = s.robots.filter(r => r.mode !== 'none');
    const targets = s.robots.length ? flow(last, `${roleName(last.role)} 的动作经 policy.py 的 decode_action 换成每台机器人各关节的目标，`
        + '按这里选的方式（位置 / 速度 / 力矩）下发；“不控制”的机器人保持默认姿态', driven.length > 0)
      + `<div class="rl-comp rl-targets"><div class="top"><strong>控制对象</strong><span class="grow"></span></div>${s.robots.map(r =>
        `<div class="rl-robot${r.mode === 'none' ? ' none' : ''}"><span class="ico">${svg(ICON.robot)}</span>
          <span class="nm" data-path="${esc(r.path || '')}">${esc(r.key)}</span><em>${r.joints ? `${r.joints} 个关节` : ''}</em>
          ${dd(MODES, r.mode, mode => act('outliner/mode', {id: 'entity:' + r.path, mode}), {disabled: !s.automatic || !r.path})}</div>`).join('')}</div>` : '';
    const train = two ? `<div class="rl-field"><span class="rl-lbl">训练</span><div class="rl-seg">${layers.map(c =>
      `<button data-train="${c.role}" class="${c.trained ? 'on' : ''}">${c.trained ? '🔥 ' : ''}${esc(roleName(c.role))}</button>`).join('')}</div></div>` : '';
    const shape = !s.layered ? '' : two
      ? `<button class="rl-btn wide subtle" data-act="merge_policy" title="去掉 ${esc(layers[0].role)}，只留 ${esc(last.role)}">${I.x}<span>去掉 ${esc(layers[0].role)}</span></button>`
      : `<button class="rl-btn wide subtle" data-act="split_policy" title="加一层上层规划：不直接控制机器人，把目标、路径或指令交给下一层">${I.plus}<span>增加一层…</span></button>`;
    out.push(['policy', '策略', train + layers.map((c, i) => card(c) + (layers[i + 1] ? between(c, layers[i + 1]) : '')).join('') + targets + shape]);
    return out;
  }
  // A component's inputs as one list: the entities it reads, then what it reads from outside the scene (the
  // simulation, earlier components). Each group unfolds in place to tick its quantities; the arrow opens the
  // entity's own properties. Entities it does not read yet come from "添加".
  const inputsOpen = new Set(JSON.parse((store && store.getItem('rl.inputs')) || '[]')), inputsAdded = new Set();
  function inputList(c) {
    // What it reads, and what was just added; the rest (unread entities, the simulation's gravity, other
    // components' outputs) waits in 添加输入.
    const shown = c.inputs.filter(g => g.read || inputsAdded.has(`${c.role}|${g.key}`));
    const more = c.inputs.filter(g => !shown.includes(g)).map(g => [g.key, g.path ? g.label : roleName(g.key)]);
    const add = more.length ? dd(more, null, key => { inputsAdded.add(`${c.role}|${key}`); inputsOpen.add(`${c.role}|${key}`);
      saveInputs(); renderInspector(); }, {placeholder: '添加输入…'}) : '';
    return `<div class="rl-sub">输入</div><div class="rl-inputs">${shown.map(g => {
      const id = `${c.role}|${g.key}`, open = inputsOpen.has(id);
      const head = `<div class="rl-ig${g.read ? '' : ' none'}" data-group="${esc(id)}" title="${esc(g.path || g.key)}">
        <span class="caret${open ? ' open' : ''}">${I.caret}</span><span class="ico">${g.path ? svg(ICON[g.icon] || ICON.box) : g.key === 'simulation' ? svg(ICON.sim)
          : `<i style="background:${roleVar(g.key)}"></i>`}</span>
        <span class="nm">${esc(g.path ? g.label : roleName(g.key))}</span><b>${g.read}/${g.rows.length}</b>
        ${g.path ? `<button class="rl-icon sm" data-path="${esc(g.path)}" title="打开它的属性">${I.open}</button>` : '<span class="rl-icon sm"></span>'}
        <button class="rl-icon sm" data-drop="${esc(id)}" title="不再读取它（取消它的全部量）">${I.x}</button></div>`;
      const rows = open ? g.rows.map(q => `<label class="rl-iq" data-extra="${esc(q.key)}" data-role="${c.role}" data-on="${q.enabled ? 1 : 0}"
          title="obs.${esc(q.key)} · ${esc([q.label, q.shape, q.unit, q.description].filter(Boolean).join(' · '))}"><span class="nm">${esc(g.path ? short(q.label || q.name).replace(/（.*）/, '') : extraLabel({...q, group: g.key}, c.role))}</span>
          <em>${esc([q.shape, q.unit].filter(Boolean).join(' '))}</em><span class="rl-tick${q.enabled ? ' on' : ''}" style="${q.enabled ? `background:${roleVar(c.role)}` : ''}"></span></label>`).join('') : '';
      return head + rows;
    }).join('') || `<span class="rl-none">${add ? '还没有输入' : '场景里还没有可读的东西：先添加机器人或物体'}</span>`}${add ? `<div class="rl-iadd">${add}</div>` : ''}</div>`;
  }
  const saveInputs = () => store && store.setItem('rl.inputs', JSON.stringify([...inputsOpen]));
  // A quantity outside the scene's entities, named for what it is to this component.
  function extraLabel(q, role) {
    const label = String(q.label || q.name).replace(/（.*）/, '');
    if (q.group === 'simulation') return label;
    if (q.group === role) return q.name === 'output' ? roleName(role) : label;
    return `${roleName(q.group)} ${label}`;
  }
  function entitySections(d) {
    const out = [];
    let transform = vec3('位置', 'pos', d.position, {step: .01, digits: 4, off: d.locked}) + vec3('旋转', 'rot', d.rotation, {step: 1, digits: 2, off: d.locked});
    if (d.scale) transform += vec3('缩放', 'scale', d.scale, {step: .01, digits: 3});
    if (d.kind === 'entity') transform += parentLines(d);
    out.push(['transform', '变换', transform]);
    if (d.camera) out.push(['view', '画面', `<div class="rl-cam" style="aspect-ratio:${d.camera.width} / ${d.camera.height}" title="编辑器示意渲染：位姿与内参同 scene.yaml，不是仿真传感器图像">
      <img id="rl-cam" alt="">${d.camera.error ? `<span class="err">${esc(d.camera.error)}</span>` : ''}<button class="rl-icon" data-act="camera" title="重新渲染">${I.rotate}</button></div>
      <button class="rl-btn wide" data-act="camera_view" title="相机移到当前 3D 视图的位置并朝同一方向看；装在连杆上的相机保持安装，记录相对连杆的位姿">${I.focus}<span>对准当前视角</span></button>`]);
    const params = (d.parameters || []).map(paramField).join('') + (d.parameters_json !== undefined ? `<textarea class="rl-text" data-json rows="5">${esc(d.parameters_json)}</textarea>` : '');
    // A sensor's or a light's parameters are what it is; a robot's or an object's are tuning, folded away.
    if (params) out.push(['sensor', 'light'].includes(d.category) ? ['params', '参数', params] : ['advanced', '高级', params]);
    const roles = (d.roles || []).filter(role => (d.signals || []).some(q => role in q.roles));
    if (roles.length) out.push(['observe', '观察', `<div class="rl-obs" style="grid-template-columns:1fr ${roles.map(() => '24px').join(' ')}"><span></span>
      ${roles.map(r => `<span class="h" style="color:${roleVar(r)}" title="${esc(roleName(r))}${d.train === r ? '（训练中）' : ''}">${esc(roleMark(r))}</span>`).join('')}
      ${d.signals.map(q => `<span class="q" title="obs.${esc(q.key)} · ${esc([q.shape, q.unit].filter(Boolean).join(' '))}">${esc(short(q.label || q.name))}</span>` + roles.map(r => {
        const on = q.roles[r]; return `<span class="rl-tick${on ? ' on' : ''}${on === null || on === undefined ? ' na' : ''}" style="${on ? `background:${roleVar(r)}` : ''}" data-role="${r}" data-key="${esc(q.key)}" data-on="${on ? 1 : 0}"></span>`;
      }).join('')).join('')}</div>`]);
    if (d.mode || (d.joints && d.joints.length)) {
      // An imported asset with joints: a robot the policy drives, or a passive object (a drawer, a lid).
      let body = d.robot !== undefined ? `<div class="rl-line" title="打开后它的关节由策略控制（像机械臂、机器狗）；关闭则是被动的物体（像抽屉、门）">
          <span class="grow">作为机器人</span><span class="rl-toggle${d.robot ? ' on' : ''}" data-robot></span></div>` : '';
      body += d.gravity !== undefined ? `<div class="rl-line" title="打开后它的连杆不受重力（gravity_compensation），关节只管跟踪目标，像真机械臂控制器自带的重力补偿；动作是“当前位置 + 增量”时不会慢慢下垂">
          <span class="grow">重力补偿</span><span class="rl-toggle${d.gravity ? ' on' : ''}" data-gravity></span></div>` : '';
      body += d.mode ? `<div class="rl-seg">${MODES.filter(([m]) => d.automatic || m === d.mode).map(([m, label]) =>
          `<button data-mode="${m}" class="${d.mode === m ? 'on' : ''}">${label}</button>`).join('')}</div>` : '';
      body += (d.joints || []).map(j => {
        const lim = j.limits && j.limits.length === 2 && isFinite(j.limits[0]) && isFinite(j.limits[1]) && j.limits[0] < j.limits[1] ? j.limits : null;
        const t = lim && typeof j.value === 'number' ? Math.max(0, Math.min(1, (j.value - lim[0]) / (lim[1] - lim[0]))) : null;
        return `<div class="rl-joint${d.selected_joint === j.name ? ' on' : ''}" data-joint="${esc(j.name)}" title="${esc(j.name)} → ${esc(j.link)}${j.index === null ? '' : ` · 动作目标第 ${j.index + 1} 列`}">
          <span class="nm"><span>${esc(j.name)}</span>${d.couplable ? `<button class="rl-couple${j.follows ? ' on' : ''}" data-couple="${esc(j.name)}"
            title="${j.follows ? `跟随 ${esc(j.follows.leader)}：本关节 = ${j.follows.ratio} × 主动关节${j.follows.offset ? ` + ${j.follows.offset}` : ''}（点击修改）` : '联动：让它按比例跟随另一个关节（如踏板带动盖子）'}">${I.link}${j.follows ? `<em>×${+(+j.follows.ratio).toFixed(3)}</em>` : ''}</button>` : ''}</span>
          <div class="rl-track${lim ? '' : ' none'}" data-track="${esc(j.name)}" data-lo="${lim ? lim[0] : ''}" data-hi="${lim ? lim[1] : ''}">
            ${t === null ? '' : `<div class="fill" style="left:0;width:${t * 100}%"></div><div class="knob" style="left:${t * 100}%"></div>`}</div>
          ${nf(`joint:${j.name}`, j.value, {step: j.type === 'prismatic' ? .001 : .01, digits: 3})}</div>`;
      }).join('');
      out.push(['control', d.mode ? '控制' : '关节', body]);
    }
    if (d.shape) out.push(['shape', '形状', `<div class="rl-seg">${[['point', '点'], ['direction', '方向'], ['rectangle', '矩形'], ['box', '长方体']].map(([k, label]) =>
        `<button data-shape="${k}" class="${d.shape.shape === k ? 'on' : ''}">${label}</button>`).join('')}</div>` +
      (d.shape.size.length ? `<div class="rl-row3"><span>尺寸</span>${d.shape.size.map((v, i) => nf(`size:${i}`, v, {step: .01, digits: 3, axis: 'xyz'[i]})).join('')}</div>` : '')]);
    if (d.reset) out.push(['reset', '复位', resetBody(d.reset)]);
    if (d.motion && !d.motion.blocked) out.push(['motion', '运动', motionBody(d.motion)]);
    return out;
  }
  // The parent: which entity, how (mounted / followed / fixed, or only laid out with it) and at which link.
  function parentLines(d) {
    const pick = `<button class="rl-icon sm${ui && ui.picking ? ' on' : ''}" data-act="pick" title="在视图或场景树中点选父物体（点到机器人的哪个连杆就装在哪个连杆上）P">${I.target}</button>`;
    if (!d.parent) {
      const targets = d.targets.map(t => [t.path, t.label, t.path]);
      return `<div class="rl-field"><span class="rl-lbl">父物体</span>${dd(targets, null, target => parentDialog({target}), {placeholder: '无', search: true})}${pick}</div>`;
    }
    const p = d.parent, relation = (p.modes || []).find(([m]) => m === p.mode);
    const link = p.body && p.body.startsWith('.*') ? `所有 *${p.body.slice(2).replace(/\\/g, '')}` : p.body;
    const where = [relation ? relation[1] : '', link ? `在 ${link}` : ''].filter(Boolean).join(' · ');
    return `<div class="rl-field"><span class="rl-lbl">父物体</span><span class="rl-chip grow" data-act="reparent" title="改关系、连杆或位置">${I.link}<span>${esc(baseName(p.entity))}</span>
      <em>${esc(where)}</em><span class="x" data-act="unparent" title="解除（Alt+P）">${I.x}</span></span>${pick}</div>`;
  }
  // The parent dialog: how the selection hangs on ``target`` (relation, link) and where it goes (stays, or
  // moves onto the parent). Opened by a drop in the tree, a pick in the view or tree, and the panel.
  // A joint of a jointed object follows another: follower = ratio x leader + offset, held by the physics engine.
  // 增加一层 / 去掉一层: when a second layer helps, and what the editor writes (scene.yaml, a start in policy.py).
  function splitDialog() {
    const layer = ((inspector.scene || {}).components || [])[0];
    if (!layer) return;
    // What the new layer reads: one of the policy's scene inputs, a target (an abstract entity) first.
    const groups = layer.inputs.filter(g => g.path && g.read).sort((a, b) => b.abstract - a.abstract);
    const sources = groups.flatMap(g => g.rows.filter(q => q.enabled).map(q => [q.key, `${g.label} · ${short(q.label || q.name).replace(/（.*）/, '')}`]));
    if (!sources.length) return toast(`${layer.role} 还没有读场景里的东西；先给它选一项输入`, true);
    let source = sources[0][0];
    dialog('增加一层', `
      <p>在 ${esc(layer.role)} 前面加一层<b>上层规划</b>：它不直接控制机器人，只把结果（目标点、路径、指令）交给 ${esc(layer.role)}；
        可以跑得更慢，也可以是现成的模型或手写规则。</p>
      <div class="rl-field rl-newlayer"><span class="rl-lbl">名称</span><input id="rl-split-name" value="plan" spellcheck="false">
        <span class="rl-lbl">每</span><input id="rl-split-period" value="${(+layer.period * 10).toFixed(3).replace(/\.?0+$/, '')}"><em class="rl-unit">s</em></div>
      <div class="rl-field"><span class="rl-lbl">读取</span><span id="rl-split-source"></span></div>
      <p class="rl-hint">policy.py 末尾会加上它的起始代码（先原样转交所选的量），改成你的计算。</p>`, [
      ['取消', '', closeDialog],
      ['增加', 'primary', box => { const name = box.querySelector('#rl-split-name').value.trim(), period = +box.querySelector('#rl-split-period').value;
        closeDialog(); act('ui/split_policy', {name, period, source}); }]], () => {
        const slot = () => { $('rl-split-source').innerHTML = dd(sources, source, v => { source = v; slot(); }, {scope: 'g'}); };
        slot(); });
  }
  function mergeDialog() {
    const layers = ((inspector.scene || {}).components || []);
    if (layers.length !== 2) return;
    const [first, second] = layers;
    dialog(`去掉 ${first.role}`, `<p>去掉 <b>${esc(first.role)}</b>，只留 <b>${esc(second.role)}</b>（它读的其他输入不变）。</p>
      <p class="rl-hint">policy.py 里 ${esc(first.role)} 的代码保留；${esc(second.role)} 如果用到 obs.${esc(first.role)}，需要改掉。可用 Ctrl-Z 撤回。</p>`, [
      ['取消', '', closeDialog], ['去掉', 'danger', () => { closeDialog(); act('ui/merge_policy', {}); }]]);
  }
  function couplingDialog(name) {
    const d = inspector, joint = d.joints.find(j => j.name === name), others = d.joints.filter(j => j.name !== name);
    const choice = {leader: joint.follows ? joint.follows.leader : (others[0] || {}).name, ratio: joint.follows ? joint.follows.ratio : 1,
                    offset: joint.follows ? joint.follows.offset || 0 : 0};
    const unit = j => j && j.type === 'prismatic' ? 'm' : 'rad';
    const draw = box => {
      dropScope('g');
      const leader = d.joints.find(j => j.name === choice.leader);
      box.querySelector('.content').innerHTML = `
        <div class="rl-field"><span class="rl-lbl">跟随</span>${dd(others.map(j => [j.name, j.name]), choice.leader, value => { choice.leader = value; draw(box); }, {scope: 'g'})}</div>
        <div class="rl-field"><span class="rl-lbl">比例</span><input id="rl-cp-ratio" value="${choice.ratio}" spellcheck="false"></div>
        <div class="rl-field"><span class="rl-lbl">偏移</span><input id="rl-cp-offset" value="${choice.offset}" spellcheck="false"><em class="rl-unit">${unit(joint)}</em></div>
        <p class="rl-hint">${esc(name)} = 比例 × ${esc(choice.leader || '')} + 偏移（${unit(joint)} / ${unit(leader)}）。仿真里由物理引擎保持，力双向传递：
          推动任一个，另一个跟着动。${leader && leader.limits && joint.limits ? `例：${esc(choice.leader)} 从 ${leader.limits[0].toFixed(2)} 到 ${leader.limits[1].toFixed(2)}，
          ${esc(name)} 从 ${joint.limits[0].toFixed(2)} 到 ${joint.limits[1].toFixed(2)}，比例可取 ${((joint.limits[1] - joint.limits[0]) / (leader.limits[1] - leader.limits[0]) || 1).toFixed(3)} 或其相反数。` : ''}</p>`;
    };
    const read = box => ({ratio: +box.querySelector('#rl-cp-ratio').value, offset: +box.querySelector('#rl-cp-offset').value});
    dialog(`联动 ${name}`, '', [
      ...(joint.follows ? [['不再联动', 'danger', () => { closeDialog(); act('ui/coupling', {joint: name, leader: null}); }]] : []),
      ['取消', '', closeDialog],
      ['确定', 'primary', box => { closeDialog(); act('ui/coupling', {joint: name, leader: choice.leader, ...read(box)}); }]], draw);
  }
  async function parentDialog({target, body = null, mode = null}) {
    const d = inspector;
    if (!d.owner || d.kind !== 'entity') return;
    let options;
    try { options = await api('ui/attach?target=' + encodeURIComponent(target)); } catch (error) { toast(error.message, true); return; }
    const current = d.parent && d.parent.entity === target ? d.parent : null;
    let choice = {target, mode: mode || (current ? current.mode : options.mode), place: 'keep'};
    choice.body = body ?? (current ? current.body || '' : options.body);
    const PLACES = [['keep', '保持原位'], ['origin', '移到父物体原点'], ['align', '对齐父物体（位置与朝向）']];
    const draw = box => {
      dropScope('g');
      const links = [...(choice.mode === 'layout' ? [['', '整体（根部）']] : []), ...options.groups.map(([pattern, label]) => [pattern, label]),
                     ...options.bodies.map(b => [b, b])];
      if (choice.mode !== 'layout' && !links.some(([v]) => v === choice.body)) choice.body = options.body;
      box.querySelector('.content').innerHTML = `
        <div class="rl-field"><span class="rl-lbl">父物体</span>${dd(d.targets.map(t => [t.path, t.label, t.path]), choice.target, async value => {
          closeDialog(); parentDialog({target: value}); }, {scope: 'g', search: true})}</div>
        ${options.modes.length > 1 ? `<div class="rl-field"><span class="rl-lbl">关系</span><div class="rl-seg" data-group="mode">${options.modes.map(([m, l]) =>
          `<button data-v="${m}" class="${choice.mode === m ? 'on' : ''}">${l}</button>`).join('')}</div></div>` : ''}
        ${options.linkable && links.length > 1 ? `<div class="rl-field"><span class="rl-lbl">连杆</span>${dd(links, choice.body, value => { choice.body = value; draw(box); }, {scope: 'g', search: true})}</div>` : ''}
        <div class="rl-field col"><span class="rl-lbl">位置</span><div class="rl-seg" data-group="place">${PLACES.map(([m, l]) =>
          `<button data-v="${m}" class="${choice.place === m ? 'on' : ''}">${l}</button>`).join('')}</div></div>
        <p class="rl-hint">${choice.mode === 'layout' ? '只一起摆放：移动父物体时它跟着走，仿真里没有连接。' :
          d.category === 'sensor' ? '传感器装在连杆上，随连杆运动。' : d.category === 'abstract' ? '读数跟随父物体根部，不是物理约束。' :
          '两个物体在仿真里固定连接；它自己的随机复位会去掉。'}</p>`;
    };
    dialog(`设置 ${d.label} 的父物体`, '', [['取消', '', () => { closeDialog(); act('ui/parent_choice', {}); }],
      ['确定', 'primary', async () => {
        const body = choice.mode === 'layout' || options.linkable ? choice.body : '';
        if (await act('ui/attach', {target: choice.target, body: body === undefined ? null : body, mode: choice.mode, place: choice.place})) closeDialog();
      }]], box => {
        draw(box);
        box.querySelector('.content').addEventListener('click', e => {
          const b = e.target.closest('.rl-seg[data-group] button'); if (!b) return;
          choice[b.parentElement.dataset.group] = b.dataset.v; draw(box);
        });
      });
  }
  const deg = r => Math.round(r * 1800 / Math.PI) / 10;
  function resetBody(reset) {
    const options = [['', '固定位置'], ...reset.areas.map(a => [a, '随机放进 ' + baseName(a)])];
    return `<div class="rl-field"><span class="rl-lbl">每回合</span>${dd(options, reset.region || '', region =>
        act('ui/reset', {region: region || null, yaw: inspector.reset.yaw, interval: inspector.reset.interval}))}</div>` +
      (reset.region ? `<div class="rl-field"><span class="rl-lbl">朝向</span>${nf('yaw:0', deg(reset.yaw[0]), {step: 5, digits: 1, unit: '°'})}${nf('yaw:1', deg(reset.yaw[1]), {step: 5, digits: 1, unit: '°'})}</div>` +
        (reset.abstract ? `<div class="rl-field"><span class="rl-lbl">重采样</span>${nf('interval', reset.interval || 0, {step: .5, digits: 2, unit: 's'})}</div>` : '') : '');
  }
  function motionBody(motion) {
    const m = motion.draft || {}, kind = m.type || '';
    let body = `<div class="rl-seg">${[['', '无'], ['wander', '游走'], ['path', '路径'], ['custom', '自定义']].map(([k, label]) =>
      `<button data-motion="${k}" class="${kind === k ? 'on' : ''}">${label}</button>`).join('')}</div>`;
    const speed = Array.isArray(m.speed) ? m.speed : [m.speed ?? .1, m.speed ?? .1];
    if (kind === 'wander' || kind === 'path') body += `<div class="rl-field"><span class="rl-lbl">速度</span>${nf('speed:0', speed[0], {step: .01, digits: 3, unit: 'm/s'})}${nf('speed:1', speed[1], {step: .01, digits: 3, unit: 'm/s'})}</div>
      <div class="rl-line"><span class="grow">朝向运动方向</span><span class="rl-toggle${m.face ? ' on' : ''}" data-mtoggle="face"></span></div>`;
    if (kind === 'wander') {
      const options = [...(motion.own_region ? [['', '复位区域']] : []), ...motion.areas.map(a => [a, baseName(a)])];
      body += `<div class="rl-field"><span class="rl-lbl">区域</span>${dd(options, m.region || '', region => {
        const next = {...inspector.motion.draft}; if (region) next.region = region; else delete next.region; act('ui/motion', {motion: next}); }, {placeholder: '选择区域'})}</div>`;
    }
    if (kind === 'path') body += `<div class="rl-line" style="gap:6px"><span class="grow">${(m.points || []).length} 个点</span>
        <button class="rl-btn" data-point="add">在视图中点选</button><button class="rl-btn" data-point="pop">撤一个</button></div>
      <div class="rl-line"><span class="grow">循环</span><span class="rl-toggle${m.loop !== false ? ' on' : ''}" data-mtoggle="loop"></span></div>
      <div class="rl-line"><span class="grow">复位到路径上随机位置</span><span class="rl-toggle${m.start === 'random' ? ' on' : ''}" data-mtoggle="start"></span></div>`;
    if (kind === 'custom') body += `<div class="rl-field"><span class="rl-lbl">函数</span><input data-function value="${esc(m.function || '.behaviors:step')}"></div>`;
    if (kind === 'wander' || kind === 'path') body += `<div class="rl-line"><button class="rl-btn" data-act="preview">▶ 预览</button></div>`;
    return body;
  }
  function paramField(f) {
    const v = f.value, [name, unit, whole] = splitLabel(f.label);
    const label = `<span class="rl-lbl" title="${esc(whole)} · ${esc(f.name)}">${esc(name)}</span>`;
    if (f.type === 'boolean' && typeof v === 'boolean')
      return `<div class="rl-line"><span class="grow" title="${esc(whole)}">${esc(name)}</span><span class="rl-toggle${v ? ' on' : ''}" data-param-toggle="${esc(f.name)}"></span></div>`;
    if (f.type === 'choice' && (f.options || []).includes(v))
      return `<div class="rl-field">${label}${dd(f.options.map(o => [o, f.option_labels[o] || o]), v, value => act('ui/parameter', {name: f.name, value}))}</div>`;
    if (f.type === 'choices' && Array.isArray(v))
      return `<div class="rl-field col">${label}<div class="rl-options">${(f.options || []).map(o => `<button class="${v.includes(o) ? 'on' : ''}" data-choice="${esc(f.name)}" data-opt="${esc(o)}"
        title="${esc(o)}"><span class="rl-check">${I.check}</span>${esc(f.option_labels[o] || o)}</button>`).join('')}</div></div>`;
    if (f.bodies && f.bodies.length)
      return `<div class="rl-field">${label}${dd(f.bodies.map(b => [b, b]), v, value => act('ui/parameter', {name: f.name, value}), {placeholder: '选择连杆'})}</div>`;
    if ((f.type === 'integer' || f.type === 'number') && typeof v === 'number')
      return `<div class="rl-field">${label}${nf(`param:${f.name}`, v, {step: f.type === 'integer' ? 1 : .01, digits: f.type === 'integer' ? 0 : 3, unit})}</div>`;
    if (f.type === 'vector' && Array.isArray(v))
      return `<div class="rl-field">${label}${v.map((x, i) => nf(`param:${f.name}:${i}`, x, {step: .01, digits: 3, unit})).join('')}</div>`;
    if (f.type === 'text' && typeof v === 'string')
      return `<div class="rl-field">${label}<input data-param="${esc(f.name)}" value="${esc(v)}"></div>`;
    // A table of numbers per name (gains per joint): one row each, numbers editable like the others.
    const numeric = x => typeof x === 'number' || (Array.isArray(x) && x.length && x.every(n => typeof n === 'number'));
    if (v && typeof v === 'object' && !Array.isArray(v) && Object.keys(v).length && Object.values(v).every(numeric))
      return `<div class="rl-sub" title="${esc(f.name)}">${esc(name)}</div>` + Object.entries(v).map(([k, x]) =>
        `<div class="rl-field"><span class="rl-lbl" title="${esc(k)}">${esc(KEY_LABELS[k] || k)}</span>${[].concat(x).map((n, i) =>
          nf(`pdict:${f.name}:${k}:${i}`, n, {step: Math.abs(n) >= 10 ? 1 : .05, digits: 3, unit})).join('')}</div>`).join('');
    if (v === null || v === undefined)
      return `<div class="rl-field">${label}<input data-param-json="${esc(f.name)}" placeholder="资产默认"></div>`;
    return `<div class="rl-field col">${label}<textarea class="rl-text" data-param-json="${esc(f.name)}" rows="2">${esc(JSON.stringify(v))}</textarea></div>`;
  }
  function renderInspector() {
    const box = $('rl-right'), d = inspector;
    if (interacting || !$('rl-pop').hidden || (box.contains(document.activeElement) && document.activeElement.matches('input, textarea'))) {
      renderInspector.pending = true;  // drawn once the pointer, the list or the typing is done
      return;
    }
    renderInspector.pending = false;
    dropScope('i');
    let head, list;
    if (!d.owner) {
      if (!d.scene) { box.innerHTML = ''; return; }
      head = `<div id="rl-head"><span class="ico">${I.scene}</span><span class="title">场景</span></div>`;
      list = sceneSections(d.scene);
    } else {
      head = `<div id="rl-head"><span class="ico">${svg(ICON[d.icon] || ICON.box)}</span><input id="rl-name" value="${esc(d.label)}"${d.kind === 'entity' ? '' : ' disabled'}>
        <button class="rl-icon" data-act="focus" title="聚焦 F">${I.focus}</button>
        ${d.kind === 'entity' ? `<button class="rl-icon" data-act="delete" title="删除">${I.trash}</button>` : ''}</div>`;
      list = entitySections(d);
    }
    for (const [key] of list) if (!order.includes(key)) {  // a section seen for the first time: at its usual place
      const later = order.findIndex(k => SECTIONS.indexOf(k) > SECTIONS.indexOf(key));
      order.splice(later < 0 ? order.length : later, 0, key);
    }
    list.sort((a, b) => order.indexOf(a[0]) - order.indexOf(b[0]));
    const scroll = box.scrollTop;
    box.innerHTML = head + list.map(([key, title, body]) => `<section class="rl-sec${folded.has(key) ? ' folded' : ''}" data-sec="${key}">
      <h4 draggable="true">${title}<span class="chev">${I.chev}</span></h4><div class="body">${body}</div></section>`).join('');
    box.scrollTop = scroll;
    shownCamera = null; showCamera();
  }
  const right = $('rl-right');
  right.addEventListener('click', event => {
    const t = event.target, d = inspector;
    const head = t.closest('.rl-sec > h4');
    if (head) { const key = head.parentElement.dataset.sec; folded.has(key) ? folded.delete(key) : folded.add(key); saveSections(); renderInspector(); return; }
    const a = t.closest('[data-act]')?.dataset.act;
    if (a === 'focus') return act('ui/focus', {});
    if (a === 'delete') return confirmDelete();
    if (a === 'unparent') return act('ui/parent', {target: null});
    if (a === 'reparent') return parentDialog({target: d.parent.entity, body: d.parent.body || '', mode: d.parent.mode});
    if (a === 'preview') return act('ui/motion_preview', {});
    if (a === 'pick') return setPicking(!(ui && ui.picking));
    if (a === 'camera') return act('ui/camera', {});
    if (a === 'camera_view') return act('ui/camera_view', {});
    if (a === 'split_policy') return splitDialog();
    if (a === 'merge_policy') return mergeDialog();
    const couple = t.closest('[data-couple]'); if (couple) return couplingDialog(couple.dataset.couple);
    const path = t.closest('[data-path]'); if (path && path.dataset.path) return act('outliner/select', {id: 'entity:' + path.dataset.path});
    const drop = t.closest('[data-drop]'); if (drop) {
      const [role, key] = drop.dataset.drop.split('|');
      const g = ((d.scene || {}).components || []).find(c => c.role === role)?.inputs.find(x => x.key === key);
      inputsAdded.delete(drop.dataset.drop); inputsOpen.delete(drop.dataset.drop); saveInputs();
      const keys = (g ? g.rows : []).filter(q => q.enabled).map(q => q.key);
      return keys.length ? act('ui/unread', {role, keys}) : renderInspector();
    }
    const group = t.closest('[data-group]'); if (group) {
      const id = group.dataset.group; inputsOpen.has(id) ? inputsOpen.delete(id) : inputsOpen.add(id); saveInputs(); return renderInspector();
    }
    const train = t.closest('[data-train]'); if (train && !train.classList.contains('on')) return act('outliner/train', {role: train.dataset.train});
    const reads = t.closest('[data-reads]'); if (reads) return act('outliner/lens', reading() === reads.dataset.reads ? {lens: 'scene'} : {lens: 'observe', role: reads.dataset.reads});
    const extra = t.closest('[data-extra]'); if (extra) return act('outliner/observe', {role: extra.dataset.role, key: extra.dataset.extra, on: extra.dataset.on !== '1'});
    const mode = t.closest('[data-mode]'); if (mode) return act('outliner/mode', {id: d.owner, mode: mode.dataset.mode});
    const tick = t.closest('.rl-tick[data-key]'); if (tick && !tick.classList.contains('na'))
      return act('outliner/observe', {role: tick.dataset.role, key: tick.dataset.key, on: tick.dataset.on !== '1'});
    const shape = t.closest('[data-shape]'); if (shape) return act('ui/shape', {shape: shape.dataset.shape});
    const choice = t.closest('[data-choice]'); if (choice) {
      const field = d.parameters.find(f => f.name === choice.dataset.choice), o = choice.dataset.opt;
      const value = field.value.includes(o) ? field.value.filter(x => x !== o) : [...field.value, o];
      if (!value.length) return toast('至少保留一项', true);
      return act('ui/parameter', {name: field.name, value});
    }
    const motion = t.closest('[data-motion]'); if (motion) {
      const kind = motion.dataset.motion;
      return act('ui/motion', {motion: !kind ? null : kind === 'custom' ? {type: 'custom', function: '.behaviors:step'} : {type: kind, speed: .1}}); }
    const mt = t.closest('[data-mtoggle]'); if (mt) { const m = {...d.motion.draft}, k = mt.dataset.mtoggle;
      if (k === 'face') { if (m.face) delete m.face; else m.face = true; }
      if (k === 'loop') { if (m.loop === false) delete m.loop; else m.loop = false; }
      if (k === 'start') { if (m.start === 'random') delete m.start; else m.start = 'random'; }
      return act('ui/motion', {motion: m}); }
    const point = t.closest('[data-point]'); if (point) return act('ui/motion_point', {what: point.dataset.point});
    const toggle = t.closest('[data-param-toggle]'); if (toggle) return act('ui/parameter', {name: toggle.dataset.paramToggle, value: !toggle.classList.contains('on')});
    const robot = t.closest('[data-robot]'); if (robot) return act('ui/robot', {on: !robot.classList.contains('on')});
    const gravity = t.closest('[data-gravity]'); if (gravity) return act('ui/gravity', {on: !gravity.classList.contains('on')});
    const joint = t.closest('.rl-joint'); if (joint && !t.closest('.rl-nf, .rl-track')) return act('outliner/joint', {id: d.owner, joint: joint.dataset.joint});
  });
  right.addEventListener('change', event => {
    const t = event.target, d = inspector;
    if (t.id === 'rl-name') return act('ui/label', {value: t.value});
    if (t.dataset.function !== undefined) return act('ui/motion', {motion: {...d.motion.draft, function: t.value.trim()}});
    if (t.dataset.param) return act('ui/parameter', {name: t.dataset.param, value: t.value});
    if (t.dataset.paramJson && !t.value.trim()) return;
    if (t.dataset.paramJson) { try { return act('ui/parameter', {name: t.dataset.paramJson, value: JSON.parse(t.value)}); }
      catch (error) { return toast(`${t.dataset.paramJson}：${error.message}`, true); } }
    if (t.dataset.json !== undefined) return act('ui/parameters_json', {text: t.value});
  });
  right.addEventListener('keydown', event => { if (event.key === 'Enter' && event.target.matches('input:not(.rl-nf input)')) event.target.blur(); });
  // A joint's track: press and drag; the pose follows while dragging, the release is one undo step.
  right.addEventListener('pointerdown', event => {
    const track = event.target.closest('.rl-track[data-lo]'); if (!track || track.dataset.lo === '') return;
    event.preventDefault(); track.setPointerCapture(event.pointerId); interacting = true;
    const lo = +track.dataset.lo, hi = +track.dataset.hi, row = track.closest('.rl-joint'), field = row.querySelector('.rl-nf');
    const at = e => { const box = track.getBoundingClientRect(); const t = Math.max(0, Math.min(1, (e.clientX - box.left) / box.width));
      track.querySelector('.knob').style.left = track.querySelector('.fill').style.width = `${t * 100}%`;
      const value = lo + t * (hi - lo); showField(field, value); return value; };
    commitField(`joint:${row.dataset.joint}`, at(event), false);
    const move = e => commitField(`joint:${row.dataset.joint}`, at(e), false);
    const up = e => { track.removeEventListener('pointermove', move); track.removeEventListener('pointerup', up);
      interacting = false; commitField(`joint:${row.dataset.joint}`, at(e), true); };
    track.addEventListener('pointermove', move); track.addEventListener('pointerup', up);
  });
  let dragSec = null;
  right.addEventListener('dragstart', event => { const h = event.target.closest('.rl-sec > h4'); if (h) dragSec = h.parentElement.dataset.sec; });
  right.addEventListener('dragover', event => { const s = event.target.closest('.rl-sec'); if (!dragSec || !s) return; event.preventDefault();
    right.querySelectorAll('.drag-over').forEach(e => e.classList.remove('drag-over')); s.classList.add('drag-over'); });
  right.addEventListener('drop', event => { const s = event.target.closest('.rl-sec'); if (!dragSec || !s) return; event.preventDefault();
    const target = s.dataset.sec; order.splice(order.indexOf(dragSec), 1); order.splice(order.indexOf(target), 0, dragSec);
    dragSec = null; saveSections(); renderInspector(); });
  right.addEventListener('dragend', () => { dragSec = null; right.querySelectorAll('.drag-over').forEach(e => e.classList.remove('drag-over')); });

  // ------------------------------------------------------------------ keys and polling
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape') { $('rl-drawer').hidden = true; $('rl-log-panel').hidden = true; }  // dialogs, lists, menus, picks: above
    if (event.target.matches?.('input, textarea, select') || event.altKey && event.key.toLowerCase() !== 'p') return;
    const key = event.key.toLowerCase();
    if ((event.ctrlKey || event.metaKey) && key === 'g') { event.preventDefault(); act('ui/group', {}); return; }
    if (event.ctrlKey || event.metaKey) return;
    if (event.altKey && key === 'p') { event.preventDefault(); act('ui/parent', {target: null}); }
    else if (event.key === 'A' && event.shiftKey) { event.preventDefault(); openDrawer(); }
    else if (event.key === '/') { event.preventDefault(); $('rl-filter').focus(); }
    else if (!event.shiftKey && (key === 't' || key === 'n')) toggleSide(key === 't' ? 'left' : 'right');
    else if (event.key === 'End' && inspector.owner) { event.preventDefault(); act('ui/drop', {}); }
    else if (event.key === 'F2') { event.preventDefault(); const input = $('rl-name'); if (input) { input.focus(); input.select(); } }
    else if (!event.shiftKey && key === 'p' && inspector.owner) setPicking(!(ui && ui.picking));
    else if (!event.shiftKey && {g: 'move', r: 'rotate', s: 'scale'}[key]) act('ui/gizmo', {mode: {g: 'move', r: 'rotate', s: 'scale'}[key]});
  });
  let wake = null;
  function kick() { if (wake) wake(); }
  const sleep = ms => new Promise(resolve => { const t = setTimeout(resolve, ms); wake = () => { clearTimeout(t); resolve(); }; });
  api('ui/classic', {on: false}).catch(() => {});  // viser's own panels stay hidden and are not kept up to date
  // The 3D view gets its scene over viser's websocket; the patched client marks the end of the server's replay
  // (window.__rlinfReplayDone, tools/viser_client/rlinf.patch). A page whose socket connected but replayed nothing
  // (seen in VS Code's browser on reloading) loads once more; a second time it says so instead of looping.
  setTimeout(() => {
    const received = window.__rlinfReplayDone === true, tried = sessionStorage.getItem('rl-view-retry');
    if (received) sessionStorage.removeItem('rl-view-retry');
    else if (!tried) { sessionStorage.setItem('rl-view-retry', '1'); location.reload(); }
    else toast('3D 视图没有收到场景：刷新页面再试，或重启编辑器', true);
  }, 10000);
  (async function poll() {
    for (;;) {
      try {
        ui = await api('ui/state'); renderTop();
        if (ui.start && !startShown) startDialog();
        await loadTree();
        const data = await api('ui/inspector' + (inspector.revision === null ? '' : `?revision=${inspector.revision}`));
        if (data.owner !== undefined) { if (!interacting) { inspector = data; renderInspector(); } } else inspector.revision = data.revision;
        if (renderInspector.pending) renderInspector();
      } catch (error) { /* the editor restarts: keep what is shown */ }
      await sleep(document.hidden ? 1500 : 350);
    }
  })();
})();
