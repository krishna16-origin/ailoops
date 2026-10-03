/*
 * Code-mode Artifacts panel (drawer opened by the #artifactsBtn header button).
 *
 * Two modes:
 *   - Assets:    media the user DROPS or CHOOSES in this tab's own drop zone
 *                (images, 3D models, fonts, video, audio, data). "Use in
 *                project" uploads the file into the sandbox project under its
 *                kind folder and asks the agent to wire it in.
 *   - All files: files the session was given, created, or fixed. Images are
 *                never listed here. Chat file cards still show only files
 *                changed this turn.
 *
 * Composer '+' uploads are "given" files for All files only. They never
 * become assets and can never be used in the project.
 *
 * Hooks from index.html (all guarded by window.ArtifactsPanel):
 *   syncMode(mode)      show the button only in Code mode
 *   recordTurn(result)  files touched by a Code-mode turn
 *   recordUploads(list) composer '+' uploads (non-images only are kept)
 */
(function () {
    'use strict';

    const ASSET_KINDS = {
        image: ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg', '.avif', '.ico'],
        model3d: ['.glb', '.gltf', '.obj', '.fbx', '.stl', '.usdz', '.ply'],
        font: ['.woff', '.woff2', '.ttf', '.otf'],
        video: ['.mp4', '.webm', '.mov'],
        audio: ['.mp3', '.wav', '.ogg', '.m4a'],
        data: ['.csv', '.json', '.geojson', '.xlsx'],
    };
    const KIND_LABELS = { image: 'Image', model3d: '3D model', font: 'Font', video: 'Video', audio: 'Audio', data: 'Data' };
    const DEFAULT_DIR = {
        image: 'public/assets/images', model3d: 'public/assets/models', font: 'public/assets/fonts',
        video: 'public/assets/video', audio: 'public/assets/audio', data: 'public/assets/data',
    };
    const STATUS_LABELS = { given: 'Given', created: 'Created', fixed: 'Fixed' };
    const ACCEPT = Object.values(ASSET_KINDS).flat().join(',');

    // All-files entries: path -> { path, status, kind }. Images are never stored here.
    const files = new Map();
    // Assets: only items added through this tab's drop zone.
    const assets = [];
    let nextAssetId = 1;

    let open = false;
    let activeTab = 'assets';
    let btn = null, countEl = null, drawer = null, scrim = null, notice = '';

    function classify(filename) {
        const dot = (filename || '').lastIndexOf('.');
        if (dot < 0) return null;
        const ext = filename.slice(dot).toLowerCase();
        for (const kind of Object.keys(ASSET_KINDS)) {
            if (ASSET_KINDS[kind].includes(ext)) return kind;
        }
        return null;
    }
    const basename = (p) => { const i = p.lastIndexOf('/'); return i < 0 ? p : p.slice(i + 1); };

    function isImagePath(p) { return classify(p) === 'image'; }

    function recordTurn(result) {
        const paths = result && result.files ? Object.keys(result.files) : [];
        for (const path of paths) {
            if (isImagePath(path)) continue;
            const prev = files.get(path);
            files.set(path, { path, status: prev ? 'fixed' : 'created', kind: classify(path) });
        }
        refresh();
    }

    function recordUploads(list) {
        for (const rec of list || []) {
            const path = rec.filename || (rec.file && rec.file.name);
            if (!path || isImagePath(path)) continue; // '+' images are not shown anywhere in this panel
            const prev = files.get(path);
            files.set(path, { path, status: prev ? prev.status : 'given', kind: classify(path) });
        }
        refresh();
    }

    function addAssetFiles(fileList) {
        let rejected = 0;
        for (const file of Array.from(fileList || [])) {
            const kind = classify(file.name);
            if (!kind) { rejected++; continue; }
            assets.push({
                id: nextAssetId++, name: file.name, kind, file,
                previewUrl: kind === 'image' ? URL.createObjectURL(file) : null,
            });
        }
        notice = rejected ? `${rejected} file(s) skipped: not a supported asset type.` : '';
        refresh();
    }

    function sessionIdForCode() {
        return (typeof modes !== 'undefined' && modes.code && modes.code.sessionId) || null;
    }

    async function useInProject(asset) {
        const sessionId = sessionIdForCode();
        if (!sessionId) {
            notice = 'Send one message in Code mode first so the project exists, then try again.';
            render();
            return;
        }
        notice = `Uploading ${asset.name} to the project…`;
        render();
        const form = new FormData();
        form.append('session_id', sessionId);
        form.append('file', asset.file, asset.name);
        let data;
        try {
            const res = await fetch(`${typeof API_BASE !== 'undefined' ? API_BASE : ''}/project-asset`, { method: 'POST', body: form });
            data = await res.json();
            if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
        } catch (err) {
            notice = `Could not add ${asset.name} to the project: ${err.message}`;
            render();
            return;
        }
        const input = document.getElementById('userInput');
        if (!input || typeof sendMessage !== 'function') return;
        if (typeof currentMode !== 'undefined' && currentMode !== 'code' && typeof switchMode === 'function') switchMode('code');
        input.value =
            `The ${asset.kind} asset is now in the project at \`${data.path}\`. ` +
            `Wire it into the code so it is actually used: reference it from the component or page that should display it, ` +
            `and confirm the path resolves. Do not only mention the file.`;
        if (typeof autoResize === 'function') autoResize(input);
        notice = '';
        setOpen(false);
        sendMessage();
    }

    function injectStyles() {
        if (document.getElementById('artifactsPanelStyles')) return;
        const css = `
        .artifacts-btn[hidden] { display:none; }
        .artifacts-btn:not([hidden]) { display:flex; align-items:center; gap:7px; background-color:var(--bg-input); color:var(--text-primary);
            border:1px solid transparent; padding:7px 13px; border-radius:999px; font-size:13px; font-weight:500; cursor:pointer; }
        .artifacts-btn:hover { border-color:var(--border-color); }
        .artifacts-count { min-width:18px; height:18px; padding:0 5px; border-radius:999px; background:var(--accent); color:#fff; font-size:11px; line-height:18px; text-align:center; }
        .artifacts-count[hidden] { display:none; }
        .artifacts-scrim { position:fixed; inset:0; background:rgba(0,0,0,.35); z-index:60; }
        .artifacts-scrim[hidden] { display:none; }
        .artifacts-drawer { position:fixed; top:0; right:0; bottom:0; width:min(400px,100vw); background:var(--bg-sidebar);
            border-left:1px solid var(--border-color); z-index:61; display:flex; flex-direction:column; transform:translateX(100%); transition:transform .25s var(--transition-bezier); }
        .artifacts-drawer.open { transform:translateX(0); }
        .artifacts-head { display:flex; align-items:center; justify-content:space-between; padding:16px 18px 10px; }
        .artifacts-title { font:600 15px 'Inter',sans-serif; color:var(--text-primary); }
        .artifacts-close { background:transparent; border:none; color:var(--text-secondary); cursor:pointer; font-size:20px; line-height:1; padding:4px; }
        .artifacts-close:hover { color:var(--text-primary); }
        .artifacts-tabs { display:flex; gap:4px; padding:0 18px 12px; }
        .artifacts-tab { background:transparent; border:1px solid var(--border-color); color:var(--text-secondary); font:500 12.5px 'Inter',sans-serif; padding:6px 12px; border-radius:999px; cursor:pointer; }
        .artifacts-tab.active { background:var(--bg-input); color:var(--text-primary); }
        .artifacts-body { flex:1; overflow-y:auto; padding:4px 18px 24px; display:flex; flex-direction:column; gap:8px; }
        .artifacts-drop { border:1px dashed var(--border-color); border-radius:12px; padding:22px 14px; text-align:center; color:var(--text-secondary);
            font:13px/1.5 'Inter',sans-serif; cursor:pointer; background:var(--bg-hover); }
        .artifacts-drop.dragover { border-color:var(--accent); color:var(--text-primary); }
        .artifacts-drop strong { color:var(--text-primary); font-weight:600; }
        .artifacts-notice { color:var(--text-secondary); font:12px/1.4 'Inter',sans-serif; }
        .artifacts-empty { color:var(--text-secondary); font:13px/1.5 'Inter',sans-serif; padding:12px 4px; }
        .artifacts-card { display:flex; gap:12px; align-items:center; padding:10px; border:1px solid var(--border-color); border-radius:10px; background:var(--bg-hover); }
        .artifacts-thumb { width:46px; height:46px; flex:0 0 46px; border-radius:8px; background:var(--bg-main); display:flex; align-items:center; justify-content:center; color:var(--text-secondary); font-size:11px; overflow:hidden; }
        .artifacts-thumb img { width:100%; height:100%; object-fit:cover; }
        .artifacts-meta { flex:1; min-width:0; }
        .artifacts-name { color:var(--text-primary); font:600 13px 'Inter',sans-serif; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
        .artifacts-sub { color:var(--text-secondary); font:11.5px 'Inter',sans-serif; margin-top:2px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
        .artifacts-action { flex:0 0 auto; border:1px solid var(--border-color); border-radius:7px; padding:6px 11px; background:var(--bg-main); color:var(--text-primary); font:12px 'Inter',sans-serif; font-weight:500; cursor:pointer; }
        .artifacts-action:hover { background:var(--bg-input); }
        .artifacts-badge { font:500 10.5px 'Inter',sans-serif; padding:2px 7px; border-radius:999px; border:1px solid var(--border-color); color:var(--text-secondary); }
        .artifacts-badge.fixed { color:#aff5b4; border-color:rgba(46,160,67,.5); }
        .artifacts-badge.created { color:#9ecbff; border-color:rgba(77,163,255,.5); }`;
        const style = document.createElement('style');
        style.id = 'artifactsPanelStyles';
        style.textContent = css;
        document.head.appendChild(style);
    }

    function el(tag, cls, text) {
        const n = document.createElement(tag);
        if (cls) n.className = cls;
        if (text !== undefined) n.textContent = text;
        return n;
    }

    function renderAssetsTab(body) {
        const input = el('input');
        input.type = 'file';
        input.multiple = true;
        input.accept = ACCEPT;
        input.hidden = true;
        input.addEventListener('change', () => { addAssetFiles(input.files); input.value = ''; });

        const drop = el('div', 'artifacts-drop');
        drop.setAttribute('role', 'button');
        drop.tabIndex = 0;
        drop.append(el('strong', null, 'Drop media here'), document.createTextNode(' or click to choose. Images, 3D models, fonts, video, audio, data.'));
        drop.addEventListener('click', () => input.click());
        drop.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); } });
        drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('dragover'); });
        drop.addEventListener('dragleave', () => drop.classList.remove('dragover'));
        drop.addEventListener('drop', (e) => { e.preventDefault(); drop.classList.remove('dragover'); addAssetFiles(e.dataTransfer.files); });
        body.append(drop, input);

        if (notice) body.appendChild(el('div', 'artifacts-notice', notice));

        if (!assets.length) {
            body.appendChild(el('div', 'artifacts-empty', 'No assets yet. Assets appear here only when you drop or choose them above.'));
            return;
        }
        for (const a of assets) {
            const card = el('div', 'artifacts-card');
            const thumb = el('div', 'artifacts-thumb');
            if (a.previewUrl) {
                const img = document.createElement('img');
                img.src = a.previewUrl;
                img.alt = '';
                thumb.appendChild(img);
            } else {
                thumb.textContent = (KIND_LABELS[a.kind] || 'File').slice(0, 6);
            }
            const meta = el('div', 'artifacts-meta');
            meta.append(el('div', 'artifacts-name', a.name), el('div', 'artifacts-sub', `${KIND_LABELS[a.kind]} · ${DEFAULT_DIR[a.kind]}/${a.name}`));
            const action = el('button', 'artifacts-action', 'Use in project');
            action.type = 'button';
            action.addEventListener('click', () => useInProject(a));
            card.append(thumb, meta, action);
            body.appendChild(card);
        }
    }

    function renderFilesTab(body) {
        const all = [...files.values()];
        if (!all.length) {
            body.appendChild(el('div', 'artifacts-empty', 'No files yet. Files you give, create, or fix in Code mode appear here. Images are not listed.'));
            return;
        }
        for (const rec of all) {
            const card = el('div', 'artifacts-card');
            const meta = el('div', 'artifacts-meta');
            meta.append(el('div', 'artifacts-name', basename(rec.path)), el('div', 'artifacts-sub', rec.path));
            card.append(meta, el('span', `artifacts-badge ${rec.status}`, STATUS_LABELS[rec.status] || ''));
            body.appendChild(card);
        }
    }

    function render() {
        if (!drawer) return;
        const body = drawer.querySelector('.artifacts-body');
        body.replaceChildren();
        drawer.querySelectorAll('.artifacts-tab').forEach(t => {
            const on = t.dataset.tab === activeTab;
            t.classList.toggle('active', on);
            t.setAttribute('aria-selected', String(on));
        });
        if (activeTab === 'assets') renderAssetsTab(body);
        else renderFilesTab(body);
    }

    function refresh() {
        if (countEl) {
            const count = files.size + assets.length;
            countEl.textContent = String(count);
            countEl.hidden = count === 0;
        }
        if (open) render();
    }

    function setOpen(next) {
        open = !!next;
        if (drawer) drawer.classList.toggle('open', open);
        if (scrim) scrim.hidden = !open;
        if (btn) btn.setAttribute('aria-expanded', String(open));
        if (open) render();
    }

    function build() {
        injectStyles();
        btn = document.getElementById('artifactsBtn');
        countEl = document.getElementById('artifactsCount');
        if (btn) btn.addEventListener('click', () => setOpen(!open));

        scrim = el('div', 'artifacts-scrim');
        scrim.hidden = true;
        scrim.addEventListener('click', () => setOpen(false));

        drawer = el('aside', 'artifacts-drawer');
        drawer.setAttribute('role', 'dialog');
        drawer.setAttribute('aria-label', 'Artifacts');

        const head = el('div', 'artifacts-head');
        head.append(el('div', 'artifacts-title', 'Artifacts'));
        const close = el('button', 'artifacts-close', '×');
        close.type = 'button';
        close.setAttribute('aria-label', 'Close artifacts');
        close.addEventListener('click', () => setOpen(false));
        head.appendChild(close);

        const tabs = el('div', 'artifacts-tabs');
        tabs.setAttribute('role', 'tablist');
        for (const [key, label] of [['assets', 'Assets'], ['files', 'All files']]) {
            const t = el('button', 'artifacts-tab', label);
            t.type = 'button';
            t.dataset.tab = key;
            t.setAttribute('role', 'tab');
            t.addEventListener('click', () => { activeTab = key; notice = ''; render(); });
            tabs.appendChild(t);
        }

        drawer.append(head, tabs, el('div', 'artifacts-body'));
        document.body.append(scrim, drawer);

        document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && open) setOpen(false); });
    }

    function syncMode(mode) {
        if (!btn) return;
        btn.hidden = mode !== 'code';
        if (mode !== 'code' && open) setOpen(false);
    }

    function init() {
        build();
        syncMode(typeof currentMode !== 'undefined' ? currentMode : 'chat');
        refresh();
    }

    window.ArtifactsPanel = { init, toggle: () => setOpen(!open), syncMode, recordTurn, recordUploads };

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
    else init();
})();
