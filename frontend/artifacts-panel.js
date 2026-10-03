/*
 * Code-mode Artifacts panel.
 *
 * Adds an "Artifacts" button to the header (Code mode only) that opens a
 * drawer with two modes:
 *   - Assets:    images, 3D models, fonts, video, audio, data that the user
 *                uploaded or the agent produced. "Use in project" sends a
 *                prompt so the agent places the asset AND wires it into code.
 *   - All files: every file the session has been given, created, or fixed.
 *                Chat file cards still show only what changed this turn; the
 *                complete list lives here and is shown only when opened.
 *
 * Hooks called from index.html (all guarded with window.ArtifactsPanel):
 *   ArtifactsPanel.syncMode(mode)     - show/hide button, close on leaving Code
 *   ArtifactsPanel.recordTurn(result) - files touched by a Code-mode turn
 *   ArtifactsPanel.recordUploads(list)- files the user attached this session
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
    const KIND_LABELS = {
        image: 'Image', model3d: '3D model', font: 'Font', video: 'Video', audio: 'Audio', data: 'Data',
    };
    const DEFAULT_DIR = {
        image: 'public/assets/images', model3d: 'public/assets/models', font: 'public/assets/fonts',
        video: 'public/assets/video', audio: 'public/assets/audio', data: 'public/assets/data',
    };
    const STATUS_LABELS = { given: 'Given', created: 'Created', fixed: 'Fixed' };

    // path -> { path, status: 'given'|'created'|'fixed', kind: string|null, previewUrl: string|null }
    const records = new Map();
    let open = false;
    let activeTab = 'assets';
    let btn = null;
    let drawer = null;
    let scrim = null;

    function classify(filename) {
        const dot = (filename || '').lastIndexOf('.');
        if (dot < 0) return null;
        const ext = filename.slice(dot).toLowerCase();
        for (const kind of Object.keys(ASSET_KINDS)) {
            if (ASSET_KINDS[kind].includes(ext)) return kind;
        }
        return null;
    }

    function basename(path) {
        const i = path.lastIndexOf('/');
        return i < 0 ? path : path.slice(i + 1);
    }

    function upsert(path, patch) {
        const prev = records.get(path) || { path, status: null, kind: null, previewUrl: null };
        const next = Object.assign({}, prev, patch);
        records.set(path, next);
        return next;
    }

    function recordTurn(result) {
        const files = result && result.files ? Object.keys(result.files) : [];
        for (const path of files) {
            const prev = records.get(path);
            // First time the agent touches a file it is "created"; any later
            // change, including a change to a file the user gave, is "fixed".
            const status = !prev ? 'created' : 'fixed';
            upsert(path, { status, kind: classify(path) });
        }
        refresh();
    }

    function recordUploads(list) {
        for (const rec of list || []) {
            const path = rec.filename || (rec.file && rec.file.name);
            if (!path) continue;
            const prev = records.get(path);
            upsert(path, {
                status: prev ? prev.status : 'given',
                kind: classify(path),
                previewUrl: rec.localPreview || (prev && prev.previewUrl) || null,
            });
        }
        refresh();
    }

    function assetRecords() {
        return [...records.values()].filter(r => r.kind);
    }

    function buildUsePrompt(rec) {
        const dir = DEFAULT_DIR[rec.kind] || 'public/assets';
        const name = basename(rec.path);
        return (
            `Add the ${rec.kind} asset \`${name}\` to the project at \`${dir}/${name}\` ` +
            `(use the project's existing asset folder instead if it already has one), ` +
            `then wire it into the code so it is actually used: reference it from the ` +
            `component or page that should display it, and confirm the path resolves. ` +
            `Do not only copy the file.`
        );
    }

    function usePrompt(rec) {
        const input = document.getElementById('userInput');
        if (!input || typeof sendMessage !== 'function') return;
        if (typeof currentMode !== 'undefined' && currentMode !== 'code' && typeof switchMode === 'function') {
            switchMode('code');
        }
        input.value = buildUsePrompt(rec);
        if (typeof autoResize === 'function') autoResize(input);
        setOpen(false);
        sendMessage();
    }

    function injectStyles() {
        if (document.getElementById('artifactsPanelStyles')) return;
        const css = `
        .artifacts-btn { align-items:center; gap:7px; background-color:var(--bg-input); color:var(--text-primary);
            border:1px solid transparent; padding:7px 13px; border-radius:999px; font-size:13px; font-weight:500;
            cursor:pointer; transition:background .2s ease, border-color .2s ease; }
        .artifacts-btn[hidden] { display:none; }
        .artifacts-btn:not([hidden]) { display:flex; }
        .artifacts-btn:hover { border-color:var(--border-color); }
        .artifacts-count { min-width:18px; height:18px; padding:0 5px; border-radius:999px;
            background:var(--accent); color:#fff; font-size:11px; line-height:18px; text-align:center; }
        .artifacts-count[hidden] { display:none; }
        .artifacts-scrim { position:fixed; inset:0; background:rgba(0,0,0,.35); z-index:60; }
        .artifacts-scrim[hidden] { display:none; }
        .artifacts-drawer { position:fixed; top:0; right:0; bottom:0; width:min(400px,100vw); background:var(--bg-sidebar);
            border-left:1px solid var(--border-color); z-index:61; display:flex; flex-direction:column;
            transform:translateX(100%); transition:transform .25s var(--transition-bezier); }
        .artifacts-drawer.open { transform:translateX(0); }
        .artifacts-head { display:flex; align-items:center; justify-content:space-between; padding:16px 18px 10px; }
        .artifacts-title { font:600 15px 'Inter',sans-serif; color:var(--text-primary); }
        .artifacts-close { background:transparent; border:none; color:var(--text-secondary); cursor:pointer; font-size:20px; line-height:1; padding:4px; }
        .artifacts-close:hover { color:var(--text-primary); }
        .artifacts-tabs { display:flex; gap:4px; padding:0 18px 12px; }
        .artifacts-tab { background:transparent; border:1px solid var(--border-color); color:var(--text-secondary);
            font:500 12.5px 'Inter',sans-serif; padding:6px 12px; border-radius:999px; cursor:pointer; }
        .artifacts-tab.active { background:var(--bg-input); color:var(--text-primary); }
        .artifacts-body { flex:1; overflow-y:auto; padding:4px 18px 24px; display:flex; flex-direction:column; gap:8px; }
        .artifacts-empty { color:var(--text-secondary); font:13px/1.5 'Inter',sans-serif; padding:24px 4px; }
        .artifacts-card { display:flex; gap:12px; align-items:center; padding:10px; border:1px solid var(--border-color);
            border-radius:10px; background:var(--bg-hover); }
        .artifacts-thumb { width:46px; height:46px; flex:0 0 46px; border-radius:8px; background:var(--bg-main);
            display:flex; align-items:center; justify-content:center; color:var(--text-secondary); font-size:11px; overflow:hidden; }
        .artifacts-thumb img { width:100%; height:100%; object-fit:cover; }
        .artifacts-meta { flex:1; min-width:0; }
        .artifacts-name { color:var(--text-primary); font:600 13px 'Inter',sans-serif; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
        .artifacts-sub { color:var(--text-secondary); font:11.5px 'Inter',sans-serif; margin-top:2px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
        .artifacts-action { flex:0 0 auto; border:1px solid var(--border-color); border-radius:7px; padding:6px 11px;
            background:var(--bg-main); color:var(--text-primary); font:12px 'Inter',sans-serif; font-weight:500; cursor:pointer; }
        .artifacts-action:hover { background:var(--bg-input); }
        .artifacts-badge { font:500 10.5px 'Inter',sans-serif; padding:2px 7px; border-radius:999px; border:1px solid var(--border-color); color:var(--text-secondary); }
        .artifacts-badge.fixed { color:#aff5b4; border-color:rgba(46,160,67,.5); }
        .artifacts-badge.created { color:#9ecbff; border-color:rgba(77,163,255,.5); }
        `;
        const style = document.createElement('style');
        style.id = 'artifactsPanelStyles';
        style.textContent = css;
        document.head.appendChild(style);
    }

    function el(tag, cls, text) {
        const node = document.createElement(tag);
        if (cls) node.className = cls;
        if (text !== undefined) node.textContent = text;
        return node;
    }

    function renderAssets(body) {
        const assets = assetRecords();
        if (!assets.length) {
            body.appendChild(el('div', 'artifacts-empty',
                'No assets yet. Upload an image, 3D model, font, or other media in Code mode and it will appear here.'));
            return;
        }
        for (const rec of assets) {
            const card = el('div', 'artifacts-card');
            const thumb = el('div', 'artifacts-thumb');
            if (rec.kind === 'image' && rec.previewUrl) {
                const img = document.createElement('img');
                img.src = rec.previewUrl;
                img.alt = '';
                thumb.appendChild(img);
            } else {
                thumb.textContent = (KIND_LABELS[rec.kind] || 'File').slice(0, 6);
            }
            const meta = el('div', 'artifacts-meta');
            meta.appendChild(el('div', 'artifacts-name', basename(rec.path)));
            meta.appendChild(el('div', 'artifacts-sub', `${KIND_LABELS[rec.kind]} · ${rec.path}`));
            const action = el('button', 'artifacts-action', 'Use in project');
            action.type = 'button';
            action.addEventListener('click', () => usePrompt(rec));
            card.append(thumb, meta, action);
            body.appendChild(card);
        }
    }

    function renderAllFiles(body) {
        const all = [...records.values()];
        if (!all.length) {
            body.appendChild(el('div', 'artifacts-empty',
                'No files yet. Files you give, create, or fix in Code mode will be listed here.'));
            return;
        }
        for (const rec of all) {
            const card = el('div', 'artifacts-card');
            const meta = el('div', 'artifacts-meta');
            meta.appendChild(el('div', 'artifacts-name', basename(rec.path)));
            meta.appendChild(el('div', 'artifacts-sub', rec.path));
            const badge = el('span', `artifacts-badge ${rec.status}`, STATUS_LABELS[rec.status] || '');
            card.append(meta, badge);
            body.appendChild(card);
        }
    }

    function render() {
        if (!drawer) return;
        const body = drawer.querySelector('.artifacts-body');
        body.replaceChildren();
        drawer.querySelectorAll('.artifacts-tab').forEach(t => {
            t.classList.toggle('active', t.dataset.tab === activeTab);
            t.setAttribute('aria-selected', String(t.dataset.tab === activeTab));
        });
        if (activeTab === 'assets') renderAssets(body);
        else renderAllFiles(body);
    }

    function refresh() {
        if (btn) {
            const count = records.size;
            const badge = document.getElementById('artifactsCount');
            if (badge) { badge.textContent = String(count); badge.hidden = count === 0; }
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
            t.addEventListener('click', () => { activeTab = key; render(); });
            tabs.appendChild(t);
        }

        drawer.append(head, tabs, el('div', 'artifacts-body'));
        document.body.append(scrim, drawer);

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && open) setOpen(false);
        });
    }

    function syncMode(mode) {
        if (!btn) return;
        btn.hidden = mode !== 'code';
        if (mode !== 'code' && open) setOpen(false);
    }

    function init() {
        build();
        // Pick up the current mode on load.
        const mode = typeof currentMode !== 'undefined' ? currentMode : 'chat';
        syncMode(mode);
    }

    window.ArtifactsPanel = {
        init,
        toggle: () => setOpen(!open),
        syncMode,
        recordTurn,
        recordUploads,
    };

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
