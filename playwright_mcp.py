"""Playwright-backed browser tools for the ailoops MCP gateway.

Each MCP session gets a persistent browser context and a set of pages (tabs).
The profile is opt-in through PLAYWRIGHT_PERSIST_SESSIONS; otherwise the profile
is temporary and is deleted when the context closes. Every navigation creates a
new page so browser work never replaces an ailoops web-app page.
"""
from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from playwright.async_api import BrowserContext, Error as PlaywrightError, Page, async_playwright

MAX_TEXT = 24_000
MAX_ITEMS = 200

TOOLS = [
    "browser_status", "browser_open", "browser_navigate", "browser_search",
    "browser_click", "browser_click_link", "browser_type", "browser_fill",
    "browser_select", "browser_check", "browser_upload", "browser_download",
    "browser_content", "browser_extract_text", "browser_extract_links",
    "browser_read_table", "browser_screenshot", "browser_scroll",
    "browser_tabs", "browser_new_tab", "browser_switch_tab", "browser_popup",
    "browser_close_tab", "browser_close_session", "browser_evaluate",
    "browser_test", "browser_inspect", "browser_verify", "browser_record",
]


def _limit(value: Any, limit: int = MAX_TEXT) -> Any:
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "\n…[truncated]"
    return value


def _safe_name(value: str, fallback: str = "browser-file") -> str:
    name = Path(value or fallback).name
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name)[:160] or fallback


def _workspace() -> Path:
    root = Path(os.getenv("MCP_WORKSPACE_ROOT", os.getcwd())).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _session_dir(session_id: str) -> Path:
    root = Path(os.getenv("PLAYWRIGHT_DOWNLOAD_DIR", str(_workspace() / ".playwright"))).resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / _safe_name(session_id, "default")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _url(value: str) -> str:
    value = str(value or "").strip()
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("A complete http(s) URL is required")
    return value


class BrowserManager:
    def __init__(self) -> None:
        self._playwright = None
        self._contexts: dict[str, BrowserContext] = {}
        self._traces: set[str] = set()
        self._lock = asyncio.Lock()
        self._browser_install_lock = asyncio.Lock()
        self._browser_install_attempted = False

    async def _install_browser_if_needed(self) -> None:
        """Download Chromium into Playwright's cache when a Render build omitted it."""
        if self._browser_install_attempted:
            return
        async with self._browser_install_lock:
            if self._browser_install_attempted:
                return
            self._browser_install_attempted = True
            if os.getenv("PLAYWRIGHT_AUTO_INSTALL", "1").lower() in {"0", "false", "no"}:
                raise RuntimeError(
                    "Playwright Chromium is not installed. Run `python -m playwright install chromium` "
                    "during the Render build or set PLAYWRIGHT_AUTO_INSTALL=1."
                )
            try:
                await asyncio.to_thread(
                    subprocess.run,
                    [sys.executable, "-m", "playwright", "install", "chromium"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
            except Exception as exc:
                detail = getattr(exc, "stderr", "") or str(exc)
                raise RuntimeError(f"Playwright could not install Chromium: {detail[-800:]}") from exc

    async def _launch_context(self, profile: str | None, launch_args: dict[str, Any]) -> BrowserContext:
        try:
            if profile:
                return await self._playwright.chromium.launch_persistent_context(profile, **launch_args)
            browser = await self._playwright.chromium.launch(**launch_args)
            return await browser.new_context(accept_downloads=True)
        except PlaywrightError as exc:
            if "Executable doesn't exist" not in str(exc) and "executable doesn't exist" not in str(exc):
                raise
            await self._install_browser_if_needed()
            if profile:
                return await self._playwright.chromium.launch_persistent_context(profile, **launch_args)
            browser = await self._playwright.chromium.launch(**launch_args)
            return await browser.new_context(accept_downloads=True)

    async def _context(self, session_id: str) -> BrowserContext:
        async with self._lock:
            context = self._contexts.get(session_id)
            if context:
                return context
            if self._playwright is None:
                self._playwright = await async_playwright().start()
            persistent = os.getenv("PLAYWRIGHT_PERSIST_SESSIONS", "0").lower() in {"1", "true", "yes"}
            profile = _session_dir(session_id) if persistent else None
            headless = os.getenv("PLAYWRIGHT_HEADLESS", "1").lower() not in {"0", "false", "no"}
            executable = os.getenv("PLAYWRIGHT_EXECUTABLE_PATH")
            # Render runs the service in a restricted container without a usable
            # Chromium sandbox or large /dev/shm. These flags are safe for the
            # isolated browser worker and prevent the browser process from
            # exiting immediately after it is found and launched.
            chromium_args = [
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
            ]
            extra_args = os.getenv("PLAYWRIGHT_ARGS", "").split()
            launch_args: dict[str, Any] = {"headless": headless, "args": chromium_args + extra_args}
            if executable:
                launch_args["executable_path"] = executable
            elif os.getenv("PLAYWRIGHT_BROWSER_CHANNEL"):
                launch_args["channel"] = os.getenv("PLAYWRIGHT_BROWSER_CHANNEL")
            context = await self._launch_context(str(profile) if profile else None, launch_args)
            self._contexts[session_id] = context
            return context

    async def _page(self, session_id: str, args: dict[str, Any], new_tab: bool = False) -> Page:
        context = await self._context(session_id)
        pages = context.pages
        if not pages or new_tab:
            page = await context.new_page()
        else:
            index = int(args.get("tab_index", 0))
            page = pages[index] if 0 <= index < len(pages) else pages[-1]
        page.set_default_timeout(float(args.get("timeout_ms", 15000)))
        return page

    @staticmethod
    def _selector(args: dict[str, Any], key: str = "selector") -> str:
        selector = str(args.get(key) or "").strip()
        if not selector:
            raise ValueError(f"{key} is required")
        return selector

    async def call(self, tool: str, args: dict[str, Any], session_id: str) -> Any:
        if tool == "browser_status":
            return {"status": "ready", "tools": TOOLS, "new_tab_policy": True,
                    "persistent_login": os.getenv("PLAYWRIGHT_PERSIST_SESSIONS", "0").lower() in {"1", "true", "yes"}}
        if tool == "browser_close_session":
            context = self._contexts.pop(session_id, None)
            if context:
                if session_id in self._traces:
                    await context.tracing.stop(path=str(_session_dir(session_id) / "browser-trace.zip"))
                    self._traces.discard(session_id)
                await context.close()
                if not self._contexts and self._playwright is not None:
                    await self._playwright.stop()
                    self._playwright = None
            return {"closed": bool(context), "session_id": session_id}
        if tool == "browser_close_tab":
            page = await self._page(session_id, args)
            await page.close()
            return {"closed": True, "remaining_tabs": len(page.context.pages)}

        if tool in {"browser_open", "browser_new_tab"}:
            page = await self._page(session_id, args, new_tab=True)
            url = _url(args.get("url"))
            await page.goto(url, wait_until=args.get("wait_until", "domcontentloaded"))
            return await self._page_info(page, open_in_new_tab=True)

        page = await self._page(session_id, args, new_tab=tool in {"browser_navigate", "browser_search"})
        if tool == "browser_navigate":
            await page.goto(_url(args.get("url")), wait_until=args.get("wait_until", "domcontentloaded"))
            return await self._page_info(page, open_in_new_tab=True)
        if tool == "browser_search":
            query = str(args.get("query") or "").strip()
            if not query: raise ValueError("query is required")
            engine = args.get("engine", "google")
            base = "https://www.google.com/search?q=" if engine == "google" else "https://www.bing.com/search?q="
            from urllib.parse import quote_plus
            await page.goto(base + quote_plus(query), wait_until="domcontentloaded")
            return await self._page_info(page, open_in_new_tab=True)
        if tool in {"browser_click", "browser_click_link"}:
            selector = str(args.get("selector") or "").strip()
            target = None
            if tool == "browser_click_link":
                if not selector:
                    text = str(args.get("text") or "").strip()
                    if not text: raise ValueError("text or selector is required")
                    target = page.get_by_role("link", name=text).first
            elif not selector:
                raise ValueError("selector is required")
            if target is None:
                target = page.locator(selector).first
            before = set(page.context.pages)
            opens_tab = await target.get_attribute("target") == "_blank"
            if not opens_tab:
                if tool == "browser_click_link":
                    href = await target.get_attribute("href")
                    if href:
                        from urllib.parse import urljoin
                        new_page = await self._page(session_id, args, new_tab=True)
                        await new_page.goto(urljoin(page.url, href), wait_until="domcontentloaded")
                        return await self._page_info(new_page, open_in_new_tab=True)
                await target.click()
                return await self._page_info(page, open_in_new_tab=True, popup_created=len(page.context.pages) > len(before))
            try:
                async with page.expect_popup(timeout=1000) as popup_info:
                    await target.click()
                popup = await popup_info.value
                await popup.wait_for_load_state("domcontentloaded")
                return await self._page_info(popup, open_in_new_tab=True)
            except Exception:
                await target.click()
                return await self._page_info(page, open_in_new_tab=True, popup_created=len(page.context.pages) > len(before))
        if tool in {"browser_type", "browser_fill"}:
            selector = self._selector(args)
            value = str(args.get("text", args.get("value", "")))
            locator = page.locator(selector).first
            if tool == "browser_fill": await locator.fill(value)
            else: await locator.press_sequentially(value, delay=float(args.get("delay_ms", 0)))
            return {"ok": True, "selector": selector, "value_length": len(value)}
        if tool == "browser_select":
            selector = self._selector(args); values = args.get("values", args.get("value"))
            if not isinstance(values, list): values = [values]
            return {"selected": await page.locator(selector).first.select_option([str(v) for v in values if v is not None])}
        if tool == "browser_check":
            selector = self._selector(args); checked = bool(args.get("checked", True))
            await page.locator(selector).first.set_checked(checked)
            return {"selector": selector, "checked": await page.locator(selector).first.is_checked()}
        if tool == "browser_upload":
            selector = self._selector(args); paths = args.get("paths", args.get("path"))
            if not isinstance(paths, list): paths = [paths]
            resolved = [str((_workspace() / str(p)).resolve()) for p in paths]
            for p in resolved:
                if _workspace() not in Path(p).parents and Path(p) != _workspace(): raise ValueError("Upload path is outside the workspace")
            await page.locator(selector).first.set_input_files(resolved)
            return {"uploaded": [_safe_name(p) for p in resolved]}
        if tool == "browser_download":
            selector = self._selector(args)
            async with page.expect_download() as download_info:
                await page.locator(selector).first.click()
            download = await download_info.value
            target = _session_dir(session_id) / _safe_name(args.get("filename") or download.suggested_filename)
            await download.save_as(str(target))
            return {"path": str(target), "filename": target.name, "url": download.url}
        if tool in {"browser_content", "browser_extract_text"}:
            return {"url": page.url, "title": await page.title(), "text": _limit(await page.locator("body").inner_text())}
        if tool == "browser_extract_links":
            links = await page.locator("a").evaluate_all("els => els.slice(0, 200).map(a => ({text:(a.innerText||a.textContent||'').trim(), url:a.href})).filter(x => x.url)")
            return {"url": page.url, "links": links}
        if tool == "browser_read_table":
            selector = args.get("selector", "table")
            tables = await page.locator(selector).evaluate_all("els => els.slice(0, 50).map(t => [...t.rows].slice(0,200).map(r => [...r.cells].map(c => (c.innerText||c.textContent||'').trim())))")
            return {"tables": tables}
        if tool == "browser_screenshot":
            target = _session_dir(session_id) / _safe_name(args.get("filename") or f"screenshot-{uuid.uuid4().hex[:8]}.png")
            await page.screenshot(path=str(target), full_page=bool(args.get("full_page", False)), type="png")
            return {"path": str(target), "filename": target.name, "url": page.url}
        if tool == "browser_scroll":
            amount = int(args.get("amount", 700)); direction = -1 if str(args.get("direction", "down")).lower() == "up" else 1
            await page.mouse.wheel(0, amount * direction)
            return {"scrolled": amount * direction, "url": page.url}
        if tool == "browser_tabs":
            return {"tabs": [await self._page_info(p, open_in_new_tab=True) for p in page.context.pages], "active_url": page.url}
        if tool == "browser_switch_tab":
            index = int(args.get("tab_index", 0)); pages = page.context.pages
            if index < 0 or index >= len(pages): raise ValueError("tab_index is out of range")
            return await self._page_info(pages[index], open_in_new_tab=True)
        if tool == "browser_popup":
            return {"popups": [await self._page_info(p, open_in_new_tab=True) for p in page.context.pages if p is not page]}
        if tool == "browser_evaluate":
            expression = str(args.get("expression") or args.get("script") or "").strip()
            if not expression: raise ValueError("expression is required")
            return {"result": _limit(await page.evaluate(expression)), "url": page.url}
        if tool == "browser_test":
            return {"passed": await page.locator(self._selector(args)).count() > 0, "url": page.url}
        if tool == "browser_inspect":
            selector = self._selector(args); loc = page.locator(selector).first
            return {"count": await page.locator(selector).count(), "tag": await loc.evaluate("el => el.tagName"), "text": _limit(await loc.inner_text(), 4000), "attributes": await loc.evaluate("el => Object.fromEntries([...el.attributes].map(a => [a.name,a.value]))")}
        if tool == "browser_verify":
            selector = self._selector(args); loc = page.locator(selector).first
            return {"exists": await page.locator(selector).count() > 0, "visible": await loc.is_visible() if await page.locator(selector).count() else False, "text": _limit(await loc.inner_text(), 4000) if await page.locator(selector).count() else ""}
        if tool == "browser_record":
            action = str(args.get("action", "start")).lower()
            if action == "start":
                await page.context.tracing.start(screenshots=True, snapshots=True, sources=True)
                self._traces.add(session_id)
                return {"recording": True, "action": "started", "url": page.url}
            if action == "stop":
                target = _session_dir(session_id) / _safe_name(args.get("filename") or "browser-trace.zip")
                if session_id not in self._traces:
                    return {"recording": False, "action": "not_started", "url": page.url}
                await page.context.tracing.stop(path=str(target))
                self._traces.discard(session_id)
                return {"recording": False, "action": "stopped", "path": str(target), "filename": target.name, "url": page.url}
            raise ValueError("browser_record action must be start or stop")
        raise ValueError(f"Unknown Playwright tool: {tool}")

    @staticmethod
    def _quote(text: str) -> str:
        return '"' + text.replace('"', '\\"') + '"'

    async def _page_info(self, page: Page, open_in_new_tab: bool = False, **extra: Any) -> dict:
        return {"url": page.url, "title": await page.title(), "tab_index": page.context.pages.index(page),
                "open_in_new_tab": open_in_new_tab, "browser": "chrome/chromium", **extra}


manager = BrowserManager()


async def call(tool: str, args: dict[str, Any], session_id: str) -> Any:
    return await manager.call(tool, args, session_id)
