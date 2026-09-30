"""Real-time voice over a single WebSocket.

    browser mic --(16 kHz PCM, VAD-gated)--> /ws/voice
        -> Groq Whisper (speech-to-text)
        -> Groq LLM (streamed)  -> sentence chunker
        -> Fish Audio TTS (streamed PCM, one task per sentence, played in order)
    <-- binary PCM frames + JSON events

Why this is faster than the old POST /voice-chat flow:
  * one persistent socket: no per-turn TLS/HTTP handshake, warm upstream pools
  * the mic audio is uploaded *while the user is still talking*, so STT starts
    the instant the client detects the end of speech
  * TTS audio is forwarded chunk-by-chunk as raw PCM (no base64, no mp3 decode)
  * the first spoken piece is flushed after a clause / ~7 words, not a full sentence
  * a new utterance or an explicit `interrupt` cancels LLM + TTS work immediately

Wire protocol
-------------
client -> server (JSON):
    {"type":"config","session_id":..,"mcp_servers":[..],"voice":".."}
    {"type":"speech_start"} / {"type":"speech_end"} / {"type":"speech_abort"}
    {"type":"interrupt"}                       # user barged in over the assistant
    {"type":"text","text":".."}                # typed message, skips STT
    {"type":"ping","t":..}
client -> server (binary): raw little-endian int16 mono 16 kHz PCM, only while speaking

server -> client (JSON):
    ready{pcm_rate} state{state,turn} transcript{text,turn} assistant_text{text,turn}
    tts_fallback{text,turn} turn_done{turn,response} cancelled{turn} metrics{..}
    error{message} pong{t}
server -> client (binary): [kind:u8][turn:u32 BE][payload]
    kind 1 = int16 mono PCM at `pcm_rate`, kind 2 = a complete mp3 (fallback tier)
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import re
import struct
import time
import wave
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional, Tuple

import httpx
import numpy as np
from fastapi import APIRouter, WebSocket

# --------------------------------------------------------------------------- config
STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
STT_MODEL = os.getenv("VOICE_STT_MODEL", "whisper-large-v3-turbo")
# Pinning the language skips Whisper's language-detection pass (faster). Set to "" for auto.
STT_LANG = os.getenv("VOICE_STT_LANG", "en").strip()
FISH_TTS_URL = "https://api.fish.audio/v1/tts"
FISH_MODEL = os.getenv("FISH_MODEL", "s2.1-pro-free")
FISH_LATENCY = os.getenv("FISH_LATENCY", "low")
FISH_REFERENCE_ID = os.getenv("FISH_REFERENCE_ID", "").strip()
PCM_RATE = int(os.getenv("VOICE_PCM_RATE", "24000"))
HOLD_MS = int(os.getenv("VOICE_HOLD_MS", "150"))          # minimal grace period for unfinished speech
TTS_CONCURRENCY = int(os.getenv("VOICE_TTS_CONCURRENCY", "3"))

IN_RATE = 16000
MIN_UTTERANCE_BYTES = int(IN_RATE * 2 * 0.25)              # ignore clicks / coughs < 250 ms
MAX_UTTERANCE_BYTES = IN_RATE * 2 * 60                     # hard cap: 60 s per utterance
KIND_PCM, KIND_MP3 = 1, 2


# --------------------------------------------------------------------------- helpers
def pcm16_to_wav(pcm: bytes, rate: int = IN_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def pcm_rms(pcm: bytes) -> float:
    n = len(pcm) // 2
    if n == 0:
        return 0.0
    a = np.frombuffer(pcm[: n * 2], dtype="<i2").astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(a * a)))


_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*(?:\s+|$)|\n+")
_CLAUSE_END = re.compile(r"(?<=[,;:\u2014])\s+")
_ABBREV = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Prof|St|vs|etc|e\.g|i\.e)\.$", re.I)
_FIRST_CLAUSE_MIN_CHARS = 8
_FIRST_FLUSH_WORDS = 3


def pop_speakable(buffer: str, first: bool) -> Tuple[List[str], str]:
    """Pull speakable pieces out of a streaming LLM buffer.

    Later pieces are whole sentences. The *first* piece is flushed as early as
    possible - at a clause boundary, or after ~7 words - so speech starts fast."""
    out: List[str] = []
    while True:
        cut = None
        m = _SENTENCE_END.search(buffer)
        while m and (len(buffer[: m.start()].strip()) < 8 or _ABBREV.search(buffer[: m.start() + 1].strip())):
            m = _SENTENCE_END.search(buffer, m.end())
        if m:
            cut = (m.start(), m.end())
        elif first and not out:
            c = _CLAUSE_END.search(buffer)
            if c and len(buffer[: c.start()].strip()) >= _FIRST_CLAUSE_MIN_CHARS:
                cut = (c.start(), c.end())
            else:
                words = buffer.split(" ")
                # N complete words + a partial one still arriving
                if len(words) > _FIRST_FLUSH_WORDS:
                    head = " ".join(words[:_FIRST_FLUSH_WORDS])
                    cut = (len(head), len(head) + 1)
        if not cut:
            break
        piece = buffer[: cut[0]].strip()
        if piece:
            out.append(piece)
        buffer = buffer[cut[1]:]
        first = False
    return out, buffer


_TRAILING_FUNCTION_WORDS = {
    "and", "but", "so", "or", "because", "then", "the", "a", "an", "to", "of", "in", "on", "with",
    "my", "is", "um", "uh", "like", "that", "which", "if", "when", "for", "from", "at", "about", "i",
}


def looks_incomplete(text: str) -> bool:
    """Endpointing heuristic: did the speaker probably stop mid-thought?"""
    t = text.strip()
    if not t:
        return False
    if t.endswith((",", "-", "\u2014", "\u2026", "...")):
        return True
    if t[-1] in ".!?":
        return False
    last = re.sub(r"[^\w']", "", t.split()[-1]).lower()
    return last in _TRAILING_FUNCTION_WORDS


def clean_for_speech(text: str) -> str:
    clean = re.sub(r"```[\s\S]*?```", " ", text or "")
    clean = re.sub(r"https?://\S+", " ", clean)
    clean = re.sub(r"[*_`#>~|]", "", clean)
    clean = re.sub(r"^\s*[-\u2022]\s+", "", clean, flags=re.M)
    return re.sub(r"\s+", " ", clean).strip()


# --------------------------------------------------------------------------- upstream clients
_http: Optional[httpx.AsyncClient] = None


def get_http() -> httpx.AsyncClient:
    global _http
    if _http is None or _http.is_closed:
        _http = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=8.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=40, keepalive_expiry=120.0),
        )
    return _http


async def warmup() -> None:
    """Open TLS connections to Groq/Fish up-front so the first turn skips the handshake."""
    client = get_http()
    for url in ("https://api.groq.com/openai/v1/models", "https://api.fish.audio/"):
        try:
            await client.head(url, timeout=5.0)
        except Exception:
            pass


async def groq_stt(pcm: bytes) -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is missing")
    data = {"model": STT_MODEL, "response_format": "verbose_json", "temperature": "0"}
    if STT_LANG:
        data["language"] = STT_LANG
    resp = await get_http().post(
        STT_URL,
        headers={"Authorization": f"Bearer {key}"},
        data=data,
        files={"file": ("speech.wav", pcm16_to_wav(pcm), "audio/wav")},
        timeout=20.0,
    )
    if resp.is_error:
        raise RuntimeError(f"Groq STT failed ({resp.status_code}): {resp.text[:200]}")
    body = resp.json()
    text = (body.get("text") or "").strip()
    segments = body.get("segments") or []
    # Whisper invents "Thank you." on near-silence; its own no-speech score catches most of it.
    if segments and all(float(s.get("no_speech_prob", 0.0)) > 0.6 for s in segments):
        return ""
    if text.lower().strip(" .!?,") in {"you", "thank you", "thanks for watching", "bye"} and pcm_rms(pcm) < 0.01:
        return ""
    return text


def _fish_request(text: str, voice: str, fmt: str) -> Tuple[Dict[str, str], Dict[str, Any]]:
    key = os.getenv("FISH_API_KEY")
    if not key:
        raise RuntimeError("FISH_API_KEY is missing")
    payload: Dict[str, Any] = {
        # Small chunks reduce time-to-first-audio; the WebSocket forwards each
        # PCM chunk immediately instead of waiting for a whole sentence.
        "text": text, "format": fmt, "latency": FISH_LATENCY, "normalize": False, "chunk_length": 80,
    }
    if fmt == "pcm":
        payload["sample_rate"] = PCM_RATE
    else:
        payload["mp3_bitrate"] = 64
    ref = (voice or FISH_REFERENCE_ID).strip()
    if ref:
        payload["reference_id"] = ref
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "model": FISH_MODEL}
    return headers, payload


async def fish_tts_pcm(text: str, voice: str = "") -> AsyncIterator[bytes]:
    """Tier 1: stream raw PCM as Fish generates it. Chunks are always whole int16 samples."""
    headers, payload = _fish_request(text, voice, "pcm")
    async with get_http().stream("POST", FISH_TTS_URL, headers=headers, json=payload) as r:
        if r.is_error:
            detail = (await r.aread()).decode("utf-8", "ignore")[:200]
            raise RuntimeError(f"Fish TTS stream failed ({r.status_code}): {detail}")
        carry = b""
        async for chunk in r.aiter_bytes(1920):  # ~40 ms @ 24 kHz
            data = carry + chunk
            carry = data[-1:] if len(data) % 2 else b""
            if carry:
                data = data[:-1]
            if data:
                yield data


async def fish_tts_mp3(text: str, voice: str = "") -> Optional[bytes]:
    """Tier 2: the non-streaming mp3 call the app already used."""
    headers, payload = _fish_request(text, voice, "mp3")
    r = await get_http().post(FISH_TTS_URL, headers=headers, json=payload, timeout=20.0)
    if r.is_error:
        raise RuntimeError(f"Fish TTS mp3 failed ({r.status_code}): {r.text[:200]}")
    return r.content or None


# --------------------------------------------------------------------------- dependency bundle
@dataclass
class VoiceDeps:
    """Everything that touches app.py state or the network, injectable for tests."""
    get_session: Callable[[str], Dict[str, Any]]
    trim_memory: Callable[[list], list]
    llm_stream: Callable[[Dict[str, Any]], AsyncIterator[str]]
    mcp_text: Callable[[str, str, List[str], Dict[str, Any]], Awaitable[str]]
    human_message: Callable[[str], Any]
    ai_message: Callable[[str], Any]
    stt: Callable[[bytes], Awaitable[str]] = groq_stt
    tts_pcm: Callable[[str, str], AsyncIterator[bytes]] = fish_tts_pcm
    tts_mp3: Callable[[str, str], Awaitable[Optional[bytes]]] = fish_tts_mp3
    allowed_origin: str = ""
    hold_ms: int = HOLD_MS


# --------------------------------------------------------------------------- connection / turn state
@dataclass
class Turn:
    id: int
    committed: bool = False          # true once the first audio/text of the reply is on the wire
    user_msg: Any = None
    user_text: str = ""
    spoken: List[str] = field(default_factory=list)
    t0: float = field(default_factory=time.perf_counter)
    marks: Dict[str, float] = field(default_factory=dict)
    task: Optional[asyncio.Task] = None


class VoiceConnection:
    def __init__(self, ws: WebSocket, deps: VoiceDeps):
        self.ws, self.deps = ws, deps
        self.session_id = ""
        self.voice = ""
        self.mcp: List[str] = []
        self.pcm = bytearray()          # audio of the utterance being collected (kept across a "hold")
        self.utt_mark = 0
        self.in_speech = False
        self.turn: Optional[Turn] = None
        self.turn_counter = 0
        self.tts_sem = asyncio.Semaphore(TTS_CONCURRENCY)
        self._send_lock = asyncio.Lock()
        self.closed = False

    # ---- outbound
    async def send_json(self, **msg: Any) -> None:
        if self.closed:
            return
        async with self._send_lock:
            try:
                await self.ws.send_text(json.dumps(msg))
            except Exception:
                self.closed = True

    async def send_audio(self, kind: int, turn_id: int, payload: bytes) -> None:
        if self.closed:
            return
        async with self._send_lock:
            try:
                await self.ws.send_bytes(struct.pack(">BI", kind, turn_id) + payload)
            except Exception:
                self.closed = True

    # ---- main loop
    async def run(self) -> None:
        await self.send_json(type="ready", pcm_rate=PCM_RATE)
        asyncio.create_task(warmup())
        try:
            while True:
                msg = await self.ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    self.on_audio(msg["bytes"])
                elif msg.get("text") is not None:
                    try:
                        data = json.loads(msg["text"])
                    except ValueError:
                        continue
                    await self.on_message(data)
        finally:
            self.closed = True
            await self.cancel_turn("disconnect", notify=False)

    def on_audio(self, chunk: bytes) -> None:
        """Append bounded, sample-aligned PCM received while speaking."""
        if not self.in_speech or not chunk:
            return
        remaining = MAX_UTTERANCE_BYTES - len(self.pcm)
        if remaining <= 0:
            return
        # A client frame can be larger than the remaining budget. Keep the cap
        # strict and avoid retaining a dangling byte for int16 decoding.
        data = chunk[:remaining]
        if len(data) % 2:
            data = data[:-1]
        self.pcm.extend(data)

    async def on_message(self, m: Dict[str, Any]) -> None:
        t = m.get("type")
        if t == "ping":
            await self.send_json(type="pong", t=m.get("t"))
        elif t == "config":
            self.session_id = str(m.get("session_id") or self.session_id)
            self.voice = str(m.get("voice") or "")
            self.mcp = [str(x) for x in (m.get("mcp_servers") or [])]
        elif t == "speech_start":
            await self.on_speech_start()
        elif t == "speech_end":
            await self.on_speech_end()
        elif t == "speech_abort":
            self.in_speech = False
            del self.pcm[self.utt_mark:]
        elif t == "interrupt":
            await self.cancel_turn("interrupt")
        elif t == "text":
            text = str(m.get("text") or "").strip()
            if text:
                await self.cancel_turn("new_text")
                self.pcm.clear()
                self.start_turn(text=text)

    # ---- utterance lifecycle
    async def on_speech_start(self) -> None:
        self.in_speech = True
        if self.turn and not self.turn.task.done():
            if self.turn.committed:
                # assistant is already talking: this is a barge-in, the old audio is dead
                await self.cancel_turn("barge_in")
                self.pcm.clear()
            else:
                # still transcribing / holding / thinking: the user is *continuing* their sentence.
                # Cancel the pending turn but keep its audio so the whole thing is re-transcribed together.
                await self.cancel_turn("merge", notify=False)
        self.utt_mark = len(self.pcm)

    async def on_speech_end(self) -> None:
        self.in_speech = False
        if len(self.pcm) < MIN_UTTERANCE_BYTES:
            self.pcm.clear()
            await self.send_json(type="state", state="listening", turn=self.turn_counter)
            return
        self.start_turn(pcm=bytes(self.pcm))

    def start_turn(self, pcm: Optional[bytes] = None, text: Optional[str] = None) -> None:
        self.turn_counter += 1
        turn = Turn(id=self.turn_counter)
        self.turn = turn
        turn.task = asyncio.create_task(self._run_turn(turn, pcm, text))

    async def cancel_turn(self, reason: str, notify: bool = True) -> None:
        turn = self.turn
        if not turn or not turn.task or turn.task.done():
            return
        turn.task.cancel()
        try:
            await turn.task
        except (asyncio.CancelledError, Exception):
            pass
        # keep the conversation history consistent with what actually happened
        if self.session_id:
            msgs = self.deps.get_session(self.session_id)["messages"]
            if not turn.committed:
                self._drop_message(msgs, turn.user_msg)       # merge: the user text will be re-created
            elif turn.spoken:
                msgs.append(self.deps.ai_message(" ".join(turn.spoken)))  # what the user actually heard
        if notify:
            await self.send_json(type="cancelled", turn=turn.id, reason=reason)

    @staticmethod
    def _drop_message(msgs: list, target: Any) -> None:
        if target is None:
            return
        for i, m in enumerate(msgs):
            if m is target:
                del msgs[i]
                return

    async def _commit(self, turn: Turn) -> None:
        if turn.committed:
            return
        turn.committed = True
        turn.marks["first_audio"] = time.perf_counter()
        self.pcm.clear()                      # this utterance has been consumed
        await self.send_json(type="transcript", text=turn.user_text, turn=turn.id)
        await self.send_json(type="state", state="speaking", turn=turn.id)

    # ---- one conversational turn
    async def _run_turn(self, turn: Turn, pcm: Optional[bytes], text: Optional[str]) -> None:
        try:
            if not self.session_id:
                await self.send_json(type="error", message="No session_id configured.")
                return
            if text is None:
                await self.send_json(type="state", state="transcribing", turn=turn.id)
                ts = time.perf_counter()
                try:
                    text = await self.deps.stt(pcm or b"")
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    print(f"[voice:{self.session_id}] STT failed: {exc}")
                    await self.send_json(type="error", message="I couldn't hear that. Please try again.")
                    self.pcm.clear()
                    await self.send_json(type="state", state="listening", turn=turn.id)
                    return
                turn.marks["stt_ms"] = (time.perf_counter() - ts) * 1000
                if not text:
                    self.pcm.clear()
                    await self.send_json(type="state", state="listening", turn=turn.id)
                    return
                if looks_incomplete(text):
                    # Sounded unfinished: give the user a moment to continue. If they do, speech_start
                    # cancels this task and the audio is merged with what they say next.
                    await self.send_json(type="state", state="waiting", turn=turn.id)
                    await asyncio.sleep(self.deps.hold_ms / 1000)
            turn.user_text = text
            await self.send_json(type="state", state="thinking", turn=turn.id)

            session = self.deps.get_session(self.session_id)
            session["messages"] = self.deps.trim_memory(session["messages"])
            turn.user_msg = self.deps.human_message(text)
            session["messages"].append(turn.user_msg)

            full = await self._speak_reply(turn, session, text)
            await self._commit(turn)          # covers replies that produced no audio at all
            session["messages"].append(self.deps.ai_message(full.strip()))
            await self.send_json(type="turn_done", turn=turn.id, response=full.strip())
            await self.send_json(type="metrics", turn=turn.id, **{
                k: round(v, 1) for k, v in self._metrics(turn).items()})
            await self.send_json(type="state", state="listening", turn=turn.id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"[voice:{self.session_id}] turn failed: {exc}")
            await self.send_json(type="error", message="Something went wrong. Please try again.")
            await self.send_json(type="state", state="listening", turn=turn.id)

    @staticmethod
    def _metrics(turn: Turn) -> Dict[str, float]:
        m = {}
        if "stt_ms" in turn.marks:
            m["stt_ms"] = turn.marks["stt_ms"]
        if "llm_first" in turn.marks:
            m["llm_first_token_ms"] = (turn.marks["llm_first"] - turn.marks.get("llm_start", turn.t0)) * 1000
        if "first_audio" in turn.marks:
            m["turn_to_first_audio_ms"] = (turn.marks["first_audio"] - turn.t0) * 1000
        return m

    async def _speak_reply(self, turn: Turn, session: Dict[str, Any], user_text: str) -> str:
        order: asyncio.Queue = asyncio.Queue()   # (piece, chunk_queue) in speaking order, then None
        synth_tasks: List[asyncio.Task] = []
        state = {"full": ""}

        def start_piece(piece: str) -> None:
            q: asyncio.Queue = asyncio.Queue()
            synth_tasks.append(asyncio.create_task(self._synth(piece, q)))
            order.put_nowait((piece, q))

        async def producer() -> None:
            turn.marks["llm_start"] = time.perf_counter()
            try:
                if self.mcp:   # slow path: tools are involved, so speak the finished answer
                    state["full"] = await self.deps.mcp_text(self.session_id, user_text, self.mcp, session)
                    turn.marks["llm_first"] = time.perf_counter()
                    pieces, rest = pop_speakable(state["full"] + " ", False)
                    for p in pieces + ([rest.strip()] if rest.strip() else []):
                        start_piece(p)
                else:
                    buf, first = "", True
                    async for delta in self.deps.llm_stream(session):
                        turn.marks.setdefault("llm_first", time.perf_counter())
                        state["full"] += delta
                        buf += delta
                        pieces, buf = pop_speakable(buf, first)
                        for p in pieces:
                            first = False
                            start_piece(p)
                    if buf.strip():
                        start_piece(buf.strip())
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[voice:{self.session_id}] LLM failed: {exc}")
                if not state["full"]:
                    state["full"] = "Sorry, something went wrong. Please try again."
                    start_piece(state["full"])
            finally:
                order.put_nowait(None)

        async def sender() -> None:
            while True:
                item = await order.get()
                if item is None:
                    return
                piece, q = item
                while True:
                    kind, payload = await q.get()
                    if kind == "end":
                        break
                    await self._commit(turn)
                    if kind == "pcm":
                        await self.send_audio(KIND_PCM, turn.id, payload)
                    elif kind == "mp3":
                        await self.send_audio(KIND_MP3, turn.id, payload)
                    elif kind == "text":
                        await self.send_json(type="tts_fallback", text=payload, turn=turn.id)
                turn.spoken.append(piece)
                await self.send_json(type="assistant_text", text=piece, turn=turn.id)

        prod = asyncio.create_task(producer())
        try:
            await sender()
            await prod
        finally:
            for t in [prod, *synth_tasks]:
                if not t.done():
                    t.cancel()
            await asyncio.gather(prod, *synth_tasks, return_exceptions=True)
        return state["full"]

    async def _synth(self, text: str, q: asyncio.Queue) -> None:
        """Speak one piece. Tier 1: streamed PCM. Tier 2: whole mp3. Tier 3: tell the client to use the
        browser's own speech synthesis. A failure half-way through a stream keeps the audio already sent."""
        spoken = clean_for_speech(text)
        sent = 0
        try:
            if not spoken:
                return
            async with self.tts_sem:
                try:
                    async for chunk in self.deps.tts_pcm(spoken, self.voice):
                        sent += len(chunk)
                        q.put_nowait(("pcm", chunk))
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    print(f"[voice:{self.session_id}] TTS stream failed: {exc}")
                if sent:
                    return
                try:
                    mp3 = await self.deps.tts_mp3(spoken, self.voice)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    print(f"[voice:{self.session_id}] TTS mp3 fallback failed: {exc}")
                    mp3 = None
                if mp3:
                    q.put_nowait(("mp3", mp3))
                else:
                    q.put_nowait(("text", spoken))
        finally:
            q.put_nowait(("end", None))


# --------------------------------------------------------------------------- router
def create_router(deps: VoiceDeps) -> APIRouter:
    router = APIRouter()

    @router.websocket("/ws/voice")
    async def ws_voice(ws: WebSocket) -> None:
        origin = (ws.headers.get("origin") or "").rstrip("/")
        if deps.allowed_origin and origin and origin != deps.allowed_origin:
            await ws.close(code=1008)
            return
        await ws.accept()
        await VoiceConnection(ws, deps).run()

    return router
