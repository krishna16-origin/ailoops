import asyncio
import json
import unittest

import voice_ws


class FakeWebSocket:
    def __init__(self):
        self.json = []
        self.audio = []

    async def send_text(self, value):
        self.json.append(value)

    async def send_bytes(self, value):
        self.audio.append(value)


async def empty_stream(_session):
    if False:
        yield ""


async def one_sentence_stream(_session):
    yield "The answer is ready."


async def hanging_tts(_text, _voice):
    await asyncio.sleep(1)
    if False:
        yield b""


async def hanging_mp3(_text, _voice):
    await asyncio.sleep(1)
    return None


class VoiceHelpersTests(unittest.TestCase):
    def test_first_piece_flushes_after_three_words(self):
        pieces, rest = voice_ws.pop_speakable(
            "Tell me about the weather today and", first=True
        )
        self.assertEqual(pieces, ["Tell me about"])
        self.assertEqual(rest, "the weather today and")

    def test_sentence_chunker_does_not_split_abbreviation(self):
        pieces, rest = voice_ws.pop_speakable("Dr. Smith is here. Next step.", first=True)
        self.assertEqual(pieces, ["Dr. Smith is here.", "Next step."])
        self.assertEqual(rest, "")

    def test_pcm_input_is_bounded_and_sample_aligned(self):
        ws = FakeWebSocket()
        deps = voice_ws.VoiceDeps(
            get_session=lambda _: {"messages": []},
            trim_memory=lambda messages, **kwargs: messages,
            llm_stream=empty_stream,
            human_message=lambda text: text,
            ai_message=lambda text: text,
        )
        connection = voice_ws.VoiceConnection(ws, deps)
        connection.in_speech = True
        connection.on_audio(b"x" * (voice_ws.MAX_UTTERANCE_BYTES + 1))
        self.assertEqual(len(connection.pcm), voice_ws.MAX_UTTERANCE_BYTES)
        self.assertEqual(len(connection.pcm) % 2, 0)
        connection.on_audio(b"y" * 100)
        self.assertEqual(len(connection.pcm), voice_ws.MAX_UTTERANCE_BYTES)

    def test_speech_abort_discards_only_current_utterance(self):
        ws = FakeWebSocket()
        deps = voice_ws.VoiceDeps(
            get_session=lambda _: {"messages": []},
            trim_memory=lambda messages, **kwargs: messages,
            llm_stream=empty_stream,
            human_message=lambda text: text,
            ai_message=lambda text: text,
        )
        connection = voice_ws.VoiceConnection(ws, deps)
        connection.pcm.extend(b"a" * 20)
        connection.utt_mark = len(connection.pcm)
        connection.pcm.extend(b"b" * 40)
        asyncio.run(connection.on_message({"type": "speech_abort"}))
        self.assertEqual(connection.pcm, b"a" * 20)
        self.assertFalse(connection.in_speech)

    def test_text_is_sent_before_a_hanging_tts_provider_finishes(self):
        ws = FakeWebSocket()
        deps = voice_ws.VoiceDeps(
            get_session=lambda _: {"messages": []},
            trim_memory=lambda messages, **kwargs: messages,
            llm_stream=one_sentence_stream,
            human_message=lambda text: text,
            ai_message=lambda text: text,
            tts_pcm=hanging_tts,
            tts_mp3=hanging_mp3,
        )
        old_stream_timeout = voice_ws.TTS_STREAM_TIMEOUT
        old_fallback_timeout = voice_ws.TTS_FALLBACK_TIMEOUT
        voice_ws.TTS_STREAM_TIMEOUT = 0.01
        voice_ws.TTS_FALLBACK_TIMEOUT = 0.01
        try:
            connection = voice_ws.VoiceConnection(ws, deps)
            connection.session_id = "s"
            asyncio.run(connection._speak_reply(voice_ws.Turn(id=1), {"messages": []}))
        finally:
            voice_ws.TTS_STREAM_TIMEOUT = old_stream_timeout
            voice_ws.TTS_FALLBACK_TIMEOUT = old_fallback_timeout
        events = [json.loads(item) for item in ws.json]
        text_index = next(i for i, event in enumerate(events) if event["type"] == "assistant_text")
        self.assertEqual(events[text_index]["text"], "The answer is ready.")
        self.assertTrue(any(event["type"] == "error" for event in events))


if __name__ == "__main__":
    unittest.main()
