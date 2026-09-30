import asyncio
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
            trim_memory=lambda messages: messages,
            llm_stream=empty_stream,
            mcp_text=lambda *args: None,
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
            trim_memory=lambda messages: messages,
            llm_stream=empty_stream,
            mcp_text=lambda *args: None,
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


if __name__ == "__main__":
    unittest.main()
