import pathlib
import unittest

import app
import voice_ws


class ModelRoutingTests(unittest.TestCase):
    def test_chat_glm_uses_current_nvidia_id(self):
        llm = app.get_llm("fast", 0.2, 128)
        self.assertEqual(llm.model, "z-ai/glm-5.3")

    def test_chat_kimi_uses_current_nvidia_id(self):
        llm = app.get_llm("balanced", 0.2, 128)
        self.assertEqual(llm.model, "moonshotai/kimi-k3")

    def test_code_glm_uses_current_nvidia_id(self):
        llm = app.get_code_llm("step-flash", 0.2, 128)
        self.assertEqual(llm.model, "z-ai/glm-5.3")

    def test_fast_defaults_are_low_and_short(self):
        self.assertEqual(app.DEFAULT_THINKING_LEVEL, "low")
        self.assertEqual(app.THINKING_LEVELS["low"]["max_tokens"], 4000)
        frontend = pathlib.Path(__file__).with_name("frontend") / "index.html"
        html = frontend.read_text(encoding="utf-8")
        self.assertIn('<select id="tempSetting">\n                <option value="low" selected>', html)
        self.assertIn('<select id="codeReasoningLevel">\n                <option value="low" selected>', html)

    def test_chat_models_use_unbounded_transport_timeout(self):
        # Chat responses should not be cut off by an arbitrary 300-second
        # ceiling. Normal mode remains fast through its direct-answer prompt.
        for llm in (
            app.get_llm("balanced", 0.2, 128),
            app.get_llm("fast", 0.2, 128),
            app.get_llm("reasoning", 0.2, 128),
        ):
            self.assertEqual(llm._client.timeout, app.LONG_GENERATION_TRANSPORT_TIMEOUT)

    def test_code_models_keep_standard_transport_timeout(self):
        self.assertEqual(app.get_code_llm("gemma", 0.2, 128)._client.timeout, 300)

    def test_deep_think_toggle_controls_reasoning_prompt(self):
        normal = app.build_messages([], "low", deep_think=False)[0].content
        deep = app.build_messages([], "low", deep_think=True)[0].content
        self.assertIn("Answer directly and clearly", normal)
        self.assertNotIn("Before answering, think inside", normal)
        self.assertIn("Before answering, think inside", deep)

    def test_frontend_has_no_stale_deepseek_id(self):
        frontend = pathlib.Path(__file__).with_name("frontend") / "index.html"
        html = frontend.read_text(encoding="utf-8")
        self.assertNotIn("deepseek", html.lower())
        self.assertIn("z-ai/glm-5.3", html)

    def test_code_workflow_modes_are_explicit_and_build_is_default(self):
        self.assertEqual(app.normalize_code_workflow_mode("plan"), "plan")
        self.assertEqual(app.normalize_code_workflow_mode("build"), "build")
        self.assertEqual(app.normalize_code_workflow_mode("unknown"), "build")
        self.assertEqual(app.CodeChatRequest(message="x", session_id="s").mode, "build")
        frontend = pathlib.Path(__file__).with_name("frontend") / "index.html"
        html = frontend.read_text(encoding="utf-8")
        self.assertIn('data-workflow-mode="plan"', html)
        self.assertIn('data-workflow-mode="build"', html)
        self.assertIn("mode: document.getElementById('codeWorkflowMode').value", html)

    def test_voice_chat_uses_separate_fish_audio_request(self):
        frontend = pathlib.Path(__file__).with_name("frontend") / "index.html"
        html = frontend.read_text(encoding="utf-8")
        self.assertFalse(hasattr(app.ChatRequest(message="x", session_id="s"), "voice_mode"))
        voice_request = app.VoiceChatRequest(message="x", session_id="s")
        self.assertEqual(voice_request.voice, "")
        self.assertTrue(app.FISH_MODEL)
        self.assertNotIn("speakVoiceTextFallback", html)
        self.assertNotIn("using browser speech", html.lower())

    def test_voice_uses_groq_gpt_oss_20b_by_default(self):
        self.assertEqual(app.VOICE_LLM_MODEL, "openai/gpt-oss-20b")

    def test_voice_uses_one_fixed_sarah_reference_across_transports(self):
        expected = "933563129e564b19a115bedd57b7406a"
        self.assertEqual(app.FISH_REFERENCE_ID, expected)
        self.assertEqual(voice_ws.FISH_REFERENCE_ID, expected)

    def test_reopening_voice_mode_resets_paused_session_state(self):
        source = pathlib.Path(__file__).with_name("frontend") / "index.html"
        html = source.read_text(encoding="utf-8")
        reopen = html[html.index("function openMobileVoiceMode") : html.index("function setVoiceLevel")]
        self.assertIn("voiceRecognitionPaused = false", reopen)
        self.assertIn("resetVoiceTurnState()", reopen)
        self.assertIn(".mobile-voice-agent { position:absolute; left:0;", html)
        self.assertIn("font-size:14px; pointer-events:none;", html)
        activate = html[html.index("async function toggleVoiceMode") :]
        self.assertIn("closeVoiceSocket();", activate)

    def test_voice_chat_streams_groq_text_through_fish_audio(self):
        source = pathlib.Path(__file__).with_name("app.py").read_text(encoding="utf-8")
        voice_source = source[source.index("async def voice_chat"):]
        voice_source = voice_source[:voice_source.index("\n\n@app.post(\"/chat\")")]
        self.assertIn("async for delta in _groq_voice_stream(session)", voice_source)
        self.assertIn("_fish_tts(piece, request.voice)", voice_source)
        self.assertIn('media_type="application/x-ndjson"', voice_source)
        self.assertNotIn("voice_mode=True", voice_source)
        self.assertNotIn("_groq_orpheus_audio", voice_source)

    def test_voice_router_wiring_has_no_legacy_mcp_text_dependency(self):
        # Importing app constructs this dependency bundle at module load time;
        # this assertion documents the startup contract that previously broke
        # the Render deploy when app.py and voice_ws.py were out of sync.
        self.assertNotIn("mcp_text", app.VoiceDeps.__dataclass_fields__)
        self.assertIn("llm_stream", app.VoiceDeps.__dataclass_fields__)

    def test_chat_history_supports_titles_and_three_dot_actions(self):
        frontend = pathlib.Path(__file__).with_name("frontend") / "index.html"
        html = frontend.read_text(encoding="utf-8")
        for marker in (
            "history-item-more",
            "toggleLocalHistoryPin",
            "renameLocalHistory",
            "deleteLocalHistory",
            "customTitle",
            "record.pinned",
        ):
            self.assertIn(marker, html)

    def test_glm_low_and_medium_use_direct_answer_effort(self):
        self.assertEqual(
            app._map_reasoning_effort("low", "z-ai/glm-5.3"),
            "none",
        )
        self.assertEqual(
            app._map_reasoning_effort("medium", "z-ai/glm-5.3"),
            "none",
        )
        self.assertEqual(
            app._map_reasoning_effort("high", "z-ai/glm-5.3"),
            "high",
        )
        self.assertEqual(
            app._map_reasoning_effort("max", "z-ai/glm-5.3"),
            "max",
        )

    def test_kimi_low_and_medium_use_low_effort(self):
        self.assertEqual(
            app._map_reasoning_effort("low", "moonshotai/kimi-k3"),
            "low",
        )
        self.assertEqual(
            app._map_reasoning_effort("medium", "moonshotai/kimi-k3"),
            "low",
        )
        self.assertEqual(
            app._map_reasoning_effort("high", "moonshotai/kimi-k3"),
            "high",
        )
        self.assertEqual(
            app._map_reasoning_effort("max", "moonshotai/kimi-k3"),
            "max",
        )

    def test_glm_max_tokens_clamped_to_16384(self):
        llm = app.get_llm("fast", 0.2, 40000)
        self.assertEqual(llm.max_tokens, 16384)
        llm2 = app.get_code_llm("step-flash", 0.2, 32000)
        self.assertEqual(llm2.max_tokens, 16384)

    def test_kimi_max_tokens_clamped_to_65536(self):
        llm = app.get_llm("balanced", 0.2, 80000)
        self.assertEqual(llm.max_tokens, 65536)
        llm2 = app.get_code_llm("glimmer", 0.2, 40000)
        self.assertEqual(llm2.max_tokens, 40000)  # under cap

    def test_kimi_budget_meets_reasoning_endpoint_minimum(self):
        self.assertEqual(app.get_llm("balanced", 0.2, 4000).max_tokens, 8000)
        self.assertEqual(app.get_code_llm("glimmer", 0.2, 4000).max_tokens, 8000)

    def test_kimi_effort_budgets_are_capped_below_32000(self):
        expected = {"low": 8000, "medium": 12000, "high": 16000, "extra": 24000, "max": 32000}
        for level, budget in expected.items():
            self.assertEqual(
                app._model_thinking_budget("moonshotai/kimi-k3", level, 40000),
                budget,
            )

    def test_kimi_and_glm_force_temperature_1_but_chat_timeout_is_unbounded(self):
        # Correctness (temperature pin) and transport behavior are separate
        # concerns: both models keep the 1.0 temperature their NIM endpoint
        # needs for coherent output, while Chat mode has no finite ceiling.
        for llm in (
            app.get_llm("balanced", 0.2, 128),
            app.get_llm("fast", 0.7, 128),
            app.get_code_llm("glimmer", 0.3, 128),
            app.get_code_llm("step-flash", 0.5, 128),
        ):
            self.assertEqual(llm.temperature, 1.0)
            if llm.model in (app.KIMI_MODEL, app.GLM_MODEL):
                self.assertEqual(llm._client.timeout, app.LONG_GENERATION_TRANSPORT_TIMEOUT)
            else:
                self.assertEqual(llm._client.timeout, 300)


if __name__ == "__main__":
    unittest.main()
