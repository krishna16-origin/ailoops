import pathlib
import unittest

import app


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

    def test_kimi_and_glm_now_use_standard_transport_timeout(self):
        # Kimi and GLM 5.3 respond on the same fast, standard-timeout path as
        # Nemotron now — no model is pinned to the long-running 24h transport
        # window anymore.
        for llm in (
            app.get_llm("balanced", 0.2, 128),
            app.get_llm("fast", 0.2, 128),
            app.get_code_llm("medium", 0.2, 128),
            app.get_code_llm("step-flash", 0.2, 128),
        ):
            self.assertEqual(llm._client.timeout, 300)

    def test_non_reasoning_models_keep_transport_timeout(self):
        self.assertEqual(app.get_llm("reasoning", 0.2, 128)._client.timeout, 300)
        self.assertEqual(app.get_code_llm("gemma", 0.2, 128)._client.timeout, 300)

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

    def test_glm_rejects_low_effort_maps_to_none(self):
        self.assertEqual(
            app._map_reasoning_effort("low", "z-ai/glm-5.3"),
            "none",
        )
        self.assertEqual(
            app._map_reasoning_effort("medium", "z-ai/glm-5.3"),
            "high",
        )
        self.assertEqual(
            app._map_reasoning_effort("max", "z-ai/glm-5.3"),
            "max",
        )

    def test_kimi_effort_keeps_low(self):
        self.assertEqual(
            app._map_reasoning_effort("low", "moonshotai/kimi-k3"),
            "low",
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

    def test_kimi_and_glm_force_temperature_1_but_use_fast_timeout(self):
        # Correctness (temperature pin) and speed (timeout) are separate
        # concerns: both models keep the 1.0 temperature their NIM endpoint
        # needs for coherent output, but neither is pinned to the old 24h
        # long-running timeout anymore — that part now matches Nemotron.
        for llm in (
            app.get_llm("balanced", 0.2, 128),
            app.get_llm("fast", 0.7, 128),
            app.get_code_llm("glimmer", 0.3, 128),
            app.get_code_llm("step-flash", 0.5, 128),
        ):
            self.assertEqual(llm.temperature, 1.0)
            self.assertEqual(llm._client.timeout, 300)


if __name__ == "__main__":
    unittest.main()
