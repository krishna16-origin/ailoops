import pathlib
import unittest

import app


class ModelRoutingTests(unittest.TestCase):
    def test_chat_deepseek_uses_current_nvidia_id(self):
        llm = app.get_llm("fast", 0.2, 128)
        self.assertEqual(llm.model, "deepseek-ai/deepseek-v4-pro-0813")

    def test_chat_kimi_uses_current_nvidia_id(self):
        llm = app.get_llm("balanced", 0.2, 128)
        self.assertEqual(llm.model, "moonshotai/kimi-k3")

    def test_code_deepseek_uses_current_nvidia_id(self):
        llm = app.get_code_llm("step-flash", 0.2, 128)
        self.assertEqual(llm.model, "deepseek-ai/deepseek-v4-pro-0813")

    def test_fast_defaults_are_low_and_short(self):
        self.assertEqual(app.DEFAULT_THINKING_LEVEL, "low")
        self.assertEqual(app.THINKING_LEVELS["low"]["max_tokens"], 4000)
        frontend = pathlib.Path(__file__).with_name("frontend") / "index.html"
        html = frontend.read_text(encoding="utf-8")
        self.assertIn('<select id="tempSetting">\n                <option value="low" selected>', html)
        self.assertIn('<select id="codeReasoningLevel">\n                <option value="low" selected>', html)

    def test_deepseek_and_kimi_use_long_stream_inactivity_timeout(self):
        for llm in (
            app.get_llm("balanced", 0.2, 128),
            app.get_llm("fast", 0.2, 128),
            app.get_code_llm("medium", 0.2, 128),
            app.get_code_llm("step-flash", 0.2, 128),
        ):
            self.assertEqual(llm._client.timeout, app.LONG_GENERATION_TRANSPORT_TIMEOUT)

    def test_non_reasoning_models_keep_transport_timeout(self):
        self.assertEqual(app.get_llm("reasoning", 0.2, 128)._client.timeout, 300)
        self.assertEqual(app.get_code_llm("gemma", 0.2, 128)._client.timeout, 300)

    def test_frontend_has_no_stale_deepseek_id(self):
        frontend = pathlib.Path(__file__).with_name("frontend") / "index.html"
        self.assertNotIn("deepseek-ai/deepseek-v4-pro'", frontend.read_text(encoding="utf-8"))
        self.assertIn("deepseek-ai/deepseek-v4-pro-0813", frontend.read_text(encoding="utf-8"))

    def test_deepseek_rejects_low_effort_maps_to_none(self):
        self.assertEqual(
            app._map_reasoning_effort("low", "deepseek-ai/deepseek-v4-pro-0813"),
            "none",
        )
        self.assertEqual(
            app._map_reasoning_effort("medium", "deepseek-ai/deepseek-v4-pro-0813"),
            "high",
        )
        self.assertEqual(
            app._map_reasoning_effort("max", "deepseek-ai/deepseek-v4-pro-0813"),
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

    def test_deepseek_max_tokens_clamped_to_16384(self):
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

    def test_kimi_and_deepseek_force_temperature_1(self):
        for llm in (
            app.get_llm("balanced", 0.2, 128),
            app.get_llm("fast", 0.7, 128),
            app.get_code_llm("glimmer", 0.3, 128),
            app.get_code_llm("step-flash", 0.5, 128),
        ):
            self.assertEqual(llm.temperature, 1.0)


if __name__ == "__main__":
    unittest.main()
