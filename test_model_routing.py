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

    def test_model_factories_use_long_stream_timeout(self):
        for llm in (
            app.get_llm("balanced", 0.2, 128),
            app.get_llm("fast", 0.2, 128),
            app.get_code_llm("medium", 0.2, 128),
            app.get_code_llm("step-flash", 0.2, 128),
        ):
            self.assertEqual(llm._client.timeout, 300)

    def test_frontend_has_no_stale_deepseek_id(self):
        frontend = pathlib.Path(__file__).with_name("frontend") / "index.html"
        self.assertNotIn("deepseek-ai/deepseek-v4-pro'", frontend.read_text(encoding="utf-8"))
        self.assertIn("deepseek-ai/deepseek-v4-pro-0813", frontend.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
