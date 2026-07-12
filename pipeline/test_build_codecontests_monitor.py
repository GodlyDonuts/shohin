import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_codecontests_monitor import normalized_question, render_text


class CodeContestsMonitorTest(unittest.TestCase):
    def test_normalized_question_is_punctuation_insensitive(self):
        self.assertEqual(normalized_question("Add [1, 2]!"), normalized_question("add 1 2"))

    def test_render_text_has_prompt_and_python_fence(self):
        rendered = render_text("Solve it", "print(1)")
        self.assertEqual(rendered, "Solve it\n\n```python\nprint(1)\n```")


if __name__ == "__main__":
    unittest.main()
