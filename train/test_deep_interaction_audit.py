import unittest

from deep_interaction_audit import score


class DeepInteractionAuditTest(unittest.TestCase):
    def test_extracts_exact_final_values(self):
        self.assertTrue(score({"kind": "number", "answer": "63"}, "state=19; 63"))
        self.assertTrue(score({"kind": "yesno", "answer": "no"}, "The answer is no."))
        self.assertTrue(score({"kind": "list", "answer": "[2,4,7,11]"}, "[2, 4, 7, 11]"))
        self.assertTrue(score({"kind": "string", "answer": "moPQsaic"}, "moPQsaic"))

    def test_requires_code_contract_fragments(self):
        case = {"kind": "code", "answer": ["def is_even", "% 2", "== 0"]}
        self.assertTrue(score(case, "def is_even(n):\n    return n % 2 == 0"))
        self.assertFalse(score(case, "def is_even(n):\n    return True"))
        self.assertFalse(score(case, "def is_even(n):\nreturn n % 2 == 0"))


if __name__ == "__main__":
    unittest.main()
