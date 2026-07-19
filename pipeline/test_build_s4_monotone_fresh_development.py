#!/usr/bin/env python3

import unittest

from build_s4_monotone_fresh_development import main


class MonotoneFreshBuilderImportTest(unittest.TestCase):
    def test_builder_entrypoint_is_importable(self):
        self.assertTrue(callable(main))


if __name__ == "__main__":
    unittest.main()
