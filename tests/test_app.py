import unittest

from app import chunk_text, stable_prefix


class AppHelpersTests(unittest.TestCase):
    def test_chunk_text_preserves_punctuation_and_word_cap(self):
        self.assertEqual(
            chunk_text("Hello there. This is a longer sentence with more words", max_words=4),
            ["Hello there.", "This is a longer", "sentence with more words"],
        )


    def test_stable_prefix_emits_only_new_complete_words(self):
        self.assertEqual(stable_prefix("hello wor", "hello world today"), "hello")
        self.assertEqual(stable_prefix("hello world", "hello world today"), "hello world")


if __name__ == "__main__":
    unittest.main()
