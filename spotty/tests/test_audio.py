# tests/test_module1.py
import unittest
from spotty.audio import WakeWordConversationAgent

class TestFunction1(unittest.TestCase):
    def test_function1(self):
        self.assertEqual(WakeWordConversationAgent().get_wake_word(), "Hey Spot!")

if __name__ == "__main__":
    unittest.main()
