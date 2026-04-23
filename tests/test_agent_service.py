import unittest
from pathlib import Path

from src.agent.service import BCIAgentService


class TestBCIAgentService(unittest.TestCase):
    def setUp(self):
        self.service = BCIAgentService(project_root=Path(__file__).resolve().parents[1])

    def test_set_algorithm_validation(self):
        result = self.service.set_algorithm("non_existing_algo")
        self.assertEqual(result["status"], "error")

    def test_set_preprocess(self):
        result = self.service.set_preprocess(8, 30)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["current"]["low"], 8)
        self.assertEqual(result["current"]["high"], 30)

    def test_generate_report_without_run(self):
        result = self.service.generate_report("markdown")
        self.assertEqual(result["status"], "error")


if __name__ == "__main__":
    unittest.main()
