"""Tests for competition module"""
import unittest


class TestCompetitionPlatform(unittest.TestCase):
    """Test CompetitionPlatform functionality."""
    
    def test_import(self):
        """Test that modules can be imported."""
        from zipline.competition import CompetitionPlatform
        platform = CompetitionPlatform("Test Platform")
        self.assertEqual(platform.name, "Test Platform")


if __name__ == '__main__':
    unittest.main()
