"""
Tests for education platform
"""

import unittest
from zipline.education.courses.platform import CoursePlatform, CourseStatus
from zipline.education.library.glossary import TradingGlossary


class TestCoursePlatform(unittest.TestCase):
    """Test course platform functionality"""
    
    def setUp(self):
        """Set up test course platform"""
        self.platform = CoursePlatform()
        self.user_id = "test_user_123"
    
    def test_get_catalog(self):
        """Test getting course catalog"""
        catalog = self.platform.get_catalog()
        self.assertGreater(len(catalog), 0)
        
        # Check first course structure
        course = catalog[0]
        self.assertIn('title', course)
        self.assertIn('level', course)
        self.assertIn('modules', course)
    
    def test_enroll_user(self):
        """Test user enrollment"""
        success = self.platform.enroll_user(
            self.user_id,
            'intro_to_trading'
        )
        self.assertTrue(success)


class TestTradingGlossary(unittest.TestCase):
    """Test trading glossary"""
    
    def setUp(self):
        """Set up test glossary"""
        self.glossary = TradingGlossary()
    
    def test_get_term(self):
        """Test getting a term"""
        term = self.glossary.get_term('algorithmic_trading')
        self.assertIsNotNone(term)
        self.assertEqual(term['term'], 'Algorithmic Trading')
        self.assertIn('definition', term)


if __name__ == '__main__':
    unittest.main()
