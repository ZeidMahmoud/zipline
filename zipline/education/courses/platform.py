"""
Course Platform

Learning Management System for Zipline education.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class CourseStatus(Enum):
    """Course enrollment status"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PAUSED = "paused"


class CoursePlatform:
    """
    Course Platform - Learning Management System
    
    Features:
    - Course catalog
    - Progress tracking
    - Certificates
    - Instructor tools
    
    Example:
        >>> platform = CoursePlatform()
        >>> course = platform.get_course("intro_to_trading")
        >>> platform.enroll_user("user123", "intro_to_trading")
        >>> progress = platform.get_progress("user123", "intro_to_trading")
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize course platform
        
        Args:
            data_path: Path to store course data
        """
        self.data_path = data_path or "./education_data"
        self._courses: Dict[str, Dict] = {}
        self._enrollments: Dict[str, List[str]] = {}
        self._progress: Dict[str, Dict] = {}
        
        # Load default courses
        self._load_default_courses()
    
    def _load_default_courses(self):
        """Load default course catalog"""
        self._courses = {
            "intro_to_trading": {
                "id": "intro_to_trading",
                "title": "Introduction to Trading",
                "description": "Learn the basics of trading and financial markets",
                "level": "beginner",
                "duration_hours": 10,
                "modules": [
                    "What is Trading?",
                    "Types of Financial Instruments",
                    "Reading Charts",
                    "Basic Order Types"
                ],
                "prerequisites": []
            },
            "market_basics": {
                "id": "market_basics",
                "title": "Market Basics",
                "description": "Understanding how markets work",
                "level": "beginner",
                "duration_hours": 12,
                "modules": [
                    "Market Participants",
                    "Supply and Demand",
                    "Market Hours",
                    "Market Indicators"
                ],
                "prerequisites": ["intro_to_trading"]
            },
            "python_for_trading": {
                "id": "python_for_trading",
                "title": "Python for Trading",
                "description": "Learn Python programming for algorithmic trading",
                "level": "intermediate",
                "duration_hours": 20,
                "modules": [
                    "Python Basics",
                    "Pandas for Data Analysis",
                    "NumPy for Calculations",
                    "Building Your First Strategy"
                ],
                "prerequisites": ["market_basics"]
            },
            "backtesting_mastery": {
                "id": "backtesting_mastery",
                "title": "Backtesting Mastery",
                "description": "Master the art of backtesting trading strategies",
                "level": "intermediate",
                "duration_hours": 15,
                "modules": [
                    "Backtesting Fundamentals",
                    "Data Quality",
                    "Avoiding Overfitting",
                    "Performance Metrics"
                ],
                "prerequisites": ["python_for_trading"]
            },
            "machine_learning_trading": {
                "id": "machine_learning_trading",
                "title": "Machine Learning for Trading",
                "description": "Apply ML techniques to trading strategies",
                "level": "advanced",
                "duration_hours": 30,
                "modules": [
                    "ML Fundamentals",
                    "Feature Engineering",
                    "Model Selection",
                    "ML Strategy Development"
                ],
                "prerequisites": ["backtesting_mastery"]
            }
        }
    
    def get_catalog(self, level: Optional[str] = None) -> List[Dict]:
        """
        Get course catalog
        
        Args:
            level: Filter by level (beginner, intermediate, advanced, expert)
            
        Returns:
            List of courses
        """
        courses = list(self._courses.values())
        
        if level:
            courses = [c for c in courses if c["level"] == level]
        
        return courses
    
    def get_course(self, course_id: str) -> Optional[Dict]:
        """Get course by ID"""
        return self._courses.get(course_id)
    
    def enroll_user(self, user_id: str, course_id: str) -> bool:
        """
        Enroll user in a course
        
        Args:
            user_id: User identifier
            course_id: Course identifier
            
        Returns:
            True if enrolled successfully
        """
        if course_id not in self._courses:
            raise ValueError(f"Course not found: {course_id}")
        
        # Check prerequisites
        course = self._courses[course_id]
        for prereq in course.get("prerequisites", []):
            if not self.has_completed(user_id, prereq):
                raise ValueError(f"Prerequisite not met: {prereq}")
        
        # Enroll user
        if user_id not in self._enrollments:
            self._enrollments[user_id] = []
        
        if course_id not in self._enrollments[user_id]:
            self._enrollments[user_id].append(course_id)
        
        # Initialize progress
        progress_key = f"{user_id}:{course_id}"
        self._progress[progress_key] = {
            "status": CourseStatus.NOT_STARTED.value,
            "enrolled_at": datetime.utcnow().isoformat(),
            "progress_percent": 0,
            "completed_modules": [],
            "quiz_scores": {}
        }
        
        return True
    
    def get_progress(self, user_id: str, course_id: str) -> Dict:
        """
        Get user's progress in a course
        
        Args:
            user_id: User identifier
            course_id: Course identifier
            
        Returns:
            Progress data
        """
        progress_key = f"{user_id}:{course_id}"
        return self._progress.get(progress_key, {})
    
    def update_progress(
        self,
        user_id: str,
        course_id: str,
        module_completed: Optional[str] = None,
        quiz_score: Optional[Dict] = None
    ):
        """
        Update user's progress
        
        Args:
            user_id: User identifier
            course_id: Course identifier
            module_completed: Module that was completed
            quiz_score: Quiz results
        """
        progress_key = f"{user_id}:{course_id}"
        
        if progress_key not in self._progress:
            raise ValueError("User not enrolled in course")
        
        progress = self._progress[progress_key]
        
        if module_completed:
            if module_completed not in progress["completed_modules"]:
                progress["completed_modules"].append(module_completed)
        
        if quiz_score:
            progress["quiz_scores"].update(quiz_score)
        
        # Update progress percentage
        course = self._courses[course_id]
        total_modules = len(course["modules"])
        completed = len(progress["completed_modules"])
        progress["progress_percent"] = (completed / total_modules) * 100
        
        # Update status
        if progress["progress_percent"] == 100:
            progress["status"] = CourseStatus.COMPLETED.value
            progress["completed_at"] = datetime.utcnow().isoformat()
        elif progress["progress_percent"] > 0:
            progress["status"] = CourseStatus.IN_PROGRESS.value
    
    def has_completed(self, user_id: str, course_id: str) -> bool:
        """Check if user has completed a course"""
        progress = self.get_progress(user_id, course_id)
        return progress.get("status") == CourseStatus.COMPLETED.value
    
    def issue_certificate(self, user_id: str, course_id: str) -> Dict:
        """
        Issue course completion certificate
        
        Args:
            user_id: User identifier
            course_id: Course identifier
            
        Returns:
            Certificate data
        """
        if not self.has_completed(user_id, course_id):
            raise ValueError("Course not completed")
        
        course = self._courses[course_id]
        progress = self.get_progress(user_id, course_id)
        
        certificate = {
            "certificate_id": f"CERT-{user_id}-{course_id}-{datetime.utcnow().timestamp()}",
            "user_id": user_id,
            "course_id": course_id,
            "course_title": course["title"],
            "issued_at": datetime.utcnow().isoformat(),
            "completed_at": progress.get("completed_at"),
            "verification_url": f"https://zipline.io/verify/certificate/{user_id}/{course_id}"
        }
        
        return certificate
