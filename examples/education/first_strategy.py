"""
First Strategy Tutorial

An interactive example demonstrating the education platform's capabilities.
"""

from zipline.education.courses.platform import CoursePlatform
from zipline.education.courses.tracks import list_learning_tracks
from zipline.education.library.glossary import TradingGlossary


def main():
    """
    Run education platform demonstration
    """
    print("""
    ╔═══════════════════════════════════════════╗
    ║   Zipline Education Platform Example     ║
    ╚═══════════════════════════════════════════╝
    
    Welcome to the Zipline Education Platform!
    """)
    
    # Initialize platform
    platform = CoursePlatform()
    
    # Display learning tracks
    print("\n📚 Available Learning Tracks:\n")
    tracks = list_learning_tracks()
    
    for i, track in enumerate(tracks, 1):
        print(f"{i}. {track['name']}")
        print(f"   Duration: {track['duration']}")
        print(f"   Certificate: {track['certificate']}")
        print()
    
    # Get course catalog
    print("\n📖 Courses:\n")
    catalog = platform.get_catalog()
    
    for course in catalog[:3]:
        print(f"• {course['title']}")
        print(f"  Level: {course['level']} | Duration: {course['duration_hours']}h")
        print()
    
    # Demonstrate glossary
    print("\n📖 Trading Glossary Sample:\n")
    glossary = TradingGlossary()
    
    term = glossary.get_term("algorithmic_trading")
    if term:
        print(f"Term: {term['term']}")
        print(f"Definition: {term['definition']}")
        print()


if __name__ == "__main__":
    main()
