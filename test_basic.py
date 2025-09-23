#!/usr/bin/env python3
"""
Basic test script to verify the project structure and imports.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from data_processing import XMLParser, TextExtractor, Event, VerseReference
        print("✓ Data processing modules imported successfully")
    except ImportError as e:
        print(f"✗ Data processing import failed: {e}")
        return False
    
    try:
        from fuzzy_relations import FuzzyRelationCalculator, MembershipFunctions
        print("✓ Fuzzy relations modules imported successfully")
    except ImportError as e:
        print(f"✗ Fuzzy relations import failed: {e}")
        return False
    
    try:
        from graph_neural_network import FuzzyEventGraph, FuzzyGNN
        print("✓ Graph neural network modules imported successfully")
    except ImportError as e:
        print(f"✗ Graph neural network import failed: {e}")
        return False
    
    try:
        from summarization import ConsolidationSummarizer
        print("✓ Summarization modules imported successfully")
    except ImportError as e:
        print(f"✗ Summarization import failed: {e}")
        return False
    
    try:
        from evaluation import EvaluationSuite
        print("✓ Evaluation modules imported successfully")
    except ImportError as e:
        print(f"✗ Evaluation import failed: {e}")
        return False
    
    return True

def test_data_structures():
    """Test basic data structure functionality"""
    print("\\nTesting data structures...")
    
    try:
        from data_processing import VerseReference, Event
        
        # Test VerseReference parsing
        ref = VerseReference.parse("21:1-7")
        assert ref.chapter == 21
        assert ref.verse_start == 1
        assert ref.verse_end == 7
        print("✓ VerseReference parsing works")
        
        # Test Event creation
        event = Event(
            id=1,
            day="Palm Sunday",
            description="Jesus enters Jerusalem",
            when_where="Jerusalem",
            gospel_refs={'matthew': ref}
        )
        assert event.id == 1
        assert 'matthew' in event.participating_gospels
        print("✓ Event creation works")
        
        return True
    except Exception as e:
        print(f"✗ Data structure test failed: {e}")
        return False

def test_file_structure():
    """Test that all required files exist"""
    print("\\nTesting file structure...")
    
    required_files = [
        'README.md',
        'requirements.txt',
        'config.yaml',
        'LICENSE',
        '.gitignore',
        'src/main.py',
        'data/raw',
        'results',
        'logs'
    ]
    
    project_root = Path(__file__).parent
    
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} missing")
            return False
    
    return True

def test_data_files():
    """Test that XML data files are present"""
    print("\\nTesting data files...")
    
    data_dir = Path(__file__).parent / 'data' / 'raw'
    
    required_files = [
        'ChronologyOfTheFourGospels_PW.xml',
        'EnglishNIVMatthew40_PW.xml',
        'EnglishNIVMark41_PW.xml',
        'EnglishNIVLuke42_PW.xml',
        'EnglishNIVJohn43_PW.xml'
    ]
    
    for filename in required_files:
        file_path = data_dir / filename
        if file_path.exists():
            print(f"✓ {filename} found")
        else:
            print(f"✗ {filename} missing")
            return False
    
    return True

def main():
    """Run all tests"""
    print("=== Fuzzy Gospel Consolidation - Basic Tests ===\\n")
    
    tests = [
        test_file_structure,
        test_data_files,
        test_imports,
        test_data_structures
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"=== Test Results: {passed}/{total} passed ===")
    
    if passed == total:
        print("🎉 All tests passed! The project structure is ready.")
        return 0
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
