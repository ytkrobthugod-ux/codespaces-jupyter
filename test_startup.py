#!/usr/bin/env python3
"""
Test script to verify Roboto SAI can start successfully
"""
import sys
import os

def test_imports():
    """Test that all required imports work"""
    try:
        print("Testing core imports...")
        import app_enhanced
        print("✓ app_enhanced imported successfully")

        import quantum_capabilities
        print("✓ quantum_capabilities imported successfully")

        import xai_grok_integration
        print("✓ xai_grok_integration imported successfully")

        import anchored_identity_gate
        print("✓ anchored_identity_gate imported successfully")

        import quantum_simulator
        print("✓ quantum_simulator imported successfully")

        print("\n🎉 All imports successful!")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_app_creation():
    """Test that Flask app can be created"""
    try:
        print("\nTesting Flask app creation...")
        from app_enhanced import app

        # Test app context
        with app.app_context():
            print("✓ Flask app created successfully")
            print(f"✓ App name: {app.name}")
            print(f"✓ Debug mode: {app.debug}")

        return True

    except Exception as e:
        print(f"❌ Flask app creation failed: {e}")
        return False

if __name__ == "__main__":
    print("🤖 Roboto SAI - Startup Test")
    print("=" * 40)

    success = True
    success &= test_imports()
    success &= test_app_creation()

    if success:
        print("\n🎉 All tests passed! Roboto SAI is ready to run.")
        print("\nTo start the app:")
        print("  Development: python run_app.py")
        print("  Production:  ./start_app.sh")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)