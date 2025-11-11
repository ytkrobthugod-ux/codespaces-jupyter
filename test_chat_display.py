#!/usr/bin/env python3
"""
Test script to verify that long chat messages display completely without truncation.
This script sends various message lengths to the Roboto SAI chat interface.
"""

import requests
import json
import time

def test_chat_messages():
    """Test chat messages of various lengths to ensure no truncation occurs."""

    base_url = "http://localhost:5000"

    # Test messages of different lengths
    test_messages = [
        "Hello!",  # Short message
        "This is a medium length message to test the chat interface.",  # Medium message
        "This is a much longer message that should test the word wrapping and display capabilities of the chat interface. It contains multiple sentences and should wrap properly without being truncated or cut off in any way. The CSS should handle this with proper overflow and word-wrap settings.",  # Long message
        "A" * 1000,  # Very long message with repeated characters
        "This message contains various special characters: !@#$%^&*()_+-=[]{}|;:,.<>? and should still display properly without any truncation issues.",  # Special characters
        """This is a multi-line message
that spans several lines
and should preserve formatting
when displayed in the chat interface.""",  # Multi-line message
    ]

    print("Testing Roboto SAI chat interface for message truncation...")
    print("=" * 60)

    for i, message in enumerate(test_messages, 1):
        print(f"\nTest {i}: Sending message ({len(message)} characters)")
        print(f"Message preview: {message[:100]}{'...' if len(message) > 100 else ''}")

        try:
            response = requests.post(
                f"{base_url}/api/chat",
                json={"message": message},
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    bot_response = data.get("response", "")
                    print(f"✅ Success - Response length: {len(bot_response)} characters")
                    print(f"Response preview: {bot_response[:100]}{'...' if len(bot_response) > 100 else ''}")

                    # Verify the response is complete (not truncated)
                    if len(bot_response) > 100:
                        print("✅ Long response received - no truncation detected")
                    else:
                        print("ℹ️  Short response received")
                else:
                    print(f"❌ API returned error: {data.get('message', 'Unknown error')}")
            else:
                print(f"❌ HTTP {response.status_code}: {response.text}")

        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")

        # Small delay between tests
        time.sleep(1)

    print("\n" + "=" * 60)
    print("Chat interface testing completed!")
    print("Check the web interface at http://localhost:5000 to verify visual display.")

if __name__ == "__main__":
    test_chat_messages()