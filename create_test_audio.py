#!/usr/bin/env python3
"""
Create a simple test audio file for voice-to-voice testing.
This generates a synthetic audio file that can be used to test the STT pipeline.
"""

import numpy as np
import wave
import struct
import os

def create_test_audio():
    """Create a simple test audio file with speech-like characteristics."""

    # Audio parameters
    sample_rate = 16000  # 16kHz - good for speech recognition
    duration = 3  # seconds
    frequency = 440  # A note

    # Generate a simple sine wave (not real speech, but can test audio pipeline)
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # Create a more complex signal that resembles speech patterns
    # Mix multiple frequencies to simulate speech-like audio
    audio = np.zeros_like(t)
    frequencies = [300, 600, 1200, 2400]  # Formant-like frequencies
    amplitudes = [0.3, 0.2, 0.1, 0.05]

    for freq, amp in zip(frequencies, amplitudes):
        audio += amp * np.sin(2 * np.pi * freq * t)

    # Add some noise to make it more realistic
    noise = 0.01 * np.random.normal(0, 1, len(audio))
    audio += noise

    # Normalize to prevent clipping
    audio = audio / np.max(np.abs(audio))

    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)

    # Save as WAV file using wave module
    output_file = "/workspaces/codespaces-jupyter/test_audio.wav"

    with wave.open(output_file, 'wb') as wav_file:
        # Set parameters: nchannels, sampwidth, framerate, nframes, comptype, compname
        wav_file.setparams((1, 2, sample_rate, len(audio_int16), 'NONE', 'not compressed'))

        # Write audio data
        for sample in audio_int16:
            wav_file.writeframes(struct.pack('<h', sample))

    print(f"✅ Test audio file created: {output_file}")
    print(f"   Duration: {duration} seconds")
    print(f"   Sample rate: {sample_rate} Hz")
    print(f"   Format: 16-bit WAV")

    return output_file

def create_silent_audio():
    """Create a silent audio file for testing audio processing pipeline."""

    sample_rate = 16000
    duration = 2  # seconds
    samples = int(sample_rate * duration)

    # Create silent audio (all zeros)
    silent_audio = np.zeros(samples, dtype=np.int16)

    output_file = "/workspaces/codespaces-jupyter/silent_test.wav"

    with wave.open(output_file, 'wb') as wav_file:
        wav_file.setparams((1, 2, sample_rate, len(silent_audio), 'NONE', 'not compressed'))

        for sample in silent_audio:
            wav_file.writeframes(struct.pack('<h', sample))

    print(f"✅ Silent test audio file created: {output_file}")
    return output_file

if __name__ == "__main__":
    print("🎵 Creating test audio files for voice processing...")

    try:
        # Create test audio files
        test_file = create_test_audio()
        silent_file = create_silent_audio()

        print("\n📝 Test files created:")
        print(f"   Synthetic speech-like audio: {test_file}")
        print(f"   Silent audio: {silent_file}")
        print("\n🎯 Use these files to test the voice-to-voice pipeline:")
        print("   cd Roboto.SAI && python -c \"from app1 import Roboto; r = Roboto(); result = r.voice_to_voice_conversation('../test_audio.wav'); print(result)\"")

    except Exception as e:
        print(f"❌ Error creating test audio: {e}")
        print("💡 Make sure numpy is installed: pip install numpy")