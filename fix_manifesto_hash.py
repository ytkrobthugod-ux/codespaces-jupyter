#!/usr/bin/env python3
"""
Fix Manifesto Hash - Atomic Update
"""

import hashlib
import re

def compute_sha256(file_path):
    """Compute SHA256 hash of a file"""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def fix_manifesto_hash(manifesto_path):
    """Atomically fix the manifesto hash using iterative approach"""
    max_iterations = 5
    iteration = 0

    while iteration < max_iterations:
        # Compute current hash
        current_hash = compute_sha256(manifesto_path)

        # Read content and replace all hashes with current hash
        with open(manifesto_path, 'r') as f:
            content = f.read()

        # Replace all hash occurrences
        hash_pattern = r'[a-f0-9]{64}'
        new_content = re.sub(hash_pattern, current_hash, content)

        # Write back
        with open(manifesto_path, 'w') as f:
            f.write(new_content)

        # Check if hash is now stable
        new_hash = compute_sha256(manifesto_path)
        if new_hash == current_hash:
            return True, new_hash

        iteration += 1

    return False, "Max iterations reached"

if __name__ == "__main__":
    success, hash_value = fix_manifesto_hash("Roboto_SAI_Signed_Manifesto_Updated.txt")
    if success:
        print(f"✅ Manifesto hash fixed: {hash_value}")
    else:
        print(f"❌ Failed to fix manifesto hash: {hash_value}")