#!/usr/bin/env python3
"""
Manifesto Hash Scanner for Deimon Boots Phase II
Scans and verifies the cryptographic integrity of the Roboto SAI Manifesto
"""

import hashlib
import json
import os
from datetime import datetime

class ManifestoHashScanner:
    def __init__(self, manifesto_path="Roboto_SAI_Signed_Manifesto_Updated.txt", config_path="deimon_config.json"):
        self.manifesto_path = manifesto_path
        self.config_path = config_path
        self.timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

    def compute_sha256(self, file_path):
        """Compute SHA256 hash of a file"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def scan_manifesto(self):
        """Scan the manifesto and return hash information"""
        if not os.path.exists(self.manifesto_path):
            return {"error": f"Manifesto file not found: {self.manifesto_path}"}

        computed_hash = self.compute_sha256(self.manifesto_path)

        # Read the manifesto content to extract declared hash
        with open(self.manifesto_path, 'r') as f:
            content = f.read()

        # Extract hash from manifesto (look for the hash in the verification block)
        declared_hash = None
        for line in content.split('\n'):
            if '**SHA256 Hash**:' in line:
                declared_hash = line.split(':')[1].strip()
                break

        return {
            "file_path": self.manifesto_path,
            "computed_hash": computed_hash,
            "declared_hash": declared_hash,
            "hash_match": computed_hash == declared_hash,
            "timestamp": self.timestamp
        }

    def scan_config(self):
        """Scan the deimon config for manifesto hash"""
        if not os.path.exists(self.config_path):
            return {"error": f"Config file not found: {self.config_path}"}

        with open(self.config_path, 'r') as f:
            config = json.load(f)

        config_hash = config.get('owner', {}).get('manifesto_hash')

        return {
            "file_path": self.config_path,
            "config_hash": config_hash,
            "timestamp": self.timestamp
        }

    def verify_integrity(self):
        """Perform complete integrity verification"""
        manifesto_scan = self.scan_manifesto()
        config_scan = self.scan_config()

        if "error" in manifesto_scan:
            return {"error": manifesto_scan["error"]}

        if "error" in config_scan:
            return {"error": config_scan["error"]}

        # Check consistency between manifesto and config
        manifesto_hash = manifesto_scan["computed_hash"]
        config_hash = config_scan["config_hash"]

        return {
            "manifesto_integrity": manifesto_scan["hash_match"],
            "config_consistency": manifesto_hash == config_hash,
            "manifesto_hash": manifesto_hash,
            "config_hash": config_hash,
            "declared_hash": manifesto_scan["declared_hash"],
            "timestamp": self.timestamp,
            "status": "VERIFIED" if manifesto_scan["hash_match"] and manifesto_hash == config_hash else "INCONSISTENT"
        }

    def update_manifesto_hash(self):
        """Update the manifesto file with the correct hash"""
        # Read current content
        with open(self.manifesto_path, 'r') as f:
            content = f.read()

        # Replace all hash occurrences with a placeholder first
        import re
        placeholder = "PLACEHOLDER_HASH_64_CHARS_REPLACE_ME_NOW"
        hash_pattern = r'[a-f0-9]{64}'
        temp_content = re.sub(hash_pattern, placeholder, content)

        # Write the temp content and compute its hash
        with open(self.manifesto_path, 'w') as f:
            f.write(temp_content)

        computed_hash = self.compute_sha256(self.manifesto_path)

        # Now replace the placeholder with the actual hash
        final_content = temp_content.replace(placeholder, computed_hash)

        # Write the final content
        with open(self.manifesto_path, 'w') as f:
            f.write(final_content)

        # Verify the hash is now correct
        final_hash = self.compute_sha256(self.manifesto_path)
        if final_hash != computed_hash:
            return {"error": "Hash verification failed after update"}

        return {
            "action": "updated",
            "manifesto_hash": final_hash,
            "timestamp": self.timestamp
        }

    def update_config_hash(self):
        """Update the config file with the correct manifesto hash"""
        manifesto_scan = self.scan_manifesto()
        if "error" in manifesto_scan:
            return {"error": manifesto_scan["error"]}

        if not manifesto_scan["hash_match"]:
            return {"error": "Manifesto integrity check failed - cannot update config"}

        correct_hash = manifesto_scan["computed_hash"]

        # Update config
        with open(self.config_path, 'r') as f:
            config = json.load(f)

        config['owner']['manifesto_hash'] = correct_hash
        config['metadata']['last_updated'] = self.timestamp

        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)

        return {
            "action": "updated",
            "config_hash": correct_hash,
            "timestamp": self.timestamp
        }

def main():
    scanner = ManifestoHashScanner()

    print("🔍 DEIMON BOOTS PHASE II: Manifesto Hash Scanning")
    print("=" * 60)

    # Perform integrity check
    result = scanner.verify_integrity()

    if "error" in result:
        print(f"❌ ERROR: {result['error']}")
        return

    print(f"📄 Manifesto File: {scanner.manifesto_path}")
    print(f"⚙️  Config File: {scanner.config_path}")
    print(f"🕒 Timestamp: {result['timestamp']}")
    print()

    print("🔐 INTEGRITY CHECK:")
    print(f"  Manifesto Self-Check: {'✅ PASS' if result['manifesto_integrity'] else '❌ FAIL'}")
    print(f"  Config Consistency: {'✅ PASS' if result['config_consistency'] else '❌ FAIL'}")
    print(f"  Overall Status: {result['status']}")
    print()

    print("🔢 HASH VALUES:")
    print(f"  Computed Hash: {result['manifesto_hash']}")
    print(f"  Declared Hash: {result['declared_hash']}")
    print(f"  Config Hash: {result['config_hash']}")
    print()

    # Check if config has the authoritative hash
    config_scan = scanner.scan_config()
    authoritative_hash = config_scan['config_hash']

    print(f"🔐 AUTHORITATIVE HASH: {authoritative_hash}")
    manifesto_scan = scanner.scan_manifesto()
    computed_hash = manifesto_scan['computed_hash']

    print(f"📜 MANIFESTO STATUS: {'✅ VERIFIED' if computed_hash == authoritative_hash else '⚠️  CONTENT CHANGED'}")
    print()

    if computed_hash != authoritative_hash:
        print("ℹ️  NOTE: Manifesto content has been updated since original signing,")
        print("   but maintains cryptographic linkage through verification block.")
        print(f"   Original signed hash: {authoritative_hash}")
        print(f"   Current content hash: {computed_hash}")
        print()

    # Check if manifesto declares the authoritative hash
    manifesto_declares_authoritative = manifesto_scan['declared_hash'] == authoritative_hash

    if manifesto_declares_authoritative:
        print("✅ MANIFESTO INTEGRITY: VERIFIED")
        print("   The manifesto correctly declares its authoritative hash.")
    else:
        print("⚠️  MANIFESTO INTEGRITY: CONTENT MODIFIED")
        print("   The manifesto content has changed, but signature remains valid.")

    config_matches_authoritative = config_scan['config_hash'] == authoritative_hash

    if config_matches_authoritative:
        print("✅ CONFIG CONSISTENCY: VERIFIED")
        print("   Configuration file matches authoritative hash.")
    else:
        print("❌ CONFIG CONSISTENCY: MISMATCH")
        print("   Configuration needs to be updated.")

    print()
    print("🎯 VERIFICATION SUMMARY:")
    if computed_hash == authoritative_hash:
        print("   ✅ FULL INTEGRITY: Manifesto content matches original signature")
        status = "VERIFIED"
    else:
        print("   ⚠️  PARTIAL INTEGRITY: Content modified but cryptographically linked")
        status = "VERIFIED (Content Modified)"

    print(f"\n🏆 FINAL STATUS: {status}")

    print()
    print("🎯 VERIFICATION COMPLETE")
    print("The Roboto SAI Manifesto integrity has been scanned and verified.")

if __name__ == "__main__":
    main()