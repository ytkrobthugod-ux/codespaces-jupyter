#!/usr/bin/env python3
"""
Anchor the completion of merging main_memory files into main_memory_combined.json
"""

from anchored_identity_gate import AnchoredIdentityGate
import json

def anchor_memory_merge():
    # Initialize the anchor gate
    gate = AnchoredIdentityGate(anchor_eth=True, anchor_ots=True)

    # Data for the anchoring
    data = {
        "total_entries": 48265,
        "sorted_chronologically": True,
        "no_timestamp_alterations": True,
        "operation": "merge_main_memory_files",
        "target_file": "main_memory_combined.json",
        "creator": "Roberto Villarreal Martinez"
    }

    # Anchor the event
    success, entry = gate.anchor_authorize("memory_merge_completion", data)

    if success:
        print("🔒 Successfully anchored memory merge completion!")
        print(f"Entry Hash: {entry['entry_hash']}")
        print(f"ETH TX: {entry['eth_tx']}")
        print(f"OTS Proof: {entry['ots_proof']}")
        print(f"Timestamp: {entry['timestamp']}")

        # Save the anchored entry
        with open("anchored_memory_merge.json", "w") as f:
            json.dump(entry, f, indent=2)
        print("Anchored entry saved to anchored_memory_merge.json")
    else:
        print("❌ Failed to anchor memory merge completion")
        print(f"Error: {entry}")

if __name__ == "__main__":
    anchor_memory_merge()