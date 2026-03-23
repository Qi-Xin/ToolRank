"""Seed the database with top MCPs."""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from api import database as db
from api.embeddings import get_default_provider, TFIDFEmbeddingProvider

SEED_FILE = os.path.join(os.path.dirname(__file__), "../seed_data/top_mcps.json")


def seed():
    db.init_db()
    provider = get_default_provider()

    with open(SEED_FILE) as f:
        tools = json.load(f)

    # Fit TF-IDF on all descriptions before inserting
    if isinstance(provider, TFIDFEmbeddingProvider):
        provider.fit([t["description"] for t in tools])

    inserted = 0
    skipped = 0
    for tool in tools:
        existing = db.get_tool_by_name(tool["name"])
        if existing:
            skipped += 1
            continue
        embedding = provider.embed(tool["description"])
        tool["embedding"] = embedding
        db.insert_tool(tool)
        inserted += 1
        print(f"  + {tool['name']}")

    print(f"\nDone: {inserted} inserted, {skipped} skipped.")


if __name__ == "__main__":
    seed()
