import pytest
from src.rag.policy_library import PolicyLibrary
from src.database import ModeratorDB

def test_policy_library_loads():
    db = ModeratorDB()
    db.connect()
    lib = PolicyLibrary(db)
    assert len(lib.policies) == 6

def test_policy_search():
    db = ModeratorDB()
    db.connect()
    lib = PolicyLibrary(db)
    results = lib.search_policies("violence", top_k=3)
    assert len(results) > 0

def test_embeddings_stored():
    db = ModeratorDB()
    db.connect()
    db.cursor.execute("SELECT COUNT(*) FROM policies")
    count = db.cursor.fetchone()[0]
    assert count == 6
    db.close()
