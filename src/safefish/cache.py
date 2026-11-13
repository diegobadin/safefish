"""Caché clave-valor en SQLite para respuestas del LLM y embeddings (§3.3).

La clave es un hash de (modelo, parámetros, prompt): re-correr la evaluación es casi gratis
y las comparaciones entre versiones son deterministas. `SAFEFISH_NO_CACHE=1` la desactiva.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from pathlib import Path


def stable_hash(payload: object) -> str:
    canon = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canon.encode()).hexdigest()


class KVCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v BLOB NOT NULL)")
        self._conn.commit()

    def get(self, key: str) -> bytes | None:
        with self._lock:
            row = self._conn.execute("SELECT v FROM kv WHERE k = ?", (key,)).fetchone()
        return row[0] if row else None

    def get_many(self, keys: list[str]) -> dict[str, bytes]:
        out: dict[str, bytes] = {}
        with self._lock:
            for i in range(0, len(keys), 500):
                chunk = keys[i : i + 500]
                q = f"SELECT k, v FROM kv WHERE k IN ({','.join('?' * len(chunk))})"
                out.update(dict(self._conn.execute(q, chunk).fetchall()))
        return out

    def set_many(self, items: dict[str, bytes]) -> None:
        with self._lock:
            self._conn.executemany("INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)", items.items())
            self._conn.commit()

    def set(self, key: str, value: bytes) -> None:
        self.set_many({key: value})


def cache_enabled() -> bool:
    return os.getenv("SAFEFISH_NO_CACHE", "") not in {"1", "true", "yes"}
