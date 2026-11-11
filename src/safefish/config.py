"""Sistema de configuración por versión (§1.5).

Cada versión de la ablación es un YAML en `configs/`. El pipeline se arma desde el config,
no por branches de código. El `config_hash` identifica la configuración en `metrics.json`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from safefish.schemas import DocTipo

REPO_ROOT = Path(__file__).resolve().parents[2]
KB_SNAPSHOTS_DIR = REPO_ROOT / "data" / "kb_snapshots"
CACHE_DIR = REPO_ROOT / ".cache"


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DedupeCfg(_Cfg):
    exact: bool = True
    near_threshold: float | None = 0.95  # coseno; None desactiva la deduplicación cercana
    near_method: Literal["tfidf", "embeddings"] = "tfidf"


class IngestCfg(_Cfg):
    sources: str = "data/sources.yaml"
    languages: list[str] = Field(default_factory=lambda: ["es"])
    min_chars_lang_detect: int = 60
    dedupe: DedupeCfg = Field(default_factory=DedupeCfg)
    # Redacción de PII por defecto según tipo de documento; cada fuente puede sobreescribirla.
    pii_por_tipo: dict[DocTipo, bool] = Field(
        default_factory=lambda: {
            DocTipo.CASO: True,
            DocTipo.MENSAJE: True,
            DocTipo.SENAL: False,
            DocTipo.GUIA: False,
        }
    )


class ChunkingCfg(_Cfg):
    strategy: Literal["recursive", "per_case"] = "recursive"
    chunk_size: int = 800
    chunk_overlap: int = 120  # 15 % de 800
    separators: list[str] = Field(default_factory=lambda: ["\n\n", "\n", ".", " "])


class EmbeddingsCfg(_Cfg):
    model: str = "text-embedding-3-small"


class QueryCfg(_Cfg):
    strategy: Literal["raw", "multi_query"] = "raw"
    facets: list[str] = Field(default_factory=list)


class HybridCfg(_Cfg):
    enabled: bool = False
    weights: dict[str, float] = Field(default_factory=lambda: {"dense": 1.0, "bm25": 0.0})


class RerankCfg(_Cfg):
    enabled: bool = False
    model: str | None = None
    top_in: int = 20


class RetrievalCfg(_Cfg):
    enabled: bool = True
    query: QueryCfg = Field(default_factory=QueryCfg)
    filter_canal: Literal["off", "hard", "soft"] = "off"
    hybrid: HybridCfg = Field(default_factory=HybridCfg)
    rerank: RerankCfg = Field(default_factory=RerankCfg)
    top_k: int = 6  # V1: una colección, top-k total
    top_n: dict[str, int] = Field(default_factory=dict)  # V2+: por colección

    @field_validator("filter_canal", mode="before")
    @classmethod
    def _yaml_off(cls, v):  # YAML 1.1 parsea `off` como False
        return "off" if v is False else v


class GenerationCfg(_Cfg):
    reasoner: str = "gpt-4.1-2025-04-14"
    temperature: float = 0.0
    seed: int | None = 7
    prompt: str = "reasoner_v1"
    max_retries: int = 1  # reintentos ante JSON inválido (§2.2)
    planner: bool = False  # V2+: plan desde Guías
    timeout_s: float = 60.0


class PostprocessCfg(_Cfg):
    verifier: bool = False
    rules: list[str] = Field(default_factory=list)


class PipelineConfig(_Cfg):
    version: str
    descripcion: str = ""
    kb_snapshot: str | None = None
    ingest: IngestCfg = Field(default_factory=IngestCfg)
    chunking: ChunkingCfg = Field(default_factory=ChunkingCfg)
    embeddings: EmbeddingsCfg = Field(default_factory=EmbeddingsCfg)
    retrieval: RetrievalCfg = Field(default_factory=RetrievalCfg)
    generation: GenerationCfg = Field(default_factory=GenerationCfg)
    postprocess: PostprocessCfg = Field(default_factory=PostprocessCfg)

    def config_hash(self) -> str:
        """Hash estable de la configuración efectiva (sin la descripción)."""
        payload = self.model_dump(mode="json", exclude={"descripcion"})
        canon = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canon.encode()).hexdigest()[:12]

    @property
    def snapshot_dir(self) -> Path | None:
        return KB_SNAPSHOTS_DIR / self.kb_snapshot if self.kb_snapshot else None


def load_config(path: str | Path) -> PipelineConfig:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return PipelineConfig.model_validate(data)


def resolve_path(p: str | Path) -> Path:
    """Rutas relativas de los configs se interpretan desde la raíz del repo."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p
