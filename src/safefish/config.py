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


class NormalizeCfg(_Cfg):
    """V2+: normalización de cada fuente a `CaseDoc | MessageDoc | SignalDoc | GuideDoc`.

    - `auto`: usa el parser de plantilla si el documento ya viene estructurado (secciones
      `Descripción:/Frases típicas:/Mecanismo:` o pasos numerados) y el LLM si no.
    - `llm`: siempre el LLM. `template`: nunca el LLM (lo que no parsea queda como texto crudo).
    """

    mode: Literal["auto", "llm", "template"] = "auto"
    model: str = "gpt-4.1-mini-2025-04-14"
    prompt: str = "normalizer_v1"
    min_frase_score: float = 0.85  # frases típicas del LLM que no están en la fuente se descartan
    review_fraction: float = 0.10  # muestra para revisión manual (§2.3)
    review_seed: int = 7
    timeout_s: float = 90.0


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
    normalize: NormalizeCfg = Field(default_factory=NormalizeCfg)
    min_casos_por_modalidad: int = 5  # umbral del panel de cobertura (§2.3)


class ChunkingCfg(_Cfg):
    strategy: Literal["recursive", "per_case"] = "recursive"
    chunk_size: int = 800
    chunk_overlap: int = 120  # 15 % de 800
    separators: list[str] = Field(default_factory=lambda: ["\n\n", "\n", ".", " "])


class IndexCfg(_Cfg):
    """`single`: una colección con los cuatro tipos (V1, V2a). `per_tipo`: una colección por
    tipo de documento (V2+), con `top_n` por colección."""

    layout: Literal["single", "per_tipo"] = "single"


class EmbeddingsCfg(_Cfg):
    model: str = "text-embedding-3-small"


class QueryCfg(_Cfg):
    strategy: Literal["raw", "multi_query"] = "raw"
    facets: list[str] = Field(default_factory=list)
    model: str = "claude-haiku-5-5"  # multi_query (V4)
    prompt: str = "query_rewrite_v1"
    max_queries: int = 4  # texto crudo + hasta 3 focalizadas


class HybridCfg(_Cfg):
    enabled: bool = False
    weights: dict[str, float] = Field(default_factory=lambda: {"dense": 1.0, "bm25": 0.0})


class RerankCfg(_Cfg):
    enabled: bool = False
    model: str | None = None  # LLM listwise (V6): p. ej. gpt-4.1-mini-2025-04-14
    prompt: str = "rerank_v1"
    top_in: int = 20  # candidatos fusionados por colección que entran al rerank
    max_chars: int = 600  # por candidato, para acotar tokens


class RetrievalCfg(_Cfg):
    enabled: bool = True
    query: QueryCfg = Field(default_factory=QueryCfg)
    filter_canal: Literal["off", "hard", "soft"] = "off"
    hybrid: HybridCfg = Field(default_factory=HybridCfg)
    rerank: RerankCfg = Field(default_factory=RerankCfg)
    # V7 candidato (evidencia contrastiva): mitad de los vecinos de `mensajes` con label=estafa
    # y mitad con label=legitimo. Solo con el retriever avanzado (layout per_tipo).
    mensajes_contrastivo: bool = False
    top_k: int = 6  # layout single: top-k total
    top_n: dict[str, int] = Field(default_factory=dict)  # layout per_tipo: por colección

    # k fijo para las métricas de retrieval de casos. Fijarlo igual en V1/V2a/V2 hace
    # comparables Hit/MRR/Recall/nDCG@k aunque cambie cuántos casos llegan al contexto.
    k_eval: int | None = None

    def k_casos(self) -> int:
        """k con el que se miden las métricas de retrieval de casos."""
        if self.k_eval is not None:
            return self.k_eval
        return self.top_n.get("casos", self.top_k) if self.top_n else self.top_k

    @field_validator("filter_canal", mode="before")
    @classmethod
    def _yaml_off(cls, v):  # YAML 1.1 parsea `off` como False
        return "off" if v is False else v


class GenerationCfg(_Cfg):
    reasoner: str = "claude-sonnet-5-5"
    temperature: float = 0.0
    seed: int | None = 7
    prompt: str = "reasoner_v1"
    # V10: cómo se le presenta al razonador el contexto del usuario. v1 = solo para interpretar;
    # v2 = sus hechos concretos (quién lo mandó, suplantación, pedidos por otro canal) pesan.
    contexto_prompt: Literal["v1", "v2"] = "v1"
    max_retries: int = 1  # reintentos ante JSON inválido (§2.2)
    planner: bool = False  # V2+: plan desde Guías (M-GEN.2)
    planner_mode: Literal["llm", "extractivo"] = "llm"  # extractivo = pasos literales, sin LLM
    planner_model: str = "claude-haiku-5-5"
    planner_prompt: str = "planner_v1"
    planner_top_guias: int = 2
    planner_max_pasos: int = 5
    timeout_s: float = 60.0


class PostprocessCfg(_Cfg):
    verifier: bool = False
    rules: list[Literal["R1", "R2", "R3", "R4", "R5"]] = Field(default_factory=list)
    cita_threshold: float = 0.9  # R1: tolerancia fuzzy de la cita literal
    bbox_threshold: float = 0.8  # alineación cita → palabras OCR (captura)
    # V8: si la modalidad del razonador no está entre los casos recuperados, antes de R2 se
    # busca el caso más parecido de esa modalidad; se acepta como respaldo si su similitud está
    # a lo sumo `margen` por debajo del mejor caso de toda la colección. None = apagado.
    soporte_dirigido_margen: float | None = None


class ExtractionCfg(_Cfg):
    """M-EXT para capturas: LLM con visión (transcripción estructurada) + OCR con cajas."""

    vision_model: str = "claude-sonnet-5-5"
    vision_prompt: str = "vision_v2"
    ocr: Literal["tesseract", "none"] = "tesseract"
    ocr_lang: str = "spa"
    ocr_min_conf: float = 30.0
    preprocess: bool = True  # resize + binarización (Clase 10)
    timeout_s: float = 90.0


class PipelineConfig(_Cfg):
    version: str
    descripcion: str = ""
    kb_snapshot: str | None = None
    ingest: IngestCfg = Field(default_factory=IngestCfg)
    chunking: ChunkingCfg = Field(default_factory=ChunkingCfg)
    index: IndexCfg = Field(default_factory=IndexCfg)
    embeddings: EmbeddingsCfg = Field(default_factory=EmbeddingsCfg)
    retrieval: RetrievalCfg = Field(default_factory=RetrievalCfg)
    generation: GenerationCfg = Field(default_factory=GenerationCfg)
    postprocess: PostprocessCfg = Field(default_factory=PostprocessCfg)
    extraction: ExtractionCfg = Field(default_factory=ExtractionCfg)

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
