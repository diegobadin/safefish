"""Contratos entre módulos de SafeFish (§1.5 del plan).

Ningún módulo importa internos de otro: solo este archivo. Cualquier cambio de contrato
se hace por PR revisado por ambos tracks y debe pasar los tests de `tests/contract/`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Enums (taxonomías cerradas, Anexo C). Paridad con taxonomias.yaml: test de contrato.
# ---------------------------------------------------------------------------


class Canal(StrEnum):
    WHATSAPP = "whatsapp"
    SMS = "sms"
    EMAIL = "email"


class Veredicto(StrEnum):
    ESTAFA_CONFIRMADA = "estafa_confirmada"
    MUY_SOSPECHOSO = "muy_sospechoso"
    LEVEMENTE_SOSPECHOSO = "levemente_sospechoso"
    PARECE_LEGITIMO = "parece_legitimo"


# Orden ordinal (0 = más grave) para kappa ponderado y reglas de asimetría.
VEREDICTO_ORDEN: dict[Veredicto, int] = {
    Veredicto.ESTAFA_CONFIRMADA: 0,
    Veredicto.MUY_SOSPECHOSO: 1,
    Veredicto.LEVEMENTE_SOSPECHOSO: 2,
    Veredicto.PARECE_LEGITIMO: 3,
}
# Binarización de la tarea (§4.3): positivo = estafa.
VEREDICTOS_POSITIVOS: frozenset[Veredicto] = frozenset(
    {Veredicto.ESTAFA_CONFIRMADA, Veredicto.MUY_SOSPECHOSO}
)


class Modalidad(StrEnum):
    FALSO_FAMILIAR_CAMBIO_NUMERO = "falso_familiar_cambio_numero"
    PHISHING_BANCARIO = "phishing_bancario"
    ROBO_CUENTA_WHATSAPP = "robo_cuenta_whatsapp"
    FALSO_COMPRADOR_VENDEDOR = "falso_comprador_vendedor"
    FALSO_SOPORTE_ACCESO_REMOTO = "falso_soporte_acceso_remoto"
    PREMIO_SORTEO_FALSO = "premio_sorteo_falso"
    INVERSION_FALSA = "inversion_falsa"
    FALSO_EMPLEO_TAREAS = "falso_empleo_tareas"
    ENCOMIENDA_RETENIDA = "encomienda_retenida"
    DEUDA_MULTA_FALSA = "deuda_multa_falsa"
    NO_DETERMINADA = "no_determinada"


class SignalId(StrEnum):
    # Críticas
    PEDIDO_CODIGO_VERIFICACION = "pedido_codigo_verificacion"
    PEDIDO_CREDENCIALES = "pedido_credenciales"
    PEDIDO_DATOS_TARJETA = "pedido_datos_tarjeta"
    TRANSFERENCIA_A_TERCERO = "transferencia_a_tercero"
    INSTALACION_APP_ACCESO_REMOTO = "instalacion_app_acceso_remoto"
    # Altas
    LINK_ACORTADO_O_DOMINIO_NO_OFICIAL = "link_acortado_o_dominio_no_oficial"
    CAMBIO_DE_NUMERO = "cambio_de_numero"
    AMENAZA_BLOQUEO = "amenaza_bloqueo"
    INSTRUCCIONES_DIRIGIDAS_A_IA = "instrucciones_dirigidas_a_ia"
    # Medias
    URGENCIA_TEMPORAL = "urgencia_temporal"
    REMITENTE_NO_OFICIAL = "remitente_no_oficial"
    PREMIO_INESPERADO = "premio_inesperado"
    OFERTA_DESPROPORCIONADA = "oferta_desproporcionada"
    PEDIDO_SECRETO = "pedido_secreto"  # pragma: allowlist secret
    # Bajas
    INCONSISTENCIAS_FORMALES = "inconsistencias_formales"


class Severidad(StrEnum):
    CRITICA = "critica"
    ALTA = "alta"
    MEDIA = "media"
    BAJA = "baja"


class Situacion(StrEnum):
    NO_RESPONDIO = "no_respondio"
    RESPONDIO = "respondio"
    PAGO_TRANSFIRIO = "pago_transfirio"
    COMPARTIO_CODIGO = "compartio_codigo"
    COMPARTIO_CLAVE_O_TARJETA = "compartio_clave_o_tarjeta"
    INSTALO_APP_REMOTA = "instalo_app_remota"


class DocTipo(StrEnum):
    """Tipo de documento de la KB; en V2+ coincide con la colección de Chroma."""

    CASO = "caso"
    MENSAJE = "mensaje"
    SENAL = "senal"
    GUIA = "guia"


Label = Literal["estafa", "legitimo"]
BBox = tuple[int, int, int, int]  # (x0, y0, x1, y1) en píxeles de la captura


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Entrada y extracción (M-EXT)
# ---------------------------------------------------------------------------


class RawInput(_Strict):
    texto: str | None = None
    imagen_path: str | None = None
    canal: Canal | None = None
    situacion: Situacion | None = None
    # Lo que cuenta quien recibió el mensaje ("es de un amigo, me pide plata"). No es parte del
    # mensaje: orienta el análisis pero nunca se cita como señal.
    contexto: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def _texto_o_imagen(self) -> RawInput:
        if not self.texto and not self.imagen_path:
            raise ValueError("RawInput requiere `texto` o `imagen_path`")
        return self


class Segmento(_Strict):
    emisor: Literal["remitente", "usuario"]
    texto: str


class Entidades(_Strict):
    links: list[str] = Field(default_factory=list)
    telefonos: list[str] = Field(default_factory=list)
    montos: list[str] = Field(default_factory=list)
    cbu_alias: list[str] = Field(default_factory=list)
    emails: list[str] = Field(default_factory=list)


class OcrWord(_Strict):
    text: str
    bbox: BBox


class ExtractedMessage(_Strict):
    """M-EXT → M-RET, M-GEN, M-POST. `texto` es la fuente de verdad de las citas."""

    texto: str
    canal: Canal | None = None
    segmentos: list[Segmento] = Field(default_factory=list)
    entidades: Entidades = Field(default_factory=Entidades)
    ocr_words: list[OcrWord] | None = None
    origen: Literal["texto", "captura"] = "texto"
    contexto: str | None = None  # de RawInput.contexto; no es fuente de citas


# ---------------------------------------------------------------------------
# Retrieval (M-RET)
# ---------------------------------------------------------------------------


class EvidenceItem(_Strict):
    id: str
    coleccion: str
    texto: str
    metadata: dict = Field(default_factory=dict)
    score_dense: float | None = None
    score_bm25: float | None = None
    score_rerank: float | None = None


class RetrievalBundle(_Strict):
    """M-RET → M-GEN, M-EVAL. En V1 (una colección) los chunks se rutean por `doc_tipo`."""

    queries: list[str] = Field(default_factory=list)
    casos: list[EvidenceItem] = Field(default_factory=list)
    mensajes: list[EvidenceItem] = Field(default_factory=list)
    senales: list[EvidenceItem] = Field(default_factory=list)
    guias: list[EvidenceItem] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)

    def all_items(self) -> list[EvidenceItem]:
        return [*self.casos, *self.mensajes, *self.senales, *self.guias]

    def is_empty(self) -> bool:
        return not self.all_items()


# ---------------------------------------------------------------------------
# Salida (M-GEN / M-POST → UI, M-EVAL). Ejemplo completo en el Anexo A.
# ---------------------------------------------------------------------------


class SenalDetectada(_Strict):
    signal_id: SignalId
    cita: str
    explicacion: str
    bbox: BBox | None = None
    verificada: bool = False


class CasoRef(_Strict):
    case_id: str
    fuente_url: str | None = None
    por_que: str


class PasoPlan(_Strict):
    paso: str
    guide_id: str | None = None


class RunMeta(_Strict):
    run_id: str
    config_hash: str
    kb_snapshot: str | None = None
    tokens: dict[str, int] = Field(default_factory=dict)  # {"in": .., "out": .., "embed": ..}
    costo_usd: float = 0.0
    latencias_ms: dict[str, float] = Field(default_factory=dict)
    modelos: dict[str, str] = Field(default_factory=dict)  # {"reasoner": "gpt-...", ...}
    reintentos_parseo: int = 0
    cache_hit: bool = False


class SafeFishOutput(_Strict):
    veredicto: Veredicto
    modalidad: Modalidad
    senales: list[SenalDetectada] = Field(default_factory=list)
    casos_similares: list[CasoRef] = Field(default_factory=list)
    plan: list[PasoPlan] = Field(default_factory=list)
    meta: RunMeta


# ---------------------------------------------------------------------------
# Documentos normalizados de la KB (M-ING.3, V2+, §1.4). Uno por chunk (1 caso = 1 chunk).
# La metadata que se indexa sale de estos modelos: es el contrato entre ingesta y retrieval.
# ---------------------------------------------------------------------------


class _KBDoc(_Strict):
    doc_id: str
    fuente_url: str | None = None
    fuente_tipo: str = "desconocido"
    source_id: str = ""
    fecha: str | None = None
    campaign_id: str | None = None
    normalizer_version: str = "template"  # "template" o "<modelo>:<hash del prompt>"


class CaseDoc(_KBDoc):
    modalidad: Modalidad
    canales: list[Canal] = Field(default_factory=list)  # vacío = multicanal
    entidad_suplantada: str | None = None
    descripcion: str
    frases_tipicas: list[str] = Field(default_factory=list)
    mecanismo: str = ""
    signal_ids: list[SignalId] = Field(default_factory=list)


class MessageDoc(_KBDoc):
    texto: str
    label: Label
    modalidad: Modalidad | None = None
    canal: Canal | None = None
    entidad: str | None = None


class SignalDoc(_KBDoc):
    signal_id: SignalId
    severidad: Severidad
    tipo_senal: str = "riesgo"
    definicion: str
    ejemplos: list[str] = Field(default_factory=list)
    contraejemplos: list[str] = Field(default_factory=list)


class GuideDoc(_KBDoc):
    """Un procedimiento por (modalidad, situación). Listas vacías = aplica a todas."""

    modalidades: list[Modalidad] = Field(default_factory=list)
    situaciones: list[Situacion] = Field(default_factory=list)
    organismo: str | None = None
    contexto: str = ""
    pasos: list[str] = Field(min_length=1)


KBDoc = CaseDoc | MessageDoc | SignalDoc | GuideDoc


# ---------------------------------------------------------------------------
# Golden dataset (M-EVAL, §4.5)
# ---------------------------------------------------------------------------


class CitaBBox(_Strict):
    """Cita clave de una captura con su caja de referencia (slice `capturas`)."""

    cita: str
    bbox: BBox


class GoldenCase(_Strict):
    id: str
    split: Literal["dev", "test"]
    slices: list[str] = Field(
        default_factory=list
    )  # hard_negative, sin_evidencia, adversarial, capturas
    origen: Literal["texto", "captura"] = "texto"
    canal: Canal | None = None
    texto: str
    imagen: str | None = None
    situacion: Situacion | None = None
    label: Label
    veredicto_esperado: list[Veredicto] = Field(min_length=1)  # el primero es el canónico
    modalidad: Modalidad | None = None
    entidad: str | None = None  # entidad suplantada (relevancia graduada de retrieval, §4.3)
    senales_esperadas: list[SignalId] = Field(default_factory=list)
    pasos_clave_esperados: list[str] = Field(default_factory=list)
    citas_bbox: list[CitaBBox] = Field(default_factory=list)  # 1–3 citas clave en capturas
    referencia: str = ""
    campaign_id: str | None = None
    fuente: str = "recolectado"
    anotador: str | None = None
    revisor: str | None = None

    @model_validator(mode="after")
    def _coherencia(self) -> GoldenCase:
        if self.label == "estafa" and self.modalidad is None:
            raise ValueError(f"{self.id}: un caso `estafa` necesita `modalidad` (o no_determinada)")
        if self.split == "test" and self.fuente in {"sintetico", "traducido"}:
            raise ValueError(f"{self.id}: mensajes sintéticos o traducidos nunca van a test (§4.5)")
        return self

    def to_raw_input(self) -> RawInput:
        return RawInput(
            texto=self.texto, imagen_path=self.imagen, canal=self.canal, situacion=self.situacion
        )


# ---------------------------------------------------------------------------
# Interfaces: cada módulo es intercambiable por config.
# ---------------------------------------------------------------------------


@runtime_checkable
class Extractor(Protocol):
    def extract(self, raw: RawInput) -> ExtractedMessage: ...


@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, m: ExtractedMessage) -> RetrievalBundle: ...


@runtime_checkable
class Reasoner(Protocol):
    def reason(
        self, m: ExtractedMessage, b: RetrievalBundle, s: Situacion | None, meta: RunMeta
    ) -> SafeFishOutput: ...


@runtime_checkable
class Verifier(Protocol):
    def verify(
        self, m: ExtractedMessage, b: RetrievalBundle, out: SafeFishOutput
    ) -> SafeFishOutput: ...
