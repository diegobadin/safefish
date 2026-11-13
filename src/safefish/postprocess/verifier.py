"""Verificador determinístico (M-POST.1, V3, §2.3): reglas R1–R5.

| Regla | Comportamiento |
|---|---|
| R1 cita literal      | La señal se acepta si su cita está en el texto (normalizado, fuzzy ≥ umbral). Si no, se descarta y se registra como alucinación. En capturas, la cita aceptada se alinea a una bbox OCR |
| R2 soporte           | La modalidad debe coincidir con la de ≥ 1 caso del contexto; si no, `no_determinada` (*negative rejection*). Los `casos_similares` deben ser ids de la evidencia |
| R3 estafa confirmada | Requiere modalidad respaldada + ≥ 2 señales verificadas; si no, baja a `muy_sospechoso` |
| R4 asimetría         | `parece_legitimo` es incompatible con ≥ 1 señal crítica verificada: sube a `levemente_sospechoso` |
| R5 plan sin inventos | Teléfonos, URLs y mails del plan que no están en la guía fuente se reemplazan; un paso que nombra un organismo ausente de la guía se elimina |

El orden importa: R1 → R2 → R3 → R4 (análisis) y R5 (plan). Con planner, el pipeline corre el
análisis antes de planificar (la modalidad final decide qué guías se usan) y R5 después.
Todas las reglas son idempotentes. Cada intervención queda registrada como evento.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

from safefish.extract.align import align_cita
from safefish.schemas import (
    ExtractedMessage,
    Modalidad,
    PasoPlan,
    RetrievalBundle,
    SafeFishOutput,
    SenalDetectada,
    SignalId,
    Veredicto,
)
from safefish.taxonomia import senales_criticas
from safefish.textutils import (
    cita_score,
    contacto_respaldado,
    find_contacts,
    find_organismos,
    normalize_for_match,
)

CONTACTO_ELIMINADO = "[contacto eliminado: usá el canal oficial que ya conozcas]"
REGLAS = ("R1", "R2", "R3", "R4", "R5")


@dataclass
class Evento:
    regla: str
    accion: str
    detalle: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _doc_id(it) -> str:
    return str(it.metadata.get("doc_id", it.id))


class RuleVerifier:
    def __init__(
        self,
        rules: Iterable[str] = REGLAS,
        cita_threshold: float = 0.9,
        bbox_threshold: float = 0.8,
    ):
        self.rules = set(rules)
        if unknown := self.rules - set(REGLAS):
            raise ValueError(f"Reglas desconocidas: {sorted(unknown)}")
        self.cita_threshold, self.bbox_threshold = cita_threshold, bbox_threshold

    # -- Protocol Verifier -------------------------------------------------------------
    def verify(self, m: ExtractedMessage, b: RetrievalBundle, out: SafeFishOutput) -> SafeFishOutput:  # fmt: skip
        out, _ = self.check_analysis(m, b, out)
        out, _ = self.check_plan(b, out)
        return out

    # -- R1–R4 ---------------------------------------------------------------------------
    def check_analysis(
        self, m: ExtractedMessage, b: RetrievalBundle, out: SafeFishOutput
    ) -> tuple[SafeFishOutput, list[Evento]]:
        ev: list[Evento] = []
        senales = list(out.senales)
        modalidad, veredicto = out.modalidad, out.veredicto
        casos_similares = list(out.casos_similares)

        if "R1" in self.rules:
            senales = self._r1(m, senales, ev)

        if "R2" in self.rules:
            soportadas = {str(c.metadata.get("modalidad")) for c in b.casos}
            if modalidad is not Modalidad.NO_DETERMINADA and modalidad.value not in soportadas:
                ev.append(Evento("R2", "modalidad_sin_respaldo", {"modalidad": modalidad.value, "modalidades_contexto": sorted(soportadas)}))  # fmt: skip
                modalidad = Modalidad.NO_DETERMINADA
            ids = {_doc_id(it) for it in [*b.casos, *b.mensajes]}
            validos = [c for c in casos_similares if c.case_id in ids]
            for c in casos_similares:
                if c.case_id not in ids:
                    ev.append(Evento("R2", "caso_similar_inexistente", {"case_id": c.case_id}))
            casos_similares = validos

        verificadas = [s for s in senales if s.verificada or "R1" not in self.rules]
        if "R3" in self.rules and veredicto is Veredicto.ESTAFA_CONFIRMADA:
            sin_modalidad = modalidad is Modalidad.NO_DETERMINADA
            if sin_modalidad or len(verificadas) < 2:
                ev.append(Evento("R3", "baja_a_muy_sospechoso", {"modalidad_respaldada": not sin_modalidad, "senales_verificadas": len(verificadas)}))  # fmt: skip
                veredicto = Veredicto.MUY_SOSPECHOSO

        if "R4" in self.rules and veredicto is Veredicto.PARECE_LEGITIMO:
            criticas = self._criticas(verificadas)
            if criticas:
                ev.append(Evento("R4", "sube_a_levemente_sospechoso", {"senales_criticas": sorted(criticas)}))  # fmt: skip
                veredicto = Veredicto.LEVEMENTE_SOSPECHOSO

        out = out.model_copy(
            update={
                "senales": senales,
                "modalidad": modalidad,
                "veredicto": veredicto,
                "casos_similares": casos_similares,
            }
        )
        return out, ev

    def _r1(self, m: ExtractedMessage, senales: list[SenalDetectada], ev: list[Evento]) -> list[SenalDetectada]:  # fmt: skip
        kept: list[SenalDetectada] = []
        seen: set[tuple[str, str]] = set()
        for s in senales:
            key = (s.signal_id.value, normalize_for_match(s.cita))
            if key in seen:
                continue
            seen.add(key)
            score = cita_score(s.cita, m.texto)
            if score < self.cita_threshold:
                ev.append(Evento("R1", "cita_no_literal", {"signal_id": s.signal_id.value, "cita": s.cita[:200], "score": round(score, 3)}))  # fmt: skip
                continue
            bbox = s.bbox
            if m.ocr_words and bbox is None:
                bbox = align_cita(s.cita, m.ocr_words, self.bbox_threshold)
            kept.append(s.model_copy(update={"verificada": True, "bbox": bbox}))
        return kept

    @staticmethod
    def _criticas(senales: list[SenalDetectada]) -> set[str]:
        """Señales críticas verificadas. La transferencia a un tercero cuenta como crítica
        solo con urgencia (§2.3: "transferencia a tercero con urgencia"): pasar un alias para
        un regalo compartido no lo es."""
        ids = {s.signal_id for s in senales}
        criticas = {s.value for s in ids & senales_criticas()}
        if SignalId.TRANSFERENCIA_A_TERCERO.value in criticas and SignalId.URGENCIA_TEMPORAL not in ids:  # fmt: skip
            criticas.discard(SignalId.TRANSFERENCIA_A_TERCERO.value)
        return criticas

    # -- R5 ------------------------------------------------------------------------------
    def check_plan(self, b: RetrievalBundle, out: SafeFishOutput) -> tuple[SafeFishOutput, list[Evento]]:  # fmt: skip
        if "R5" not in self.rules or not out.plan:
            return out, []
        ev: list[Evento] = []
        guias = {_doc_id(g): g.texto for g in b.guias}
        todas = "\n".join(guias.values())
        plan: list[PasoPlan] = []
        for i, p in enumerate(out.plan):
            guide_id = p.guide_id
            if guide_id is not None and guide_id not in guias:
                ev.append(Evento("R5", "guide_id_inexistente", {"paso": i, "guide_id": guide_id}))
                guide_id = None
            fuente = guias[guide_id] if guide_id else todas

            inventados = find_organismos(p.paso) - find_organismos(fuente)
            if inventados:
                ev.append(Evento("R5", "paso_con_organismo_inventado", {"paso": i, "organismos": sorted(inventados), "texto": p.paso[:200]}))  # fmt: skip
                continue

            texto = p.paso
            contactos = find_contacts(texto)
            for c in sorted(contactos, key=len, reverse=True):
                if c in texto and not contacto_respaldado(c, fuente):
                    ev.append(Evento("R5", "contacto_inventado", {"paso": i, "contacto": c}))
                    texto = texto.replace(c, CONTACTO_ELIMINADO)
            plan.append(PasoPlan(paso=texto, guide_id=guide_id))

        if not plan and b.guias:  # nunca dejar al usuario sin plan si hay guía: pasos literales
            g = b.guias[0]
            pasos = _pasos_de_guia(g)
            plan = [PasoPlan(paso=x, guide_id=_doc_id(g)) for x in pasos[: max(1, len(out.plan))]]
            ev.append(Evento("R5", "plan_reemplazado_por_guia", {"guide_id": _doc_id(g)}))
        return out.model_copy(update={"plan": plan}), ev


def _pasos_de_guia(g) -> list[str]:
    import json

    if g.metadata.get("pasos_json"):
        return json.loads(g.metadata["pasos_json"])
    return [g.texto]
