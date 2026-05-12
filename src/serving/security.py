from __future__ import annotations

from dataclasses import dataclass, field
import re


@dataclass(frozen=True)
class SecurityGuardConfig:
    max_query_length: int = 180
    max_payload_field_length: int = 6000
    max_payload_fields: int = 80


@dataclass(frozen=True)
class TextDecision:
    allowed: bool
    status_code: int
    reason: str
    sanitized_value: str = ""
    redactions: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class FormDecision:
    allowed: bool
    status_code: int
    reason: str
    fields: dict[str, str] = field(default_factory=dict)
    redactions: tuple[str, ...] = field(default_factory=tuple)


class SecurityGuard:
    _EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    _PHONE_RE = re.compile(r"\b\+?\d[\d\-() ]{8,}\d\b")
    _CARD_RE = re.compile(r"\b(?:\d[ -]*){13,19}\b")

    def __init__(self, config: SecurityGuardConfig | None = None) -> None:
        self._config = config or SecurityGuardConfig()
        self._blocked_patterns: tuple[tuple[str, re.Pattern[str]], ...] = (
            ("script_injection", re.compile(r"<\s*script", re.IGNORECASE)),
            ("sql_injection", re.compile(r"\b(select|union|drop|insert|delete|update)\b.+\b(from|table|into|where)\b", re.IGNORECASE)),
            ("path_traversal", re.compile(r"\.\./|\.\.\\")),
            ("null_byte", re.compile(r"\x00")),
            ("prompt_injection", re.compile(r"ignore\s+previous\s+instructions|system\s+prompt", re.IGNORECASE)),
        )

    def validate_query_text(self, raw: str) -> TextDecision:
        return self._validate_text(raw, max_length=self._config.max_query_length)

    def validate_form(self, form_fields: dict[str, str]) -> FormDecision:
        if len(form_fields) > self._config.max_payload_fields:
            return FormDecision(allowed=False, status_code=413, reason="too_many_form_fields")

        cleaned: dict[str, str] = {}
        redactions: list[str] = []
        for key, raw_value in form_fields.items():
            value = str(raw_value)
            decision = self._validate_text(value, max_length=self._config.max_payload_field_length)
            if not decision.allowed:
                return FormDecision(allowed=False, status_code=decision.status_code, reason=f"{decision.reason}:{key}")
            cleaned[key] = decision.sanitized_value
            redactions.extend(list(decision.redactions))

        return FormDecision(
            allowed=True,
            status_code=200,
            reason="ok",
            fields=cleaned,
            redactions=tuple(sorted(set(redactions))),
        )

    def _validate_text(self, raw: str, max_length: int) -> TextDecision:
        value = self._strip_controls(raw)
        if not value:
            return TextDecision(allowed=False, status_code=400, reason="empty_input")
        if len(value) > max_length:
            return TextDecision(allowed=False, status_code=413, reason="input_too_large")

        blocked_reason = self._blocked_reason(value)
        if blocked_reason is not None:
            return TextDecision(allowed=False, status_code=400, reason=blocked_reason)

        redacted, redactions = self._redact_pii(value)
        if redactions:
            return TextDecision(
                allowed=False,
                status_code=400,
                reason="pii_detected",
                sanitized_value=redacted,
                redactions=tuple(redactions),
            )

        return TextDecision(allowed=True, status_code=200, reason="ok", sanitized_value=value)

    def _strip_controls(self, raw: str) -> str:
        # Keep only printable characters and compact whitespace for stable matching.
        printable = "".join(ch for ch in str(raw) if ch.isprintable())
        return " ".join(printable.split()).strip()

    def _blocked_reason(self, value: str) -> str | None:
        for reason, pattern in self._blocked_patterns:
            if pattern.search(value):
                return reason
        return None

    def _redact_pii(self, value: str) -> tuple[str, list[str]]:
        redactions: list[str] = []
        updated = value

        if self._EMAIL_RE.search(updated):
            updated = self._EMAIL_RE.sub("[REDACTED_EMAIL]", updated)
            redactions.append("email")
        if self._PHONE_RE.search(updated):
            updated = self._PHONE_RE.sub("[REDACTED_PHONE]", updated)
            redactions.append("phone")
        if self._CARD_RE.search(updated):
            updated = self._CARD_RE.sub("[REDACTED_CARD]", updated)
            redactions.append("card")

        return updated, redactions
