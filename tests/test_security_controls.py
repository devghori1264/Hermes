from src.serving.security import SecurityGuard, SecurityGuardConfig


def test_security_guard_blocks_script_injection() -> None:
    guard = SecurityGuard(SecurityGuardConfig(max_query_length=180))
    decision = guard.validate_query_text("<script>alert('x')</script>")
    assert not decision.allowed
    assert decision.reason == "script_injection"
    assert decision.status_code == 400


def test_security_guard_blocks_pii() -> None:
    guard = SecurityGuard(SecurityGuardConfig(max_query_length=180))
    decision = guard.validate_query_text("contact me at person@example.com")
    assert not decision.allowed
    assert decision.reason == "pii_detected"
    assert "email" in decision.redactions


def test_security_guard_sanitizes_valid_form() -> None:
    guard = SecurityGuard(SecurityGuardConfig(max_payload_field_length=50, max_payload_fields=5))
    decision = guard.validate_form({"title": "  avatar\n", "overview": "space epic"})
    assert decision.allowed
    assert decision.fields["title"] == "avatar"


def test_security_guard_blocks_oversized_field() -> None:
    guard = SecurityGuard(SecurityGuardConfig(max_payload_field_length=10))
    decision = guard.validate_form({"title": "this title is far too long"})
    assert not decision.allowed
    assert decision.status_code == 413
