"""Unit tests for PII detection and slot preservation."""

from supportiq.data.pii import PIISanitizer, sanitize_text


def test_sanitize_emails_and_phones() -> None:
    """Ensure real emails and phone numbers are redacted."""
    sanitizer = PIISanitizer()
    raw = "Contact me at user@test.com or 555-987-6543."
    sanitized = sanitizer.sanitize(raw)

    assert "user@test.com" not in sanitized
    assert "555-987-6543" not in sanitized
    assert "<EMAIL>" in sanitized
    assert "<PHONE>" in sanitized


def test_sanitize_credit_card() -> None:
    """Ensure credit card numbers (passing Luhn algorithm) are masked."""
    sanitizer = PIISanitizer()
    raw = "My card is 4532-0000-0000-0009."
    sanitized = sanitizer.sanitize(raw)

    assert "4532-0000-0000-0009" not in sanitized
    assert "<CARD>" in sanitized


def test_preserve_template_placeholders() -> None:
    """Ensure template slots like {{Order Number}} and {{Email}} are preserved."""
    sanitizer = PIISanitizer()
    raw = "Please cancel order {{Order Number}} and send receipt to {{Email Address}}."
    sanitized = sanitizer.sanitize(raw)

    # Template variables must remain 100% intact
    assert sanitized == raw
    assert "{{Order Number}}" in sanitized
    assert "{{Email Address}}" in sanitized


def test_mixed_pii_and_slots() -> None:
    """Ensure real PII is redacted while template slots remain untouched in the same text."""
    sanitizer = PIISanitizer()
    raw = "Order {{Order Number}} belonged to john.doe@realmail.com, please update."
    sanitized = sanitizer.sanitize(raw)

    assert "{{Order Number}}" in sanitized
    assert "john.doe@realmail.com" not in sanitized
    assert "<EMAIL>" in sanitized


def test_detect_pii_entities() -> None:
    """Verify entity type detection list."""
    sanitizer = PIISanitizer()
    text = "Send confirmation to support@domain.org."
    entities = sanitizer.detect_pii(text)

    assert "EMAIL_ADDRESS" in entities


def test_empty_and_clean_text() -> None:
    """Verify edge cases: empty strings and clean customer questions."""
    sanitizer = PIISanitizer()
    assert sanitizer.sanitize("") == ""
    assert sanitizer.sanitize("   ") == "   "

    clean_text = "How do I update my shipping address?"
    assert sanitizer.sanitize(clean_text) == clean_text
    assert sanitizer.detect_pii(clean_text) == []


def test_sanitize_text_convenience_function() -> None:
    """Verify module-level convenience function sanitize_text."""
    result = sanitize_text("Call 800-555-0199 for assistance.")
    assert "<PHONE>" in result
    assert "800-555-0199" not in result
