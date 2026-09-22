"""PII detection and sanitization module using Microsoft Presidio.

Protects intentional template placeholders (e.g. {{Order Number}}) from over-redaction
while redacting real personally identifiable information (emails, phones, credit cards).
"""

import os
import re

# Silence harmless advisory notice when importing Presidio without PyTorch
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from supportiq.core.logger import get_logger

logger = get_logger(__name__)

SLOT_PATTERN = re.compile(r"\{\{[^}]+\}\}")


class PIISanitizer:
    """Detects and redacts PII using Microsoft Presidio with slot preservation."""

    def __init__(self, entities: list[str] | None = None) -> None:
        """Initialize Presidio analyzer using the lightweight local spaCy model."""
        self.entities = entities or ["EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD"]

        # Configure Presidio to use local en_core_web_sm (avoids heavy 400MB model download)
        nlp_config = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }
        provider = NlpEngineProvider(nlp_configuration=nlp_config)
        self.analyzer = AnalyzerEngine(nlp_engine=provider.create_engine())
        self.anonymizer = AnonymizerEngine()

        self.operators = {
            "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),
            "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<PHONE>"}),
            "CREDIT_CARD": OperatorConfig("replace", {"new_value": "<CARD>"}),
        }

    def detect_pii(self, text: str) -> list[str]:
        """Return list of detected PII entity types in the given text."""
        if not text or not text.strip():
            return []

        # Temporarily replace {{slot}} variables so Presidio doesn't misidentify them
        slots = SLOT_PATTERN.findall(text)
        slot_map = {f"__SLOT_{i}__": s for i, s in enumerate(slots)}
        protected = text
        for token, s in slot_map.items():
            protected = protected.replace(s, token)

        results = self.analyzer.analyze(
            text=protected,
            entities=self.entities,
            language="en",
        )
        return [r.entity_type for r in results]

    def sanitize(self, text: str) -> str:
        """Sanitize text by masking PII while preserving template slots."""
        if not text or not text.strip():
            return text

        # Step 1: Temporarily replace {{slot}} placeholders with safe tokens
        slots = SLOT_PATTERN.findall(text)
        slot_map = {f"__SLOT_{i}__": s for i, s in enumerate(slots)}
        protected = text
        for token, s in slot_map.items():
            protected = protected.replace(s, token)

        # Step 2: Analyze text for PII entities
        results = self.analyzer.analyze(
            text=protected,
            entities=self.entities,
            language="en",
        )

        if not results:
            return text

        # Step 3: Anonymize detected entities
        anonymized = self.anonymizer.anonymize(
            text=protected,
            analyzer_results=results,
            operators=self.operators,
        )

        # Step 4: Restore original {{slot}} placeholders
        restored = anonymized.text
        for token, s in slot_map.items():
            restored = restored.replace(token, s)

        return restored


# Default singleton instance for easy import and reuse
_default_sanitizer: PIISanitizer | None = None


def get_pii_sanitizer() -> PIISanitizer:
    """Get or create the singleton PIISanitizer instance."""
    global _default_sanitizer
    if _default_sanitizer is None:
        _default_sanitizer = PIISanitizer()
    return _default_sanitizer


def sanitize_text(text: str) -> str:
    """Convenience helper to sanitize a single text string."""
    return get_pii_sanitizer().sanitize(text)
