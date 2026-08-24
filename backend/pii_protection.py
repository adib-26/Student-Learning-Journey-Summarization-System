import re
import hashlib
import os
import secrets
from typing import Dict, Iterable, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# A process-local random fallback prevents a predictable, repository-shipped
# hash salt. Set PII_HASH_SALT in a managed secret store only when stable
# pseudonymous identifiers are genuinely required across restarts.
_PROCESS_PII_SALT = os.getenv("PII_HASH_SALT") or secrets.token_urlsafe(32)


class ExternalTranslationRedactor:
    """Temporarily replace identifiers before text leaves the local device.

    The placeholders are deliberately opaque to the translation provider and
    are restored only in local memory after the response returns. This is a
    data-minimisation measure, not a substitute for institutional approval to
    use an external translation service.
    """

    _PATTERNS = (
        ("EMAIL", re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")),
        ("PHONE", re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")),
        ("IC", re.compile(r"\b\d{6}-\d{2}-\d{4}\b")),
        ("STUDENT_ID", re.compile(r"\b\d{7,8}\b")),
    )

    def __init__(self, known_values: Optional[Iterable[object]] = None):
        if isinstance(known_values, (str, bytes)):
            known_values = (known_values,)
        self.known_values = tuple(
            value.strip()
            for value in (str(item or "") for item in (known_values or ()))
            if len(value.strip()) >= 2
        )
        self._token_to_value: Dict[str, str] = {}
        self._value_to_token: Dict[str, str] = {}
        self._token_expected_count: Dict[str, int] = {}
        # Each request uses a fresh nonce, so a source document cannot
        # realistically contain a valid placeholder by accident.
        self._nonce = secrets.token_hex(12).upper()

    def _placeholder_for(self, value: str, category: str) -> str:
        key = value.casefold()
        existing = self._value_to_token.get(key)
        if existing:
            self._token_expected_count[existing] += 1
            return existing

        token = (
            f"[[SLJS_{self._nonce}_{category}_"
            f"{len(self._token_to_value) + 1:03d}_KEEP]]"
        )
        self._value_to_token[key] = token
        self._token_to_value[token] = value
        self._token_expected_count[token] = 1
        return token

    @property
    def has_placeholders(self) -> bool:
        """Whether this request contains values unsuitable for caching."""
        return bool(self._token_to_value)

    def protect(self, text: object) -> str:
        """Return an external-safe representation of ``text``."""
        protected = str(text or "")

        # Preserve complete structured identifiers before replacing a known
        # name. Otherwise a name such as "Aisha" could fragment
        # ``aisha@example.com`` and expose the remaining domain externally.
        for category, pattern in self._PATTERNS:
            protected = pattern.sub(
                lambda match: self._placeholder_for(match.group(0), category),
                protected,
            )

        # Replace explicit values next (for example, the resolved student
        # name), longest-first so a full name wins over an embedded fragment.
        for value in sorted(set(self.known_values), key=len, reverse=True):
            pattern = re.compile(re.escape(value), re.IGNORECASE)
            protected = pattern.sub(
                lambda match: self._placeholder_for(match.group(0), "PERSON"),
                protected,
            )
        return protected

    def restore(self, translated_text: object) -> Optional[str]:
        """Restore placeholders, or fail closed if a provider altered one."""
        restored = str(translated_text or "")
        if any(
            restored.count(token) != self._token_expected_count[token]
            for token in self._token_to_value
        ):
            return None
        for token, value in self._token_to_value.items():
            restored = restored.replace(token, value)
        return restored


class PIIProtector:
    """Remove/mask PII before external API calls"""

    # Refined patterns for resilient PII detection
    PATTERNS = {
        'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        'phone': r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        'ic_number': r'\d{6}-\d{2}-\d{4}',  # Malaysian MyKad Identity Card
        'student_id': r'\b\d{7,8}\b',  # Word boundaries ensure scores/metrics aren't eaten
    }

    def __init__(self, salt: Optional[str] = None):
        # Use a configured or process-local random salt to prevent lookup
        # attacks against predictable pseudonymous identifiers.
        self.salt = salt or _PROCESS_PII_SALT
        self.redacted_map = {}

    def redact_pii(self, text: str) -> str:
        """Replace unstructured text PII seamlessly using regular expressions"""
        if not text:
            return ""

        redacted = text
        for pii_type, pattern in self.PATTERNS.items():
            # Find matching structures
            matches = re.findall(pattern, redacted)
            for match in set(matches):  # Deduplicate to prevent redundant mutations
                placeholder = f"[REDACTED_{pii_type.upper()}]"
                self.redacted_map[match] = placeholder
                logger.warning(f"PII Protected Layer Triggered: {pii_type}")

            # Perform clean bulk substitution
            redacted = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", redacted)

        return redacted

    def anonymize_student_data(self, student_dict: Dict) -> Dict:
        """Convert student context maps to anonymous token configurations safely"""
        anonymous = student_dict.copy()

        # Salted hashing for reliable identification tokens without raw exposure
        if 'student_name' in anonymous and anonymous['student_name']:
            original_name = str(anonymous['student_name']).strip()
            salted_payload = f"{original_name}{self.salt}"
            hashed = hashlib.sha256(salted_payload.encode()).hexdigest()[:12]

            anonymous['student_id'] = f"STU_{hashed}"
            # Pop safely to protect object keys from accidental preservation
            anonymous.pop('student_name', None)
            logger.info("Student metadata tracking randomized securely to ID token.")

        # Clean potential trace structures
        for private_key in ['email', 'phone', 'ic_number', 'ic']:
            anonymous.pop(private_key, None)

        return anonymous

    def create_safe_prompt(self, stats_dict: Dict, student_context: Dict) -> Tuple[str, Dict]:
        """
        Create structured LLM analytics prompt completely cleared of PII markers.
        Returns: (safe_prompt, redaction_log)
        """
        # Clear dictionary metadata structure
        safe_context = self.anonymize_student_data(student_context)

        # Build prompt safely
        safe_prompt = f"""
        Generate a professional narrative student executive summary based on the following:
        - Student ID: {safe_context.get('student_id', 'Anonymous')}
        - Level: {safe_context.get('level', 'Not specified')}
        - Stream: {safe_context.get('stream', 'Not specified')}

        Academic Performance metrics:
        - Average Score: {stats_dict.get('average', 'N/A')}
        - Highest Subject: {stats_dict.get('highest_subject', 'N/A')}
        - Lowest Subject: {stats_dict.get('lowest_subject', 'N/A')}
        - Trend Analysis: {stats_dict.get('trend', 'N/A')}

        Contextual Data:
        {self.redact_pii(stats_dict.get('summary_context', ''))}

        Instructions:
        Do NOT write student names or individual identifiers in the output text.
        Write exactly 6-7 sentences offering balanced and plain English insights.
        """

        return safe_prompt.strip(), self.redacted_map
