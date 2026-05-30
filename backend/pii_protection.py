import re
import hashlib
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class PIIProtector:
    """Remove/mask PII before external API calls"""

    # Refined patterns for resilient PII detection
    PATTERNS = {
        'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        'phone': r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        'ic_number': r'\d{6}-\d{2}-\d{4}',  # Malaysian MyKad Identity Card
        'student_id': r'\b\d{7,8}\b',  # Word boundaries ensure scores/metrics aren't eaten
    }

    def __init__(self, salt: str = None):
        # Use a localized salt configuration to prevent lookup dictionary attacks on hashes
        self.salt = salt or "DefaultSystemSecureSaltKey_2026!"
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
            logger.info(f"Student metadata tracking randomized securely to ID token.")

        # Clean potential trace structures
        for private_key in ['email', 'phone', 'ic_number', 'ic']:
            anonymous.pop(private_key, None)

        return anonymous

    def create_safe_prompt(self, stats_dict: Dict, student_context: Dict) -> Tuple[str, Dict]:
        """
        Create structured Gemini analytics prompt completely cleared of PII markers.
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