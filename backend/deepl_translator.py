"""
DeepL translation module featuring resilient caching, quota telemetry,
and automatic string fallbacks. Translation is an optional external service;
callers must disclose that report text leaves the local device when enabled.
"""
import os
import json
import hashlib
import logging
import tempfile
from datetime import datetime
from typing import Dict, Iterable, List, Optional

import deepl

from .pii_protection import ExternalTranslationRedactor

logger = logging.getLogger(__name__)


class QuotaManager:
    """Track and enforce DeepL API quota limitations safely."""

    def __init__(self, quota_file: str = "logs/deepl_quota.json"):
        self.quota_file = quota_file
        self.monthly_limit = 500000  # DeepL API Free tier threshold (500K characters)
        self.month = datetime.now().strftime("%Y-%m")
        self.characters_used = 0
        self.load_quota()

    def load_quota(self):
        """Load character tracking metadata securely from local disk."""
        try:
            with open(self.quota_file, 'r') as f:
                data = json.load(f)
                self.month = data.get('month', datetime.now().strftime("%Y-%m"))
                self.characters_used = data.get('characters_used', 0)
                logger.info(f"✓ DeepL Quota loaded: {self.characters_used}/{self.monthly_limit} characters used.")
        except FileNotFoundError:
            self.month = datetime.now().strftime("%Y-%m")
            self.characters_used = 0
            self.save_quota()
        except Exception as e:
            logger.error(f"Failed to read quota tracking log: {e}. Resetting values.")
            self.characters_used = 0

    def save_quota(self):
        """Persist absolute usage state variables to the active log partition."""
        try:
            os.makedirs(os.path.dirname(self.quota_file), exist_ok=True)
            with open(self.quota_file, 'w') as f:
                json.dump({
                    'month': self.month,
                    'characters_used': self.characters_used,
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save current state variables to quota log: {e}")

    def reset_if_new_month(self):
        """Check for structural multi-tenant calendar resets at execution runtime."""
        current_month = datetime.now().strftime("%Y-%m")
        if current_month != self.month:
            logger.info(f"New calendar month detected ({current_month}). Resetting usage metrics.")
            self.month = current_month
            self.characters_used = 0
            self.save_quota()

    def add_usage(self, characters: int) -> float:
        """Record and validate current usage benchmarks."""
        self.reset_if_new_month()
        self.characters_used += characters
        self.save_quota()

        percentage = (self.characters_used / self.monthly_limit) * 100
        logger.info(f"DeepL Quota Status: {self.characters_used}/{self.monthly_limit} ({percentage:.2f}%)")

        if percentage > 90:
            logger.warning(f"⚠️ CRITICAL SYSTEM NOTICE: {percentage:.1f}% of DeepL monthly allocation spent.")

        return percentage

    def get_remaining(self) -> int:
        """Calculate total non-allocated transaction bandwidth."""
        self.reset_if_new_month()
        return max(0, self.monthly_limit - self.characters_used)

    def can_translate(self, text_or_len) -> bool:
        """Verify processing capacity before submitting requests downstream."""
        char_count = text_or_len if isinstance(text_or_len, int) else len(str(text_or_len or ""))
        return char_count <= self.get_remaining()

    def get_status(self) -> Dict:
        """Compile a structural telemetry status dictionary."""
        self.reset_if_new_month()
        remaining = self.get_remaining()
        percentage = (self.characters_used / self.monthly_limit) * 100

        return {
            'month': self.month,
            'used': self.characters_used,
            'limit': self.monthly_limit,
            'remaining': remaining,
            'percentage_used': percentage
        }


class CacheManager:
    """Bounded translation cache with optional safe JSON persistence.

    Only callers that explicitly mark static text as cacheable use this cache.
    PII-bearing report text is never cached. Deployments that explicitly opt
    in to persistence may set ``TRANSLATION_CACHE_PATH``; entries are JSON,
    keyed by a digest rather than raw source text, and capped to limit
    retention.
    """

    def __init__(self, cache_file: Optional[str] = None, max_entries: int = 500):
        self.cache_file = cache_file if cache_file is not None else os.getenv("TRANSLATION_CACHE_PATH")
        self.max_entries = max(1, max_entries)
        self.memory_cache = {}
        if self.cache_file:
            self.load_persistent_cache()

    def load_persistent_cache(self):
        """Load a validated JSON cache without deserialising executable data."""
        if not self.cache_file:
            return
        try:
            with open(self.cache_file, "r", encoding="utf-8") as cache_handle:
                payload = json.load(cache_handle)
            if not isinstance(payload, dict) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in payload.items()
            ):
                raise ValueError("cache schema is invalid")
            self.memory_cache = dict(list(payload.items())[-self.max_entries:])
            logger.info("Loaded %d persisted translation cache entries", len(self.memory_cache))
        except FileNotFoundError:
            logger.info("No persisted translation cache found; starting with an empty cache")
            self.memory_cache = {}
        except Exception:
            logger.warning("Could not load persisted translation cache; starting with an empty cache")
            self.memory_cache = {}

    def save_persistent_cache(self):
        """Atomically persist the optional cache as restrictive JSON."""
        if not self.cache_file:
            return
        cache_dir = os.path.dirname(os.path.abspath(self.cache_file))
        try:
            os.makedirs(cache_dir, mode=0o700, exist_ok=True)
            file_descriptor, temporary_path = tempfile.mkstemp(
                prefix=".translation-cache-", suffix=".json", dir=cache_dir
            )
            try:
                with os.fdopen(file_descriptor, "w", encoding="utf-8") as cache_handle:
                    json.dump(self.memory_cache, cache_handle, ensure_ascii=False, separators=(",", ":"))
                os.chmod(temporary_path, 0o600)
                os.replace(temporary_path, self.cache_file)
            finally:
                if os.path.exists(temporary_path):
                    os.unlink(temporary_path)
        except Exception:
            logger.warning("Failed to persist the translation cache")

    def get_cache_key(self, text: str, language: str) -> str:
        """Create a non-reversible lookup key without retaining source text."""
        payload = f"{language.casefold()}\x00{text.strip()}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def get(self, text: str, language: str) -> Optional[str]:
        """Query state tables for existing processed text targets."""
        if not text or language == 'en':
            return text

        key = self.get_cache_key(text, language)
        if key in self.memory_cache:
            logger.debug("Translation cache hit")
            return self.memory_cache[key]
        return None

    def set(self, text: str, language: str, translation: str):
        """Update active translation tables and save state to disk."""
        if not text or not translation:
            return
        key = self.get_cache_key(text, language)
        self.memory_cache[key] = translation
        while len(self.memory_cache) > self.max_entries:
            self.memory_cache.pop(next(iter(self.memory_cache)))
        self.save_persistent_cache()

    def clear(self):
        """Purge storage files and clear memory space entirely."""
        self.memory_cache = {}
        if self.cache_file and os.path.exists(self.cache_file):
            try:
                os.remove(self.cache_file)
            except Exception:
                logger.warning("Could not delete the persisted translation cache")
        logger.info("Local translation cache cleared successfully.")


class DeepLTranslator:
    """DeepL core driver incorporating caching rules and safe string fallbacks."""

    LANGUAGES = {
        'en': 'EN',  # English
        'id': 'ID',  # Indonesian
        'zh': 'ZH',  # Chinese (DeepL's supported generic target)
    }

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("DEEPL_API_KEY")
        self.quota = QuotaManager()
        self.cache = CacheManager()
        self.client = None

        if not self.api_key:
            logger.error("❌ DEEPL_API_KEY environment variable missing. Running in original text fallback mode.")
            return

        try:
            self.client = deepl.Translator(self.api_key)
            logger.info("✓ DeepL core engine running.")
        except Exception as e:
            logger.error(f"Failed to construct explicit DeepL engine driver: {e}")
            self.client = None

    @staticmethod
    def _restore_external_translation(
        source_text: str,
        redactor: ExternalTranslationRedactor,
        translated_text: str,
    ) -> str:
        """Restore local placeholders without accepting a damaged response."""
        restored = redactor.restore(translated_text)
        if restored is None:
            logger.warning(
                "Translation response did not preserve privacy placeholders; using the source text instead."
            )
            return source_text
        return restored

    def translate_text(
        self,
        text: str,
        target_language: str,
        *,
        protected_values: Optional[Iterable[object]] = None,
        cacheable: bool = False,
    ) -> str:
        """
        Translate input text safely. Returns the original English source string 
        if network issues occur, accounts are suspended, or monthly quotas are hit.
        """
        source_text = str(text or "")
        if not source_text or target_language == 'en':
            return source_text

        # Do not put known learner identifiers or common identifiers in the
        # request or cache. The placeholders are restored only in memory.
        redactor = ExternalTranslationRedactor(protected_values)
        safe_text = redactor.protect(source_text)
        # Only known static UI labels opt in to caching. Report content is
        # intentionally not cached, even in memory, when it contains a
        # placeholder-protected identifier.
        cache_allowed = cacheable and not redactor.has_placeholders

        # 1. Evaluate cache instantly (Zero character processing footprint)
        if cache_allowed:
            cached = self.cache.get(safe_text, target_language)
            if cached:
                return self._restore_external_translation(source_text, redactor, cached)

        # 2. Resilient fallback check if engine client failed initializing
        if not self.client:
            logger.warning("DeepL pipeline offline. Defaulting to original text representation.")
            return source_text

        # 3. Prevent structural pipeline errors if target string breaks monthly allocation
        if not self.quota.can_translate(safe_text):
            logger.error(f"⚠️ Translation denied. Target payload length ({len(safe_text)} chars) exceeds remaining quota.")
            return source_text

        try:
            target_lang_code = self.LANGUAGES.get(target_language, 'EN')

            result = self.client.translate_text(
                safe_text,
                target_lang=target_lang_code,
                source_lang='EN'
            )

            translation = result.text

            # Commit tracking benchmarks and append key values to cache indices
            self.quota.add_usage(len(safe_text))
            if cache_allowed:
                self.cache.set(safe_text, target_language, translation)

            logger.info("Translated text to %s", target_language)
            return self._restore_external_translation(source_text, redactor, translation)

        except Exception:
            # Provider errors can include request fragments. Do not copy those
            # into logs for a report that may contain student information.
            logger.error("DeepL processing failed; falling back to the original copy.")
            return source_text

    def batch_translate(
        self,
        texts: List[str],
        target_language: str,
        *,
        protected_values: Optional[Iterable[object]] = None,
        cacheable: bool = False,
    ) -> List[str]:
        """
        Processes lists of structural labels efficiently by pulling cached items 
        first and grouping any remaining strings into a single translation payload.
        """
        if not texts or target_language == 'en':
            return texts

        source_texts = [str(item or "") for item in texts]
        if isinstance(protected_values, (str, bytes)):
            protected_values = (protected_values,)
        else:
            protected_values = tuple(protected_values or ())
        redactors = [ExternalTranslationRedactor(protected_values) for _ in source_texts]
        safe_texts = [
            redactor.protect(source_text)
            for source_text, redactor in zip(source_texts, redactors)
        ]
        results: List[Optional[str]] = [None] * len(source_texts)
        uncached_payloads = []
        uncached_indices = []

        # Pull layout identifiers already stored inside local dictionary caches
        for idx, safe_text in enumerate(safe_texts):
            cache_allowed = cacheable and not redactors[idx].has_placeholders
            cached = self.cache.get(safe_text, target_language) if cache_allowed else None
            if cached:
                results[idx] = self._restore_external_translation(
                    source_texts[idx], redactors[idx], cached
                )
            else:
                uncached_payloads.append(safe_text)
                uncached_indices.append(idx)

        if not uncached_payloads:
            return [
                value if value is not None else source_texts[index]
                for index, value in enumerate(results)
            ]

        # Run collective batch size validation checks against system usage meters
        total_batch_chars = sum(len(str(x)) for x in uncached_payloads)

        if not self.client or not self.quota.can_translate(total_batch_chars):
            logger.warning(
                "System quota limits hit or client offline. Filling batch array with default string constants.")
            for remaining_idx in uncached_indices:
                results[remaining_idx] = source_texts[remaining_idx]
            return [
                value if value is not None else source_texts[index]
                for index, value in enumerate(results)
            ]

        try:
            target_lang_code = self.LANGUAGES.get(target_language, 'EN')

            batch_outputs = self.client.translate_text(
                uncached_payloads,
                target_lang=target_lang_code,
                source_lang='EN'
            )

            self.quota.add_usage(total_batch_chars)

            for index_pointer, API_response in enumerate(batch_outputs):
                actual_translated_text = API_response.text
                target_root_index = uncached_indices[index_pointer]

                safe_source_text = uncached_payloads[index_pointer]
                if cacheable and not redactors[target_root_index].has_placeholders:
                    self.cache.set(safe_source_text, target_language, actual_translated_text)
                results[target_root_index] = self._restore_external_translation(
                    source_texts[target_root_index],
                    redactors[target_root_index],
                    actual_translated_text,
                )

            logger.info(f"✓ Batch completed processing: {len(uncached_payloads)} identifiers converted.")
            return [
                value if value is not None else source_texts[index]
                for index, value in enumerate(results)
            ]

        except Exception:
            logger.error("DeepL batch processing failed; using original text.")
            for fallback_idx in uncached_indices:
                results[fallback_idx] = source_texts[fallback_idx]
            return [
                value if value is not None else source_texts[index]
                for index, value in enumerate(results)
            ]

    def get_quota_status(self) -> Dict:
        """Expose quota status dictionary."""
        return self.quota.get_status()

    def clear_cache(self):
        """Purge system caches."""
        self.cache.clear()


# Static translation configurations for application UI strings
UI_STRINGS = {
    'title': 'Student Learning Journey Summarization System',
    'subtitle': 'Transform Raw Data into Actionable Insights',
    'upload_section': 'Upload Student Data',
    'upload_help': 'Supports CSV, XLSX, PDF, PNG/JPG files (maximum 32 MB by default)',
    'process_button': 'Process & Analyze',
    'processing': 'Processing your data...',
    'download_report': 'Download Report as PDF',
    'language_selector': 'Select Language',
    'analytics_summary': 'Analytics Summary',
    'student_performance': 'Student Performance',
    'average_score': 'Average Score',
    'highest_subject': 'Highest Subject',
    'lowest_subject': 'Lowest Subject',
    'success_message': 'Analysis complete! Download your report below.',
    'error_message': 'Error processing file. Please try again.',
    'no_file': 'Please upload a file to get started.',
    'quota_warning': 'Translation quota running low!',
    'quota_exceeded': 'Monthly translation quota exceeded. Showing original content.',
    'updating_summary': '🔄 Updating summary for your selected axes...',
    'regenerate_ai_summary': '↻ Regenerate AI Summary',
    'llm_service_warning': '⚠️ Sorry, our local AI summary service is temporarily unavailable',
    'llm_service_warning_id': '⚠️ Maaf, layanan ringkasan AI lokal kami sementara tidak tersedia',
    'llm_service_warning_zh': '⚠️ 抱歉，我们的本地AI摘要服务暂时无法使用',
}


class DeepLUITranslator:
    """Manage and resolve UI application labels using memory-cached dictionaries."""

    def __init__(self):
        self.translator = translator
        self.ui_cache = {}

    def get_string(self, key: str, language: str = 'en') -> str:
        """Resolve localized runtime display tokens for structural UI fields."""
        if language == 'en':
            return UI_STRINGS.get(key, key)

        cache_key = f"{key}_{language}"
        if cache_key in self.ui_cache:
            return self.ui_cache[cache_key]

        original_text = UI_STRINGS.get(key, key)
        # DeepLTranslator handles errors internally, guaranteeing a string fallback value
        translated_text = self.translator.translate_text(
            original_text,
            language,
            cacheable=True,
        )

        self.ui_cache[cache_key] = translated_text
        return translated_text

    def get_all_strings(self, language: str = 'en') -> Dict[str, str]:
        """Compile a fully translated dashboard label schema map."""
        if language == 'en':
            return UI_STRINGS.copy()

        all_cached = all(f"{k}_{language}" in self.ui_cache for k in UI_STRINGS.keys())
        if all_cached:
            return {k: self.ui_cache[f"{k}_{language}"] for k in UI_STRINGS.keys()}

        uncached_keys = [k for k in UI_STRINGS.keys() if f"{k}_{language}" not in self.ui_cache]
        uncached_values = [UI_STRINGS[k] for k in uncached_keys]

        translated_values = self.translator.batch_translate(
            uncached_values,
            language,
            cacheable=True,
        )

        for k, structural_translation in zip(uncached_keys, translated_values):
            self.ui_cache[f"{k}_{language}"] = structural_translation

        return {k: self.ui_cache.get(f"{k}_{language}", UI_STRINGS.get(k, k)) for k in UI_STRINGS.keys()}

    def show_quota_warning_if_needed(self) -> Optional[str]:
        """Monitor allocation health metrics and emit standard system alerts."""
        try:
            status = self.translator.get_quota_status()
            used_pct = status.get('percentage_used', 0)
            if used_pct > 80:
                return f"⚠️ System Warning: {used_pct:.1f}% of translation limit consumed. ({status.get('remaining', 0)} characters remain)"
        except Exception:
            pass
        return None


# Instantiate unified global interface elements safely across multi-tenant imports
translator = DeepLTranslator()
ui_translator = DeepLUITranslator()
