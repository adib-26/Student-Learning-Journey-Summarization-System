"""
DeepL translation module featuring resilient caching, quota telemetry, 
and automatic string fallbacks.
"""
import io
import os
import json
import pickle
import logging
from datetime import datetime
from typing import Dict, List, Optional

import deepl

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
    """Persistent compilation layer preventing identical string reprocessing costs."""

    def __init__(self, cache_file: str = "logs/translation_cache.pkl"):
        self.cache_file = cache_file
        self.memory_cache = {}
        self.load_persistent_cache()

    def load_persistent_cache(self):
        """Hydrate runtime binary storage tables from local disk blocks."""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'rb') as f:
                self.memory_cache = pickle.load(f)
                logger.info(f"✓ Cached translation schema online: {len(self.memory_cache)} elements loaded.")
        except FileNotFoundError:
            logger.info("No prior serialization tables detected. Starting clean session matrix.")
            self.memory_cache = {}
        except Exception as e:
            logger.error(f"Error restoring serialization map: {e}. Flushing cache mapping.")
            self.memory_cache = {}

    def save_persistent_cache(self):
        """Serialize current in-memory translation indices back to disk files."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.memory_cache, f)
        except Exception as e:
            logger.error(f"Failed to checkpoint serialization layer to disk: {e}")

    def get_cache_key(self, text: str, language: str) -> str:
        """Construct normalized index signatures for lookup keys."""
        return f"{language}:{text.strip()}"

    def get(self, text: str, language: str) -> Optional[str]:
        """Query state tables for existing processed text targets."""
        if not text or language == 'en':
            return text

        key = self.get_cache_key(text, language)
        if key in self.memory_cache:
            logger.debug(f"✓ Cache hit matching pattern: '{text[:25]}...'")
            return self.memory_cache[key]
        return None

    def set(self, text: str, language: str, translation: str):
        """Update active translation tables and save state to disk."""
        if not text or not translation:
            return
        key = self.get_cache_key(text, language)
        self.memory_cache[key] = translation
        self.save_persistent_cache()

    def clear(self):
        """Purge storage files and clear memory space entirely."""
        self.memory_cache = {}
        if os.path.exists(self.cache_file):
            try:
                os.remove(self.cache_file)
            except Exception as e:
                logger.error(f"Could not purge physical disk storage map: {e}")
        logger.info("Local translation cache cleared successfully.")


class DeepLTranslator:
    """DeepL core driver incorporating caching rules and safe string fallbacks."""

    LANGUAGES = {
        'en': 'EN',  # English
        'ms': 'ID',  # Malay (Using Indonesian mapping as closest functional tier)
        'zh': 'ZH',  # Chinese (Simplified)
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

    def translate_text(self, text: str, target_language: str) -> str:
        """
        Translate input text safely. Returns the original English source string 
        if network issues occur, accounts are suspended, or monthly quotas are hit.
        """
        if not text or target_language == 'en':
            return text

        # 1. Evaluate cache instantly (Zero character processing footprint)
        cached = self.cache.get(text, target_language)
        if cached:
            return cached

        # 2. Resilient fallback check if engine client failed initializing
        if not self.client:
            logger.warning("DeepL pipeline offline. Defaulting to original text representation.")
            return text

        # 3. Prevent structural pipeline errors if target string breaks monthly allocation
        if not self.quota.can_translate(text):
            logger.error(f"⚠️ Translation denied. Target payload length ({len(text)} chars) exceeds remaining quota.")
            return text

        try:
            target_lang_code = self.LANGUAGES.get(target_language, 'EN')

            result = self.client.translate_text(
                text,
                target_lang=target_lang_code,
                source_lang='EN'
            )

            translation = result.text

            # Commit tracking benchmarks and append key values to cache indices
            self.quota.add_usage(len(text))
            self.cache.set(text, target_language, translation)

            logger.info(f"✓ Parsed text slice successfully: '{text[:20]}...' -> {target_language}")
            return translation

        except Exception as e:
            logger.error(f"DeepL processing exception hit: {e}. Falling back to original copy.")
            return text

    def batch_translate(self, texts: List[str], target_language: str) -> List[str]:
        """
        Processes lists of structural labels efficiently by pulling cached items 
        first and grouping any remaining strings into a single translation payload.
        """
        if not texts or target_language == 'en':
            return texts

        results = [None] * len(texts)
        uncached_payloads = []
        uncached_indices = []

        # Pull layout identifiers already stored inside local dictionary caches
        for idx, item in enumerate(texts):
            cached = self.cache.get(item, target_language)
            if cached:
                results[idx] = cached
            else:
                uncached_payloads.append(item)
                uncached_indices.append(idx)

        if not uncached_payloads:
            return results

        # Run collective batch size validation checks against system usage meters
        total_batch_chars = sum(len(str(x)) for x in uncached_payloads)

        if not self.client or not self.quota.can_translate(total_batch_chars):
            logger.warning(
                "System quota limits hit or client offline. Filling batch array with default string constants.")
            for remaining_idx in uncached_indices:
                results[remaining_idx] = texts[remaining_idx]
            return results

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

                results[target_root_index] = actual_translated_text
                self.cache.set(uncached_payloads[index_pointer], target_language, actual_translated_text)

            logger.info(f"✓ Batch completed processing: {len(uncached_payloads)} identifiers converted.")
            return results

        except Exception as e:
            logger.error(f"Batch execution pipeline encountered errors: {e}. Defaulting to layout configurations.")
            for fallback_idx in uncached_indices:
                results[fallback_idx] = texts[fallback_idx]
            return results

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
    'upload_help': 'Supports CSV, XLSX, PDF, PNG/JPG files (max 200MB)',
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
    'gemini_quota_warning': '⚠️ Sorry, our AI summary service is temporarily unavailable',
    'gemini_quota_warning_ms': '⚠️ Maaf, perkhidmatan ringkasan AI kami tidak tersedia buat sementara waktu',
    'gemini_quota_warning_zh': '⚠️ 抱歉，我们的AI摘要服务暂时无法使用',
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
        translated_text = self.translator.translate_text(original_text, language)

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

        translated_values = self.translator.batch_translate(uncached_values, language)

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