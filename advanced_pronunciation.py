# -*- coding: utf-8 -*-
"""Advanced pronunciation analysis utilities.

This module builds on top of ``pronunciationTrainer`` to provide
richer feedback about a speaker's performance. It analyses word
alignment, skipped words and pauses to produce a detailed report.
"""

from __future__ import annotations

import numpy as np
import json

import pronunciationTrainer
import WordMetrics
import WordMatching as wm


class AdvancedPronunciationAnalyzer:
    """Analyse pronunciation using an underlying :class:`PronunciationTrainer`."""

    def __init__(self, language: str = "en") -> None:
        self.trainer = pronunciationTrainer.getTrainer(language)
        self.language = language

    def _compute_confidence(self, expected: str, predicted: str) -> float:
        """Return a pseudo confidence score based on edit distance."""
        if not predicted:
            return 0.0
        distance = WordMetrics.edit_distance_python(expected.lower(), predicted.lower())
        norm = max(len(expected), 1)
        score = max(0.0, 1.0 - distance / norm)
        return float(round(score, 3))

    def _compute_phoneme_match(self, expected: str, predicted: str) -> list[int]:
        """Return a list marking correct (1) or wrong (0) phonemes."""
        expected_ph = self.trainer.ipa_converter.convertToPhonem(expected)
        predicted_ph = self.trainer.ipa_converter.convertToPhonem(predicted)
        mapped_letters, _ = wm.get_best_mapped_words(predicted_ph, expected_ph)
        return wm.getWhichLettersWereTranscribedCorrectly(expected_ph, mapped_letters)

    def analyze(self, audio, expected_text: str) -> dict:
        """Process ``audio`` against ``expected_text`` and return a JSON report."""
        result = self.trainer.processAudioForGivenText(audio, expected_text)

        words_expected = [w for w in expected_text.split()]
        words_predicted = [pair[1] for pair in result["real_and_transcribed_words"]]

        starts = [float(t) for t in result["start_time"].split()] if result["start_time"] else []
        ends = [float(t) for t in result["end_time"].split()] if result["end_time"] else []

        word_details = []
        skipped = 0
        mispronounced = 0
        pauses = []
        phoneme_errors = 0

        for i, expected in enumerate(words_expected):
            info = {
                "expected": expected,
                "recognized": None,
                "status": "skipped",
                "confidence": 0.0,
                "start_ts": None,
                "end_ts": None,
                "phoneme_match": [],
                "pause_after": False,
            }
            if i < len(words_predicted) and words_predicted[i] != "-":
                predicted = words_predicted[i]
                info["recognized"] = predicted
                info["start_ts"] = starts[i] if i < len(starts) else None
                info["end_ts"] = ends[i] if i < len(ends) else None
                info["confidence"] = self._compute_confidence(expected, predicted)
                if predicted.lower() == expected.lower():
                    info["status"] = "correct"
                else:
                    info["status"] = "mispronounced"
                    mispronounced += 1
                info["phoneme_match"] = self._compute_phoneme_match(expected, predicted)
                if 0 in info["phoneme_match"]:
                    phoneme_errors += 1
            else:
                skipped += 1

            word_details.append(info)

        # analyse pauses and pacing
        for idx in range(len(word_details) - 1):
            end_curr = word_details[idx]["end_ts"]
            start_next = word_details[idx + 1]["start_ts"]
            if end_curr is None or start_next is None:
                continue
            gap = start_next - end_curr
            pauses.append(gap)
            # flag unnatural pause if larger than 0.6s and no punctuation in between
            if gap > 0.6 and not words_expected[idx].strip().endswith(('.', ',', ';', ':', '?', '!')):
                word_details[idx]["pause_after"] = True

        pacing_score = 0.0
        robotic_speech = False
        if pauses:
            mean_gap = float(np.mean(pauses))
            std_gap = float(np.std(pauses))
            pacing_score = round(1.0 / (1.0 + mean_gap + std_gap), 3)
            if mean_gap > 0.8 and std_gap < 0.1:
                robotic_speech = True

        session_summary = {
            "accuracy": float(result["pronunciation_accuracy"]),
            "words_total": len(word_details),
            "mispronounced_words": mispronounced,
            "skipped_words": skipped,
            "phoneme_errors": phoneme_errors,
            "robotic_speech": robotic_speech,
            "fluency": pacing_score,
        }

        return {
            "session_summary": session_summary,
            "words": word_details,
        }


def analyse_to_json(audio, expected_text: str, language: str = "en") -> str:
    """Convenience wrapper returning a JSON string."""
    analyzer = AdvancedPronunciationAnalyzer(language)
    report = analyzer.analyze(audio, expected_text)
    return json.dumps(report, ensure_ascii=False, indent=2)


