import os
import torch
import numpy as np
import librosa
from pathlib import Path
from core.config import CACHE_DIR

class NeuralMomentScorer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mode = "HEURISTIC"
        self._tribe_model = None
        self._check_available_engines()

    def _check_available_engines(self):
        """Checks if Meta TRIBE v2 can run or if Universal Neural Heuristic should be used."""
        try:
            if self.device == "cuda" and torch.cuda.get_device_properties(0).total_memory >= 10 * (1024**3):
                from tribev2 import TribeModel
                print("[AI Engine] NVIDIA High-VRAM GPU detected. Initializing Meta TRIBE v2...")
                self._tribe_model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=str(CACHE_DIR / "tribe"))
                self._tribe_model.to(self.device)
                self._tribe_model.eval()
                self.mode = "TRIBE_V2"
            else:
                print(f"[AI Engine] Universal Adaptive Engine active on {self.device}. (Optimized for all PC specs)")
                self.mode = "HEURISTIC"
        except Exception as e:
            print(f"[AI Engine] TRIBE v2 library not resident ({e}). Using Universal Adaptive Engine.")
            self.mode = "HEURISTIC"

    def score_slice(self, video_slice_path: str, audio_slice_path: str, slice_words: list) -> dict:
        """
        Calculates second-by-second brain retention curve and overall viral score
        for a 30s, 60s, 90s, or 180s candidate clip.
        Returns:
            {
                "overall_score": float (0-100),
                "hook_score": float (0-100),
                "mid_score": float (0-100),
                "payoff_score": float (0-100),
                "curve": list of float (second-by-second 0-1)
            }
        """
        if self.mode == "TRIBE_V2":
            return self._score_with_tribe(video_slice_path)
        else:
            return self._score_with_multimodal_heuristic(audio_slice_path, slice_words)

    def _score_with_tribe(self, video_slice_path: str) -> dict:
        """Runs Meta TRIBE v2 foundation model inference on the slice."""
        try:
            events_df = self._tribe_model.get_events_dataframe(video_path=video_slice_path)
            with torch.no_grad():
                preds, _ = self._tribe_model.predict(events=events_df)
            
            bold = preds.cpu().numpy() if isinstance(preds, torch.Tensor) else preds
            # Calculate RMS energy across cortical vertices at 1Hz
            curve = np.sqrt(np.mean(np.square(bold), axis=1))
            norm_curve = (curve - curve.min()) / (curve.max() - curve.min() + 1e-8)
            return self._compute_metrics_from_curve(norm_curve)
        except Exception as e:
            print(f"[TRIBE v2] Evaluation fallback due to: {e}")
            return self._score_with_multimodal_heuristic(video_slice_path, [])

    def _score_with_multimodal_heuristic(self, audio_path: str, words: list) -> dict:
        """
        Universal Multimodal Scorer:
        Combines acoustic energy (RMS loudness + pitch shift dynamics)
        with linguistic hook detection (question words, surprise markers, pacing).
        """
        # 1. Acoustic Signal (1-second intervals)
        y, sr = librosa.load(audio_path, sr=16000)
        hop = 16000  # 1-second hops
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        
        # Audio energy normalization
        if len(rms) == 0:
            return {"overall_score": 50.0, "hook_score": 50.0, "mid_score": 50.0, "payoff_score": 50.0, "curve": [0.5]}
        
        norm_rms = (rms - rms.min()) / (rms.max() - rms.min() + 1e-8)

        # 2. Linguistic Hook Boost
        hook_keywords = {
            "why", "how", "secret", "never", "always", "stop", "mistake",
            "billion", "million", "money", "truth", "shocking", "rule", "best",
            "worst", "everyone", "nobody", "dont", "do not", "watch",
            "bought", "entire", "grocery", "store", "dollar", "dollars",
            "win", "crazy", "insane", "challenge", "impossible", "eat", "free", "cash"
        }

        # Baseline engagement (40%) + acoustic loudness (35%) + linguistic hooks (25%)
        curve = 0.40 + (0.35 * norm_rms)
        for i in range(len(curve)):
            sec_words = [w["word"].lower().strip(".,!?\"") for w in words if i <= w["start"] < i + 1]
            matches = sum(1 for w in sec_words if any(k in w for k in hook_keywords))
            if matches > 0:
                curve[i] += min(0.35, matches * 0.12)

        curve = np.clip(curve, 0.0, 1.0)
        return self._compute_metrics_from_curve(curve)

    def _compute_metrics_from_curve(self, curve: np.ndarray) -> dict:
        n = len(curve)
        if n == 0:
            return {"overall_score": 50.0, "hook_score": 50.0, "mid_score": 50.0, "payoff_score": 50.0, "curve": []}

        # 1. Hook score: First 10% or first 10 seconds (whichever is shorter)
        hook_len = max(3, min(10, int(n * 0.15)))
        hook_score = float(np.mean(curve[:hook_len])) * 100

        # 2. Mid score: Middle 70%
        mid_score = float(np.mean(curve[hook_len:max(hook_len+1, n - hook_len)])) * 100

        # 3. Payoff score: Final 15%
        payoff_score = float(np.mean(curve[max(0, n - hook_len):])) * 100

        # Overall composite retention score
        overall = (0.45 * hook_score) + (0.35 * mid_score) + (0.20 * payoff_score)

        return {
            "overall_score": round(overall, 1),
            "hook_score": round(hook_score, 1),
            "mid_score": round(mid_score, 1),
            "payoff_score": round(payoff_score, 1),
            "curve": [round(float(val), 3) for val in curve]
        }
