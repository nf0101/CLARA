"""
Componente C del protocollo: iniezione di rumore controllato nel segnale NLP.

Permette di simulare classificatori di urgenza con accuratezza variabile
(100%, 90%, 80%, ...) per tracciare la curva di robustezza richiesta da RQ2.

Il modulo e' deliberatamente separato da urgency_score.py: la pipeline NLP
"vera" produce uno score, e il noise injector lo perturba. Cosi' possiamo
tenere fissa la pipeline e variare solo il rumore.
"""

from __future__ import annotations

import numpy as np


def inject_label_noise(
    urgency_scores: np.ndarray,
    ground_truth: np.ndarray,
    target_accuracy: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Perturba urgency_scores per simulare un classificatore con la accuratezza target.

    Args:
        urgency_scores: score di urgenza per zona/client, shape (N,).
        ground_truth: criticita' ground truth (Componente A), shape (N,).
        target_accuracy: accuratezza simulata in [0, 1].
        rng: generatore random per riproducibilita'.

    Returns:
        Score di urgenza perturbati, shape (N,).
    """
    raise NotImplementedError("TODO: implementare iniezione di rumore")
