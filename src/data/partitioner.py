"""
Partizionatore Non-IID per RDD2022.

Lo split country-wise produce 6 client FL con distribuzioni di classe
fortemente sbilanciate (Non-IID severo), come richiesto da RQ1.
"""

from __future__ import annotations

from pathlib import Path


def partition_rdd2022_by_country(
    raw_dir: Path,
    output_dir: Path,
    positive_only: bool = True,
) -> dict[str, int]:
    """Crea uno split country-wise di RDD2022.

    Args:
        raw_dir: directory contenente RDD2022 estratto.
        output_dir: directory di output (un sotto-folder per paese).
        positive_only: se True, mantiene solo immagini con almeno una bbox.

    Returns:
        Dizionario {country: num_immagini} con il count per client.
    """
    raise NotImplementedError("TODO: implementare partitioner")
