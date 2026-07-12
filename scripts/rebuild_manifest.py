"""
Ricostruisce pipeline_manifest.json scansionando il disco.

Necessario quando il manifest non è aggiornato (es. dopo un --resume che ha
completato run senza scriverle nel manifest). Legge tutti i results.json
esistenti nelle cartelle rq1/, rq2/, ablation/ e ricostruisce le tre liste
del manifest da zero.

Uso:
    python3 scripts/rebuild_manifest.py <pipeline_dir>

Esempio:
    python3 scripts/rebuild_manifest.py experiments/report_pipeline/20260710_123413
"""

import json
import sys
from pathlib import Path


def find_runs(phase_dir: Path, phase_name: str, project_root: Path) -> list[dict]:
    """Scansiona una cartella di fase e ritorna la lista di run completate."""
    runs = []
    if not phase_dir.exists():
        return runs

    for run_dir in sorted(phase_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        rjson = run_dir / "results.json"
        if not rjson.exists():
            continue

        run_id = run_dir.name
        try:
            rel_path = str(rjson.relative_to(project_root))
        except ValueError:
            rel_path = str(rjson)

        entry = {
            "phase": phase_name,
            "run_id": run_id,
            "success": True,
            "results_json_path": rel_path,
        }

        # Estrai parametri dal nome del run_id
        parts = run_id.split("_")
        try:
            if phase_name == "rq1":
                # Formato: {strategy}_seed{N}
                entry["strategy"] = parts[0]
                seed_str = [p for p in parts if p.startswith("seed")][0]
                entry["seed"] = int(seed_str.replace("seed", ""))

            elif phase_name == "rq2":
                # Formato: nlp_acc{NNN}_seed{N}
                entry["strategy"] = "nlp"
                acc_str = [p for p in parts if p.startswith("acc")][0]
                entry["noise_accuracy"] = int(acc_str.replace("acc", "")) / 100.0
                seed_str = [p for p in parts if p.startswith("seed")][0]
                entry["seed"] = int(seed_str.replace("seed", ""))

            elif phase_name == "ablation":
                # Formato: nlp_cap{X.X}_td{X.X}_seed{N}
                entry["strategy"] = "nlp"
                cap_str = [p for p in parts if p.startswith("cap")][0]
                entry["cap"] = float(cap_str.replace("cap", ""))
                td_str = [p for p in parts if p.startswith("td")][0]
                entry["time_decay"] = float(td_str.replace("td", ""))
                seed_str = [p for p in parts if p.startswith("seed")][0]
                entry["seed"] = int(seed_str.replace("seed", ""))
        except (IndexError, ValueError) as e:
            print(f"  ATTENZIONE: nome run non parsabile: {run_id} ({e})")

        runs.append(entry)

    return runs


def main() -> int:
    if len(sys.argv) < 2:
        print("Uso: python3 rebuild_manifest.py <pipeline_dir>")
        return 1

    pipeline_dir = Path(sys.argv[1]).resolve()
    if not pipeline_dir.exists():
        print(f"ERRORE: cartella non trovata: {pipeline_dir}")
        return 1

    manifest_path = pipeline_dir / "pipeline_manifest.json"
    if not manifest_path.exists():
        print(f"ERRORE: manifest non trovato: {manifest_path}")
        return 1

    # Carica il manifest esistente (per preservare metadata come scale, seeds, etc.)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # project_root = due livelli sopra pipeline_dir (experiments/report_pipeline/<id>)
    project_root = pipeline_dir.parents[1]

    # Ricostruisci le tre liste da disco
    manifest["rq1_runs"] = find_runs(pipeline_dir / "rq1", "rq1", project_root)
    manifest["rq2_runs"] = find_runs(pipeline_dir / "rq2", "rq2", project_root)
    manifest["ablation_runs"] = find_runs(
        pipeline_dir / "ablation", "ablation", project_root
    )

    manifest["completed_runs"] = (
        len(manifest["rq1_runs"])
        + len(manifest["rq2_runs"])
        + len(manifest["ablation_runs"])
    )

    # Salva
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Manifest ricostruito: {manifest_path}")
    print(f"  RQ1:      {len(manifest['rq1_runs'])} run")
    print(f"  RQ2:      {len(manifest['rq2_runs'])} run")
    print(f"  Ablation: {len(manifest['ablation_runs'])} run")
    print(f"  Totale:   {manifest['completed_runs']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())