#!/usr/bin/env bash
#
# setup_runpod.sh v2 - Bootstrap di un pod runpod con pulizia aggressiva.
#
# Versione 2: cancella gli intermedi man mano per stare sotto i 50 GB di disco.
# Picco di spazio richiesto: ~40-45 GB (vs ~50+ della v1).
#
# USO:
#   1. Avvia un pod runpod con GPU L40S e template PyTorch.
#   2. cd /workspace && git clone https://TOKEN@github.com/nf0101/DLNLP.git
#   3. cd DLNLP && chmod +x scripts/setup_runpod.sh && ./scripts/setup_runpod.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# CONFIGURAZIONE
# ---------------------------------------------------------------------------

WORKDIR="/workspace"
PROJECT_DIR="${WORKDIR}/DLNLP"
FIGSHARE_ARTICLE_ID="21431547"
RAW_DIR="${PROJECT_DIR}/data/raw/rdd2022"

echo "=========================================================="
echo " BOOTSTRAP RUNPOD v2 - urban-fl-nlp"
echo " (pulizia aggressiva intermedi, target <50GB)"
echo "=========================================================="

# ---------------------------------------------------------------------------
# STEP 0: strumenti di sistema (in caso non gia' presenti)
# ---------------------------------------------------------------------------

echo "[0/7] Installo strumenti di sistema..."
apt-get update -qq
apt-get install -y -qq unzip wget curl git

# ---------------------------------------------------------------------------
# STEP 1: repo (idempotente)
# ---------------------------------------------------------------------------

cd "${WORKDIR}"
if [ -d "${PROJECT_DIR}/.git" ]; then
    echo "[1/7] Repo gia' presente, git pull..."
    cd "${PROJECT_DIR}"
    git pull
else
    echo "[1/7] Repo non presente — questo script dovrebbe essere lanciato"
    echo "      DA DENTRO il repo gia' clonato. Esco."
    exit 1
fi

# ---------------------------------------------------------------------------
# STEP 2: dipendenze Python
# ---------------------------------------------------------------------------

echo "[2/7] Installo dipendenze Python..."
pip install --quiet --upgrade pip
pip install --quiet ultralytics "flwr[simulation]" "ray[default]" \
    pandas matplotlib pyyaml

python3 -c "import torch; print('  CUDA:', torch.cuda.is_available(), \
torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

# ---------------------------------------------------------------------------
# STEP 3: download dataset da figshare
# ---------------------------------------------------------------------------

mkdir -p "${RAW_DIR}"
cd "${RAW_DIR}"

MAIN_ZIP="RDD2022.zip"

if [ -f "${MAIN_ZIP}" ]; then
    echo "[3/7] ${MAIN_ZIP} gia' presente, salto il download."
else
    echo "[3/7] Recupero URL di download da figshare API..."
    DOWNLOAD_URL=$(curl -s "https://api.figshare.com/v2/articles/${FIGSHARE_ARTICLE_ID}" \
        | python3 -c "
import sys, json
data = json.load(sys.stdin)
for f in data.get('files', []):
    if f['name'].lower().endswith('.zip'):
        print(f['download_url'])
        break
")
    if [ -z "${DOWNLOAD_URL}" ]; then
        echo "ERRORE: file .zip non trovato. Ispeziona con:"
        echo "  curl -s https://api.figshare.com/v2/articles/${FIGSHARE_ARTICLE_ID} | python3 -m json.tool"
        exit 1
    fi
    echo "  Scarico (qualche minuto)..."
    wget --progress=dot:giga -O "${MAIN_ZIP}" "${DOWNLOAD_URL}"
fi

df -h "${RAW_DIR}" | tail -1

# ---------------------------------------------------------------------------
# STEP 4: estrazione livello 1 -> cancello subito lo zip grande
# ---------------------------------------------------------------------------

echo "[4/7] Estrazione livello 1 (RDD2022.zip)..."

if [ ! -d "RDD2022" ]; then
    unzip -q -o "${MAIN_ZIP}"
fi

# Cancello lo zip grande appena ho estratto (libera ~12 GB)
echo "  Cancello ${MAIN_ZIP} (libera ~12 GB)..."
rm -f "${MAIN_ZIP}"

df -h "${RAW_DIR}" | tail -1

# ---------------------------------------------------------------------------
# STEP 5: estrazione livello 2 (zip per-paese) con cancellazione progressiva
# ---------------------------------------------------------------------------

echo "[5/7] Estrazione livello 2 (zip per-paese, con cancellazione progressiva)..."

INNER_ZIPS=$(find . -maxdepth 2 -name "*.zip" 2>/dev/null || true)

if [ -n "${INNER_ZIPS}" ]; then
    for z in ${INNER_ZIPS}; do
        dest=$(dirname "${z}")
        echo "  unzip $(basename ${z})"
        unzip -q -o "${z}" -d "${dest}"
        rm "${z}"
    done
else
    echo "  Nessun zip annidato, struttura gia' piatta."
fi

# Verifica struttura: cerca la cartella Japan
JAPAN_DIR=$(find . -type d -name "Japan" | head -1)
if [ -z "${JAPAN_DIR}" ]; then
    echo "ERRORE: cartella Japan non trovata dopo l'estrazione."
    find . -maxdepth 3 -type d
    exit 1
fi

EXPECTED_ROOT="${RAW_DIR}/RDD2022"
ACTUAL_PARENT=$(dirname "${JAPAN_DIR}")
if [ "${ACTUAL_PARENT}" != "${EXPECTED_ROOT}" ]; then
    echo "  ATTENZIONE: paesi in ${ACTUAL_PARENT} invece di ${EXPECTED_ROOT}."
    echo "  Verifica RAW_ROOT in scripts/convert_voc_to_yolo_all.py."
fi

df -h "${RAW_DIR}" | tail -1

# ---------------------------------------------------------------------------
# STEP 6: conversione XML -> YOLO (6 client)
# ---------------------------------------------------------------------------

echo "[6/7] Conversione XML -> YOLO..."
cd "${PROJECT_DIR}"

CLIENTS_DIR="${PROJECT_DIR}/data/processed/rdd2022_clients"
if [ -f "${CLIENTS_DIR}/global_manifest.json" ]; then
    echo "  Gia' convertito, salto."
else
    python3 scripts/convert_voc_to_yolo_all.py --force
fi

# Cancello il dataset grezzo (~13 GB): non serve piu' per il training,
# Ultralytics legge solo da data/processed/rdd2022_clients/
echo "  Cancello data/raw/rdd2022/RDD2022/ (libera ~13 GB)..."
rm -rf "${RAW_DIR}/RDD2022"

df -h "${PROJECT_DIR}" | tail -1

# ---------------------------------------------------------------------------
# STEP 7: sanity check finale
# ---------------------------------------------------------------------------

echo "[7/7] Sanity check..."
python3 -c "
import json
from pathlib import Path
m = Path('${CLIENTS_DIR}/global_manifest.json')
if not m.exists():
    raise SystemExit('ERRORE: global_manifest.json non trovato')
d = json.loads(m.read_text())
print('  Client:', [s['client_name'] for s in d['per_country_stats']])
tt = sum(s['train']['images'] for s in d['per_country_stats'])
tv = sum(s['valid']['images'] for s in d['per_country_stats'])
print(f'  Immagini: train={tt}, valid={tv}')
print('  Atteso:   train=30590, valid=5394')
print('  Match:   ', 'OK' if (tt, tv) == (30590, 5394) else 'DIVERSO (verifica)')
"

echo "=========================================================="
echo " SETUP COMPLETATO. Spazio finale:"
df -h "${WORKDIR}" | tail -1
echo ""
echo " Prossimo: lancia una run di prova"
echo "   cd ${PROJECT_DIR}"
echo "   python3 scripts/fl_run.py --strategy fedavg --rounds 3 \\"
echo "       --local-epochs 1 --imgsz 416"
echo "=========================================================="
