from pathlib import Path

# ================================
# 1) Localisation racine du projet
# ================================

# config.py est dans : project/src/processing/config.py
# donc PROJECT_ROOT = project/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Vérification
data_root = PROJECT_ROOT / "data" / "raw" / "train"
if not data_root.exists():
    raise RuntimeError(
        f" data/raw/train introuvable.\n"
        f"Chemin attendu : {data_root}\n"
        "Structure obligatoire : project/data/raw/train/<classe>/"
    )

# ================================
# 2) Définition des chemins
# ================================

RAW_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_DIR = RAW_DIR / "train"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_NPZ = PROJECT_ROOT / "data" / "processed" / "processed_fer2013.npz"

for d in [RAW_DIR, TRAIN_DIR, PROCESSED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ================================
# 3) Paramètres images
# ================================

IMG_SIZE = (48, 48)
BATCH_SIZE = 64
RANDOM_SEED = 42

GRAYSCALE = True

# ================================
# 4) Détection visages
# ================================

FACE_DETECTOR = "mtcnn"
BLUR_THRESHOLD = 50.0
MIN_FACE_SIZE = 20

# ================================
# 5) Augmentation
# ================================

AUGMENTATION_ENABLED = True
AUG_PER_IMAGE = 3

# ================================
# 6) Détection des classes
# ================================

CLASS_NAMES = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])

if not CLASS_NAMES:
    raise ValueError(
        f"❌ Aucune classe trouvée dans {TRAIN_DIR}.\n"
        "Tu dois avoir : data/raw/train/<classe>/"
    )

NUM_CLASSES = len(CLASS_NAMES)
print("[CONFIG] Classes détectées :", CLASS_NAMES)

# ================================
# 7) Dossiers modèles / logs
# ================================

MODEL_DIR = PROJECT_ROOT / "experiments" / "checkpoints"
LOG_DIR = PROJECT_ROOT / "experiments" / "logs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ================================
# 8) Fonction debug
# ================================

def print_config():
    print("\n===== CONFIG =====")
    print("PROJECT_ROOT  :", PROJECT_ROOT)
    print("RAW_DIR       :", RAW_DIR)
    print("TRAIN_DIR     :", TRAIN_DIR)
    print("CLASSES       :", CLASS_NAMES)
    print("NUM_CLASSES   :", NUM_CLASSES)
    print("====================\n")
