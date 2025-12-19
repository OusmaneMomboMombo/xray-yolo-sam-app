from flask import (
    Flask,
    request,
    render_template,
    send_from_directory,
    jsonify,
    url_for,
)
from ultralytics import YOLO
import os
import uuid
import numpy as np
import cv2
from datetime import datetime

# === UTILITAIRES ===
from utils.metrics import void_rate_from_masks
from utils.sam_utils import (
    load_sam_model,
    sam_from_point,
    sam_from_bbox,
    mask_to_base64,
)
import shutil
import yaml


ACTIVE_LEARNING_BASE = os.path.join("data", "active_learning")

AL_IMAGES = os.path.join(ACTIVE_LEARNING_BASE, "images")
AL_LABELS = os.path.join(ACTIVE_LEARNING_BASE, "labels")
RETRAIN_DATASET = "dataset_retrain"

os.makedirs(AL_IMAGES, exist_ok=True)
os.makedirs(AL_LABELS, exist_ok=True)

from utils.retrain_yolo import retrain_yolo


###########################################
# CONFIG FLASK
###########################################

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs("runs_app/pred", exist_ok=True)

# Dossiers pour corrections
CORRECTED_BASE = os.path.join("data", "corrected")
CORRECTED_IMAGES_DIR = os.path.join(CORRECTED_BASE, "images")
CORRECTED_MASKS_PNG_DIR = os.path.join(CORRECTED_BASE, "masks_png")
CORRECTED_MASKS_TXT_DIR = os.path.join(CORRECTED_BASE, "masks_txt")

os.makedirs(CORRECTED_IMAGES_DIR, exist_ok=True)
os.makedirs(CORRECTED_MASKS_PNG_DIR, exist_ok=True)
os.makedirs(CORRECTED_MASKS_TXT_DIR, exist_ok=True)

# Variables globales pour stocker l'état courant
LAST_IMAGE_PATH = None           # chemin de l'image d'entrée uploadée
LAST_YOLO_IMAGE_NAME = None      # nom de fichier de l'image annotée par YOLO
LAST_BBOXES = []                 # bboxes YOLO
LAST_SAM_MASKS = []              # liste de masques SAM (np.array HxW)


###########################################
# CHARGER LES MODÈLES
###########################################

# YOLO
model = YOLO("models/best.pt")

# SAM
sam_predictor = load_sam_model(
    model_path="models/sam_vit_b.pth",
    model_type="vit_b",
    device="cpu"
)


###########################################
# ROUTES POUR SERVIR DES FICHIERS
###########################################

@app.route("/pred/<path:filename>")
def pred_image(filename):
    """Image annotée par YOLO (runs_app/pred)."""
    return send_from_directory("runs_app/pred", filename)


@app.route("/corrected/<path:filename>")
def corrected_image(filename):
    """Image corrigée (overlay) dans data/corrected/images."""
    return send_from_directory(CORRECTED_IMAGES_DIR, filename)


###########################################
# ROUTE ACCUEIL
###########################################

@app.route("/")
def index():
    return render_template("index.html", yolo_image=None)


###########################################
# 1️⃣ PREDICTION YOLO (UPLOAD + ANALYSE)
###########################################

@app.route("/predict", methods=["POST"])
def predict():
    global LAST_IMAGE_PATH, LAST_YOLO_IMAGE_NAME, LAST_BBOXES, LAST_SAM_MASKS

    # Reset des masques SAM dès qu'on change d'image
    LAST_SAM_MASKS = []

    # 1. Upload image
    f = request.files["image"]
    in_name = f"{uuid.uuid4().hex}_{f.filename}"
    in_path = os.path.join(app.config["UPLOAD_FOLDER"], in_name)
    f.save(in_path)

    LAST_IMAGE_PATH = in_path
    LAST_YOLO_IMAGE_NAME = in_name

    # 2. YOLO inference
    results = model.predict(
        source=in_path,
        save=True,
        project="runs_app",
        name="pred",
        exist_ok=True,
        imgsz=640,
    )

    r = results[0]

    # Récupérer les bounding boxes
    LAST_BBOXES = []
    if r.boxes is not None:
        for box in r.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box
            LAST_BBOXES.append([int(x1), int(y1), int(x2), int(y2)])

    # Calcul du Void Rate
    void_rate_yolo = None

    if r.masks is not None:
        mask_stack = r.masks.data.cpu().numpy()  # (N,H,W)
        ids = r.boxes.cls.int().cpu().tolist()
        names = [r.names[i] for i in ids]
        vr = void_rate_from_masks(mask_stack, names)
        void_rate_yolo = round(vr, 2)
        print(f"Void Rate (YOLO): {void_rate_yolo}%")




    # URL Flask pour l'image annotée YOLO
    yolo_url = url_for("pred_image", filename=in_name)

    return render_template(
        "index.html",
        yolo_image=yolo_url,
        void_rate_yolo=void_rate_yolo
)


###########################################
# 2️⃣ VALIDATION SANS CORRECTION (YOLO OK)
###########################################

@app.route("/validate", methods=["POST"])
def validate():
    global LAST_IMAGE_PATH
    print(f"[{datetime.now()}] Image VALIDÉE sans correction SAM : {LAST_IMAGE_PATH}")
    return ("", 204)


###########################################
# 3️⃣ SAM VIA CLIC UTILISATEUR (MULTI-SÉLECTION)
###########################################

@app.route("/segment_point", methods=["POST"])
def segment_point():
    global LAST_IMAGE_PATH, LAST_SAM_MASKS

    data = request.get_json()
    point = data.get("point")

    if LAST_IMAGE_PATH is None:
        return jsonify({"error": "Aucune image analysée"}), 400

    # Appel SAM -> masque binaire (H,W)
    mask = sam_from_point(sam_predictor, LAST_IMAGE_PATH, point)
    LAST_SAM_MASKS.append(mask.astype(np.uint8))

    # Retour pour visualisation côté front
    mask_b64 = mask_to_base64(mask)
    return jsonify({"mask": mask_b64})


###########################################
# 4️⃣ RESET DES MASQUES SAM (FRONT "Réinitialiser")
###########################################

@app.route("/reset_masks", methods=["POST"])
def reset_masks():
    global LAST_SAM_MASKS
    LAST_SAM_MASKS = []
    print(f"[{datetime.now()}] Masques SAM réinitialisés pour {LAST_IMAGE_PATH}")
    return ("", 204)

@app.route("/save_correction", methods=["POST"])
def save_correction():
    """
    Fusionne les masques SAM courants, enregistre les fichiers,
    renvoie l'URL de l'image corrigée + le void rate corrigé.
    """
    global LAST_SAM_MASKS, LAST_IMAGE_PATH

    if LAST_IMAGE_PATH is None:
        return jsonify({"error": "Aucune image en cours"}), 400

    if not LAST_SAM_MASKS:
        return jsonify({"error": "Aucun masque SAM à sauver"}), 400

    result = save_union_mask_and_overlay()
    if result is None:
        return jsonify({"error": "Erreur lors de la création de l'overlay"}), 500

    corrected_filename, void_rate_corrected = result

    # Reset les masques SAM après sauvegarde
    LAST_SAM_MASKS = []

    corrected_url = url_for("corrected_image", filename=corrected_filename)

    return jsonify({
        "corrected_image_url": corrected_url,
        "void_rate_corrected": round(void_rate_corrected, 2)
    })




###########################################
# 5️⃣ SAUVEGARDE DE LA CORRECTION (PNG + TXT + OVERLAY)
###########################################

def save_union_mask_and_overlay():
    """
    Fusionne les masques YOLO + corrections SAM, et produit :
    - mask PNG final (voids complets)
    - fichier YOLO TXT final
    - image overlay (chip + voids YOLO + voids SAM)
    - renvoie (corrected_image_name, void_rate_corrected)
    """
    global LAST_SAM_MASKS, LAST_IMAGE_PATH, LAST_BBOXES

    if LAST_IMAGE_PATH is None:
        return None

    # Charger l'image originale
    img = cv2.imread(LAST_IMAGE_PATH)
    if img is None:
        return None

    # 1️⃣ Récupérer la prédiction YOLO associée
    results = model(LAST_IMAGE_PATH, imgsz=640)[0]

    if results.masks is None:
        return None

    masks_yolo = results.masks.data.cpu().numpy()  # (N, H, W)
    classes = results.boxes.cls.cpu().numpy()      # classes prédictes

    # Séparer chip / void
    mask_chip = np.zeros_like(masks_yolo[0], dtype=np.uint8)
    mask_voids_yolo = np.zeros_like(masks_yolo[0], dtype=np.uint8)

    for mask, cls in zip(masks_yolo, classes):
        if cls == 0:  # chip
            mask_chip = np.logical_or(mask_chip, mask > 0)
        elif cls == 1:  # void
            mask_voids_yolo = np.logical_or(mask_voids_yolo, mask > 0)

    mask_chip = mask_chip.astype(np.uint8)
    mask_voids_yolo = mask_voids_yolo.astype(np.uint8)

    # 2️⃣ Fusion des masques SAM
    mask_voids_sam = np.zeros_like(mask_voids_yolo, dtype=np.uint8)
    for m in LAST_SAM_MASKS:
        m = cv2.resize(m, (mask_voids_yolo.shape[1], mask_voids_yolo.shape[0]),
                       interpolation=cv2.INTER_NEAREST)
        mask_voids_sam = np.logical_or(mask_voids_sam, m > 0)

    mask_voids_sam = mask_voids_sam.astype(np.uint8)

    # 3️⃣ Fusion YOLO + SAM pour voids
    mask_voids_final = np.logical_or(mask_voids_yolo, mask_voids_sam).astype(np.uint8)

    # 4️⃣ Sauvegarde du masque PNG final
    base_name = os.path.splitext(os.path.basename(LAST_IMAGE_PATH))[0]
    mask_png_name = f"{base_name}_void_mask.png"
    mask_png_path = os.path.join(CORRECTED_MASKS_PNG_DIR, mask_png_name)
    cv2.imwrite(mask_png_path, (mask_voids_final * 255).astype(np.uint8))

    # 5️⃣ Conversion en TXT YOLO segmentation
    h_img, w_img = mask_voids_final.shape
    bin_mask = (mask_voids_final > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    yolo_txt_name = f"{base_name}_void.txt"
    yolo_txt_path = os.path.join(CORRECTED_MASKS_TXT_DIR, yolo_txt_name)

    with open(yolo_txt_path, "w") as f:
        for cnt in contours:
            if len(cnt) < 3:
                continue
            line = ["1"]  # classe void
            for pt in cnt:
                x, y = pt[0]
                line.append(f"{x / w_img:.6f}")
                line.append(f"{y / h_img:.6f}")
            f.write(" ".join(line) + "\n")

    # 6️⃣ Création de l'IMAGE FINALE OVERLAY
    overlay = img.copy()

    # chip YOLO → ORANGE/bleu
    overlay[mask_chip > 0] = (
        overlay[mask_chip > 0] * 0.6 + np.array([255, 140, 0]) * 0.4
    ).astype(np.uint8)

    # voids YOLO → BLEU CYAN
    overlay[mask_voids_yolo > 0] = (
        overlay[mask_voids_yolo > 0] * 0.5 + np.array([56, 189, 248]) * 0.5
    ).astype(np.uint8)

    # voids SAM → JAUNE (pour les distinguer)
    overlay[mask_voids_sam > 0] = (
        overlay[mask_voids_sam > 0] * 0.5 + np.array([255, 255, 0]) * 0.5
    ).astype(np.uint8)

    corrected_name = f"{base_name}_corrected.png"
    corrected_path = os.path.join(CORRECTED_IMAGES_DIR, corrected_name)
    cv2.imwrite(corrected_path, overlay)

    # 7️⃣ CALCUL DU NOUVEAU VOID RATE (IMPORTANT)
    void_area = np.sum(mask_voids_final > 0)
    chip_area = np.sum(mask_chip > 0)

    if chip_area > 0:
        void_rate_corrected = (void_area / chip_area) * 100
    else:
        void_rate_corrected = 0.0

    print("Correction complète sauvegardée.")
    print("Overlay :", corrected_path)
    print("Mask PNG :", mask_png_path)
    print("TXT YOLO :", yolo_txt_path)
    print("Void Rate corrigé :", void_rate_corrected)
    
    # 8️⃣ AJOUT AU DATASET ACTIVE LEARNING
    add_to_active_learning(LAST_IMAGE_PATH, yolo_txt_path)

    return corrected_name, void_rate_corrected

def add_to_active_learning(image_path, yolo_txt_path):
    """
    Copie une image + son label corrigé
    dans le dataset Active Learning.
    """
    base = os.path.splitext(os.path.basename(image_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    img_dst = f"{base}_{timestamp}.jpg"
    lbl_dst = f"{base}_{timestamp}.txt"

    shutil.copy(image_path, os.path.join(AL_IMAGES, img_dst))
    shutil.copy(yolo_txt_path, os.path.join(AL_LABELS, lbl_dst))

    print(f"[ActiveLearning] Ajouté : {img_dst} / {lbl_dst}")

def prepare_retrain_dataset():
    """
    Crée un dataset YOLO pour le retrain en combinant :
    - dataset original
    - données corrigées (active learning)
    """
    os.makedirs(RETRAIN_DATASET, exist_ok=True)

    for split in ["train", "valid"]:
        os.makedirs(os.path.join(RETRAIN_DATASET, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(RETRAIN_DATASET, split, "labels"), exist_ok=True)

    # 1️⃣ Copier ancien TRAIN
    shutil.copytree(
        "dataset/train/images",
        f"{RETRAIN_DATASET}/train/images",
        dirs_exist_ok=True
    )
    shutil.copytree(
        "dataset/train/labels",
        f"{RETRAIN_DATASET}/train/labels",
        dirs_exist_ok=True
    )

    # 2️⃣ Ajouter Active Learning (TRAIN)
    shutil.copytree(AL_IMAGES, f"{RETRAIN_DATASET}/train/images", dirs_exist_ok=True)
    shutil.copytree(AL_LABELS, f"{RETRAIN_DATASET}/train/labels", dirs_exist_ok=True)

    # 3️⃣ Copier VALID
    shutil.copytree(
        "dataset/valid/images",
        f"{RETRAIN_DATASET}/valid/images",
        dirs_exist_ok=True
    )
    shutil.copytree(
        "dataset/valid/labels",
        f"{RETRAIN_DATASET}/valid/labels",
        dirs_exist_ok=True
    )

    # 4️⃣ data.yaml
    with open("dataset/data.yaml", "r") as f:
        data_cfg = yaml.safe_load(f)

    data_cfg["train"] = os.path.abspath(f"{RETRAIN_DATASET}/train/images")
    data_cfg["val"] = os.path.abspath(f"{RETRAIN_DATASET}/valid/images")

    with open(f"{RETRAIN_DATASET}/data.yaml", "w") as f:
        yaml.dump(data_cfg, f)

    print("✅ Dataset retrain prêt.")

@app.route("/retrain", methods=["POST"])
def retrain():
    global model

    try:
        prepare_retrain_dataset()

        print("🚀 Lancement du retrain YOLO...")

        model.train(
            data=f"{RETRAIN_DATASET}/data.yaml",
            epochs=5,
            imgsz=640,
            batch=4,
            device="cpu",
            name="retrain",
            project="runs_retrain",
            exist_ok=True
        )

        # Charger le nouveau best.pt
        new_model_path = "runs_retrain/retrain/weights/best.pt"
        model = YOLO(new_model_path)

        print("✅ Retrain terminé et modèle mis à jour.")

        return jsonify({
            "status": "ok",
            "message": "Retrain terminé",
            "model_path": new_model_path
        })

    except Exception as e:
        print("❌ Erreur retrain :", e)
        return jsonify({"error": str(e)}), 500



###########################################
# MAIN
###########################################

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)

