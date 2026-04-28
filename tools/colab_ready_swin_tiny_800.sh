#!/usr/bin/env bash
set -euo pipefail

# Colab-ready bootstrap for U2Seg Swin-Tiny (cluster 800).
# Idempotent: safe to rerun after runtime restart.

export DEBIAN_FRONTEND=noninteractive

U2SEG_ROOT="${U2SEG_ROOT:-/content/U2Seg}"
if [[ ! -d "${U2SEG_ROOT}" && -d "/content/drive/MyDrive/U2Seg" ]]; then
  U2SEG_ROOT="/content/drive/MyDrive/U2Seg"
fi
if [[ ! -d "${U2SEG_ROOT}" ]]; then
  echo "[ERROR] U2Seg repo not found. Set U2SEG_ROOT or clone repo first."
  exit 1
fi
cd "${U2SEG_ROOT}"

DATA_MODE="${DATA_MODE:-local}" # local | drive
DRIVE_COCO_ROOT="${DRIVE_COCO_ROOT:-/content/drive/MyDrive/Datasets/coco}"
DRIVE_U2SEG_ANNOT_ROOT="${DRIVE_U2SEG_ANNOT_ROOT:-/content/drive/MyDrive/Datasets/U2Seg_Annot/u2seg_annotations}"
DRIVE_PANOPTIC_ROOT="${DRIVE_PANOPTIC_ROOT:-/content/drive/MyDrive/Datasets/panoptic_anns}"
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-${U2SEG_ROOT}/datasets_local}"
COCO_ROOT="${COCO_ROOT:-${DRIVE_COCO_ROOT}}"
U2SEG_ANNOT_ROOT="${U2SEG_ANNOT_ROOT:-${DRIVE_U2SEG_ANNOT_ROOT}}"
CLUSTER_NUM="${CLUSTER_NUM:-800}"
SPLIT="${SPLIT:-train}"
RUN_STEP1="${RUN_STEP1:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
MAX_ITER="${MAX_ITER:-2}"
LONG_TRAIN="${LONG_TRAIN:-0}"

echo "[INFO] Repo root      : ${U2SEG_ROOT}"
echo "[INFO] Data mode      : ${DATA_MODE}"
echo "[INFO] Cluster num    : ${CLUSTER_NUM}"

python -m pip install --upgrade -q pip setuptools wheel
python -m pip install -q fvcore pycocotools scikit-image pillow gdown
if ! python - <<'PY'
import importlib
importlib.import_module("panopticapi")
print("ok")
PY
then
  python -m pip install -q "git+https://github.com/cocodataset/panopticapi.git"
fi
if ! python - <<'PY'
import importlib
importlib.import_module("detectron2")
print("ok")
PY
then
  python -m pip install -q -e .
fi

if [[ "${DATA_MODE}" == "local" ]]; then
  echo "[INFO] Preparing local dataset copy under ${LOCAL_DATA_ROOT}"
  mkdir -p "${LOCAL_DATA_ROOT}/coco" "${LOCAL_DATA_ROOT}/u2seg_annotations" "${LOCAL_DATA_ROOT}/panoptic_anns"
  rsync -a --delete "${DRIVE_COCO_ROOT}/" "${LOCAL_DATA_ROOT}/coco/"
  rsync -a --delete "${DRIVE_U2SEG_ANNOT_ROOT}/" "${LOCAL_DATA_ROOT}/u2seg_annotations/"
  if [[ -d "${DRIVE_PANOPTIC_ROOT}" ]]; then
    rsync -a --delete "${DRIVE_PANOPTIC_ROOT}/" "${LOCAL_DATA_ROOT}/panoptic_anns/"
  fi
  COCO_ROOT="${LOCAL_DATA_ROOT}/coco"
  U2SEG_ANNOT_ROOT="${LOCAL_DATA_ROOT}/u2seg_annotations"
  echo "[INFO] COCO root      : ${COCO_ROOT}"
  echo "[INFO] U2Seg ann root : ${U2SEG_ANNOT_ROOT}"
else
  echo "[INFO] COCO root      : ${COCO_ROOT}"
  echo "[INFO] U2Seg ann root : ${U2SEG_ANNOT_ROOT}"
fi

mkdir -p datasets datasets/datasets datasets/prepare_ours
if [[ -e datasets/coco || -L datasets/coco ]]; then rm -rf datasets/coco; fi
ln -s "${COCO_ROOT}" datasets/coco
if [[ -e datasets/prepare_ours/u2seg_annotations || -L datasets/prepare_ours/u2seg_annotations ]]; then
  rm -rf datasets/prepare_ours/u2seg_annotations
fi
ln -s "${U2SEG_ANNOT_ROOT}" datasets/prepare_ours/u2seg_annotations
if [[ -e datasets/datasets/coco || -L datasets/datasets/coco ]]; then rm -rf datasets/datasets/coco; fi
ln -s ../coco datasets/datasets/coco
if [[ -e datasets/datasets/panoptic_anns || -L datasets/datasets/panoptic_anns ]]; then rm -rf datasets/datasets/panoptic_anns; fi
if [[ "${DATA_MODE}" == "local" && -d "${LOCAL_DATA_ROOT}/panoptic_anns" ]]; then
  ln -s "${LOCAL_DATA_ROOT}/panoptic_anns" datasets/datasets/panoptic_anns
elif [[ -d "${DRIVE_PANOPTIC_ROOT}" ]]; then
  ln -s "${DRIVE_PANOPTIC_ROOT}" datasets/datasets/panoptic_anns
fi

if [[ ! -f datasets/datasets/panoptic_anns/panoptic_train2017.json ]]; then
  echo "[WARN] datasets/datasets/panoptic_anns/panoptic_train2017.json belum ada."
  echo "[WARN] Step 1 otomatis dimatikan kecuali RUN_STEP1=1 dan file tersedia."
fi

if [[ "${RUN_STEP1}" == "1" ]]; then
  echo "[INFO] Running Step 1..."
  python datasets/prepare_ours/generate_pseudo_panoptic.py --class_num "${CLUSTER_NUM}" --split "${SPLIT}"
fi

STUFF_DIR="datasets/prepare_ours/u2seg_annotations/panoptic_annotations/panoptic_stuff_coco${SPLIT}_${CLUSTER_NUM}"
if [[ ! -d "${STUFF_DIR}" ]]; then
  echo "[INFO] Running Step 2..."
  python datasets/prepare_ours/prepare_stuff_panoptic_fpn.py --cluster_num "${CLUSTER_NUM}" --split "${SPLIT}"
else
  echo "[INFO] Step 2 output already exists: ${STUFF_DIR}"
fi

mkdir -p weights
SWIN_WEIGHT="weights/swin_tiny_patch4_window7_224.pth"
if [[ ! -f "${SWIN_WEIGHT}" ]]; then
  echo "[INFO] Downloading Swin-Tiny pretrained weight..."
  curl -L --fail -o "${SWIN_WEIGHT}" \
    "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth"
fi

python - <<'PY'
import json, os, copy
from pathlib import Path

cluster_num = os.environ.get("CLUSTER_NUM", "800")
split = os.environ.get("SPLIT", "train")
root = Path("datasets/prepare_ours/u2seg_annotations")
ins_dir = root / "ins_annotations"
pan_dir = root / "panoptic_annotations"

ins_panoptic_path = ins_dir / f"coco{split}_{cluster_num}_ins_panoptic.json"
instance_json_path = ins_dir / f"coco{split}_{cluster_num}.json"
stuff_dir = pan_dir / f"panoptic_stuff_coco{split}_{cluster_num}"
train_img_dir = Path("datasets/coco") / f"{split}2017"

if not ins_panoptic_path.exists():
    raise FileNotFoundError(f"Missing: {ins_panoptic_path}")
if not stuff_dir.exists():
    raise FileNotFoundError(f"Missing: {stuff_dir}")
if not train_img_dir.exists():
    raise FileNotFoundError(f"Missing: {train_img_dir}")

needs_rebuild = True
if instance_json_path.exists():
    try:
        data = json.load(open(instance_json_path, "r"))
        anns = data.get("annotations", [])
        needs_rebuild = not (
            isinstance(anns, list)
            and (len(anns) == 0 or ("id" in anns[0] and "image_id" in anns[0]))
        )
    except Exception:
        needs_rebuild = True

if needs_rebuild:
    src = json.load(open(ins_panoptic_path, "r"))
    out = {
        "images": copy.deepcopy(src.get("images", [])),
        "annotations": [],
        "categories": copy.deepcopy(src.get("categories", [])),
        "info": copy.deepcopy(src.get("info", {})),
        "licenses": copy.deepcopy(src.get("licenses", [])),
    }
    ann_id = 1
    for image_id_str, item in src.get("annotations", {}).items():
        image_id = int(image_id_str)
        for seg in item.get("segments_info", []):
            x, y, w, h = seg.get("bbox", [0, 0, 0, 0])
            ann = {
                "id": ann_id,
                "image_id": seg.get("image_id", image_id),
                "category_id": int(seg["category_id"]),
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(seg.get("area", w * h)),
                "iscrowd": int(seg.get("iscrowd", 0)),
                "segmentation": seg["segmentation"],
            }
            out["annotations"].append(ann)
            ann_id += 1
    json.dump(out, open(instance_json_path, "w"), ensure_ascii=False)

data = json.load(open(instance_json_path, "r"))
image_name_set = {p.name for p in train_img_dir.glob("*.jpg")}
stuff_name_set = {p.name.replace(".png", ".jpg") for p in stuff_dir.glob("*.png")}
keep_names = image_name_set & stuff_name_set

images = data.get("images", [])
keep_images = [im for im in images if im.get("file_name") in keep_names]
keep_ids = {im["id"] for im in keep_images}
annotations = [ann for ann in data.get("annotations", []) if ann.get("image_id") in keep_ids]

data["images"] = keep_images
data["annotations"] = annotations
json.dump(data, open(instance_json_path, "w"), ensure_ascii=False)
print(f"[INFO] Final {instance_json_path}: images={len(keep_images)} annotations={len(annotations)}")
PY

export CLUSTER_NUM="${CLUSTER_NUM}"
if [[ "${LONG_TRAIN}" == "1" ]]; then
  echo "[INFO] Starting Swin-Tiny long training (resume-able, eval disabled)..."
  python tools/train_net_swin.py \
    --num-gpus "${NUM_GPUS}" \
    --resume \
    --config-file configs/COCO-PanopticSegmentation/u2seg_swin_tiny_800.yaml \
    TEST.PRECISE_BN.ENABLED False \
    DATASETS.TEST '()'
else
  echo "[INFO] Starting Swin-Tiny train smoke run..."
  python tools/train_net_swin.py \
    --num-gpus "${NUM_GPUS}" \
    --config-file configs/COCO-PanopticSegmentation/u2seg_swin_tiny_800.yaml \
    SOLVER.MAX_ITER "${MAX_ITER}" \
    TEST.PRECISE_BN.ENABLED False \
    DATASETS.TEST '()'
fi

echo "[DONE] Training command executed."
