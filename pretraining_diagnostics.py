
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

ROOT = Path(__file__).resolve().parent
DATA = Path(
    r"C:\Users\ARUSH M VERNEKAR\OneDrive\Documents\D-Fine\dfine_dataset"
)
WEIGHTS = ROOT / "weights" / "dfine_s_obj365.pth"

ANNOTATIONS = {
    "train": DATA / "annotations" / "instances_train.json",
    "valid": DATA / "annotations" / "instances_val.json",
    "test": DATA / "annotations" / "instances_test.json",
}

CONFIGS = [
    ROOT / "configs" / "dataset" / "custom_detection.yml",
    ROOT / "configs" / "dfine" / "custom"
          / "dfine_hgnetv2_s_custom.yml",
]

IMAGE_DIRS = {
    "train": DATA / "images" / "train",
    "valid": DATA / "images" / "valid",
    "test": DATA / "images" / "test",
}


def section(name):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)


def check_dataset():
    section("1. DATASET AND COCO ANNOTATIONS")

    for split, ann_path in ANNOTATIONS.items():
        print(f"\n--- {split.upper()} ---")
        print("Annotation file exists:", ann_path.exists())

        if not ann_path.exists():
            continue

        try:
            with ann_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            images = data.get("images", [])
            annotations = data.get("annotations", [])
            categories = data.get("categories", [])
            image_ids = {im["id"] for im in images}
            image_names = [im.get("file_name", "") for im in images]

            print("Images in JSON:", len(images))
            print("Annotations:", len(annotations))
            print("Categories:", categories)
            print("Duplicate image IDs:",
                  len(image_ids) != len(images))
            print("Duplicate filenames:",
                  len(set(image_names)) != len(image_names))
            print("Images without annotations:",
                  len(images) - len({
                      a.get("image_id") for a in annotations
                  }))

            category_ids = {c["id"] for c in categories}
            annotation_category_ids = {
                a.get("category_id") for a in annotations
            }
            print("Annotation category IDs:",
                  sorted(annotation_category_ids, key=str))
            print("Category IDs missing from category list:",
                  sorted(
                      annotation_category_ids - category_ids,
                      key=str
                  ))

            invalid_image_refs = sum(
                a.get("image_id") not in image_ids
                for a in annotations
            )
            print("Annotations with unknown image IDs:",
                  invalid_image_refs)

            invalid_boxes = []
            for a in annotations:
                bbox = a.get("bbox")
                if (
                    not isinstance(bbox, list)
                    or len(bbox) != 4
                    or not all(
                        isinstance(v, (int, float)) for v in bbox
                    )
                    or bbox[2] <= 0
                    or bbox[3] <= 0
                    or not all(
                        isinstance(v, (int, float))
                        for v in bbox
                    )
                ):
                    invalid_boxes.append(a.get("id"))
                    continue

                image = next(
                    (im for im in images
                     if im.get("id") == a.get("image_id")),
                    None
                )
                if image:
                    x, y, w, h = bbox
                    iw, ih = image.get("width", 0), image.get("height", 0)
                    if (
                        x < -0.01 or y < -0.01
                        or x + w > iw + 1
                        or y + h > ih + 1
                    ):
                        invalid_boxes.append(a.get("id"))

            print("Invalid/out-of-image boxes:", len(invalid_boxes))
            if invalid_boxes[:10]:
                print("First affected annotation IDs:",
                      invalid_boxes[:10])

            missing_images = []
            image_folder = IMAGE_DIRS[split]
            for name in image_names:
                if not (image_folder / name).is_file():
                    missing_images.append(name)

            print("Image folder exists:", image_folder.is_dir())
            print("Referenced image files missing:",
                  len(missing_images))
            if missing_images[:5]:
                print("First missing filenames:", missing_images[:5])

        except Exception as exc:
            print("ERROR reading split:", repr(exc))


def check_yamls():
    section("2. YAML CONFIGURATIONS")

    for path in CONFIGS:
        print(f"\n--- {path.relative_to(ROOT)} ---")
        print("Exists:", path.exists())

        if not path.exists():
            continue

        if yaml is None:
            print("PyYAML unavailable; syntax not checked.")
            print("Install only if needed: pip install pyyaml")
            continue

        try:
            with path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)

            print("YAML parses successfully:", isinstance(cfg, dict))
            print("Top-level keys:",
                  list(cfg.keys()) if isinstance(cfg, dict) else cfg)

            if isinstance(cfg, dict):
                for key in (
                    "num_classes", "remap_mscoco_category",
                    "epochs", "use_focal_loss"
                ):
                    if key in cfg:
                        print(f"{key}:", cfg[key])

                for key in ("train_dataloader", "val_dataloader"):
                    loader = cfg.get(key)
                    if isinstance(loader, dict):
                        print(f"{key}.total_batch_size:",
                              loader.get("total_batch_size"))
                        ds = loader.get("dataset", {})
                        if isinstance(ds, dict):
                            print(f"{key}.dataset.img_folder:",
                                  ds.get("img_folder"))
                            print(f"{key}.dataset.ann_file:",
                                  ds.get("ann_file"))
                            print(f"{key}.dataset.type:",
                                  ds.get("type"))

                print("Full YAML is not printed; inspect locally if needed.")

        except Exception as exc:
            print("YAML ERROR:", repr(exc))


def check_mapping():
    section("3. D-FINE CATEGORY MAPPING")

    source = ROOT / "src" / "data" / "dataset" / "coco_dataset.py"
    print("Mapping source exists:", source.exists())

    if source.exists():
        lines = source.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()

        for start, end in ((35, 65), (260, 295)):
            print(f"\n--- coco_dataset.py lines {start}-{end} ---")
            for i in range(start - 1, min(end, len(lines))):
                print(f"{i + 1}: {lines[i]}")

    post = ROOT / "src" / "zoo" / "dfine" / "postprocessor.py"
    if post.exists():
        lines = post.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        print("\n--- postprocessor.py lines 65-85 ---")
        for i in range(64, min(85, len(lines))):
            print(f"{i + 1}: {lines[i]}")

    engine = ROOT / "src" / "solver" / "det_engine.py"
    if engine.exists():
        lines = engine.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        print("\n--- det_engine.py lines 210-240 ---")
        for i in range(209, min(240, len(lines))):
            print(f"{i + 1}: {lines[i]}")


def check_environment():
    section("4. PYTHON, TORCH, CUDA AND CHECKPOINT")

    print("Python executable:", sys.executable)
    print("Python version:", sys.version)

    print("Checkpoint exists:", WEIGHTS.is_file())
    if WEIGHTS.is_file():
        print("Checkpoint size (bytes):", WEIGHTS.stat().st_size)

    try:
        import torch
        import torchvision

        print("PyTorch:", torch.__version__)
        print("Torchvision:", torchvision.__version__)
        print("CUDA available:", torch.cuda.is_available())

        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
            print("GPU memory (GiB):",
                  round(
                      torch.cuda.get_device_properties(0).total_memory
                      / (1024 ** 3), 2
                  ))
    except Exception as exc:
        print("Torch check failed:", repr(exc))


if __name__ == "__main__":
    check_dataset()
    check_yamls()
    check_mapping()
    check_environment()

    section("DIAGNOSTICS COMPLETE")
    print("This script did not modify files or start training.")
