
import json
from pathlib import Path

annotation_file = Path(
    r"C:\Users\ARUSH M VERNEKAR\OneDrive\Documents\D-Fine\dfine_dataset\annotations\instances_train.json"
)

with annotation_file.open("r", encoding="utf-8") as file:
    data = json.load(file)

print("Images:", len(data["images"]))
print("Bounding boxes:", len(data["annotations"]))
print("Categories:", data["categories"])
print("First 3 annotations:", data["annotations"][:3])
