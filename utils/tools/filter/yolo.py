from ultralytics import YOLO
from facenet_pytorch import MTCNN
import json
from torchvision.transforms.functional import to_tensor
from PIL import Image
from tqdm import tqdm
import os
import logging
import shutil


model = YOLO("yolov8n.pt")
face_detector = MTCNN(select_largest=False, device="cuda")
logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("facenet_pytorch").setLevel(logging.ERROR)


def detect_human_and_size(image_path):
    image = Image.open(image_path)
    results = model(
        source=image_path, imgsz=1024, device="cuda", conf=0.44, augment=False
    )

    boxes = []
    for box in json.loads(results[0].tojson()):
        if box["name"] == "person":
            boxes.append(box["box"])

    img_width, img_height = image.size

    for box in boxes:
        bbox_width = box["x2"] - box["x1"]
        bbox_height = box["y2"] - box["y1"]
        bbox_area = bbox_width * bbox_height
        img_area = img_width * img_height

        bbox_percentage = (bbox_area / img_area) * 100
        if bbox_percentage >= 15:
            crop_image = image.crop((box["x1"], box["y1"], box["x2"], box["y2"]))
            try:
                face_detected, _ = face_detector.detect(crop_image)
                if face_detected is not None:
                    return True
            except:
                print("Face Error")
                continue
    return False


def process_images(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for filename in tqdm(os.listdir(source_dir), desc="Filtering"):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(source_dir, filename)

            if detect_human_and_size(image_path):
                shutil.copy(image_path, target_dir)
