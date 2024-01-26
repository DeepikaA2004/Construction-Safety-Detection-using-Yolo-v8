## pip install torch torchvision
## pip install opencv-python

import torch
import cv2
from pathlib import Path
from yolov8.models.experimental import attempt_load
from yolov8.utils.general import non_max_suppression, scale_coords
from yolov8.utils.torch_utils import select_device

def load_model(weights_path):
    # Load YOLOv8 model
    device = select_device('')
    model = attempt_load(weights_path, map_location=device)
    return model

def detect_safety(model, img_path, conf_threshold=0.5, iou_threshold=0.5):
    # Load image
    img0 = cv2.imread(img_path)

    # Preprocess image
    img = model.preprocess(img0)[0]

    # Run inference
    img = torch.from_numpy(img).to(model.device)
    pred = model(img)

    # Post-process detections
    pred = non_max_suppression(pred, conf_threshold, iou_threshold)[0]
    if pred is not None:
        # Rescale coordinates to original image size
        pred = scale_coords(img.shape[2:], pred[:, :4], img0.shape).round()

    return pred

def draw_boxes(img, boxes):
    for box in boxes:
        x, y, w, h = box.int()
        cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 2)
    return img

if __name__ == "__main__":
    weights_path = 'path/to/deepi/yolov8/weights.pt'
    img_path = 'path/to/deepi/image.jpg'

    # Load YOLOv8 model
    model = load_model(weights_path)

    # Detect safety in the image
    detections = detect_safety(model, img_path)

    # Draw bounding boxes on the image
    img = cv2.imread(img_path)
    img_with_boxes = draw_boxes(img.copy(), detections)

    # Display the result
    cv2.imshow('Safety Detection', img_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
