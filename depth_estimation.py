import torch
import cv2
import numpy as np
import json
from torchvision import transforms

# Load MiDaS depth estimation model
depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
depth_model.eval()  # Set model to evaluation mode

# Define image preprocessing for MiDaS
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def estimate_depth(image_path, bounding_boxes):
    """Estimates the depth of detected pedestrians using MiDaS."""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return []

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Convert to tensor & normalize
    input_tensor = transform(image_rgb).unsqueeze(0)

    # Predict depth map
    with torch.no_grad():
        depth_map = depth_model(input_tensor)

    # Convert depth map to NumPy array
    depth_map = depth_map.squeeze().cpu().numpy()

    # Resize depth map to match original image size
    depth_map_resized = cv2.resize(depth_map, (image.shape[1], image.shape[0]))

    # Process bounding boxes
    pedestrian_depths = []
    for bbox in bounding_boxes:
        x1, y1, x2, y2 = map(int, bbox[:4])  # Get bounding box coordinates
        roi = depth_map_resized[y1:y2, x1:x2]  # Extract depth values in bounding box

        if roi.size == 0:  # Check for empty ROI
            avg_depth = float("nan")  # Assign NaN if depth cannot be calculated
        else:
            avg_depth = float(np.mean(roi))  # Compute average depth

        pedestrian_depths.append({
            "bounding_box": [x1, y1, x2, y2],
            "depth": avg_depth,
            "confidence": float(bbox[4])
        })

    return pedestrian_depths

def load_yolo_predictions(prediction_file):
    """Reads YOLO prediction file and extracts pedestrian bounding boxes."""
    bounding_boxes = []
    try:
        with open(prediction_file, "r") as file:
            for line in file:
                data = eval(line.replace("tensor", ""))  # Convert tensor string to list
                x1, y1, x2, y2, confidence, class_id = data

                # Class ID 0 is for pedestrians
                if int(class_id) == 0:
                    bounding_boxes.append([x1, y1, x2, y2, confidence])
    except FileNotFoundError:
        print(f"Error: {prediction_file} not found.")
        return []

    return bounding_boxes

# File paths
image_path = r"C:\Users\verin\OneDrive\Desktop\Cloud Computing\Project\Milestone3\TestImage.jpg"
prediction_file = "prediction_output.txt"

# Step 1: Load YOLO predictions
bounding_boxes = load_yolo_predictions(prediction_file)

# Step 2: Estimate pedestrian depths
depth_results = estimate_depth(image_path, bounding_boxes)

# Step 3: Save depth results to JSON
if depth_results:
    with open("depth_results.json", "w") as file:
        json.dump(depth_results, file, indent=4)
    print("Depth estimation completed! Results saved to depth_results.json")
else:
    print("No depth results were generated.")
