import os
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F
from ultralytics import YOLO

# Load your model file
model = YOLO(r'C:\Users\verin\OneDrive\Desktop\yolo11\myvenv\yolo11n.pt')  # Load your model file

# Perform prediction
results = model.predict(source=r'C:\Users\verin\OneDrive\Desktop\Cloud Computing\Project\Milestone3', conf=0.25)

# Open a file to write the output
output_file = 'prediction_output.txt'
#here is the format of the txt file tensor([x1, y1, x2, y2, confidence, class])


with open(output_file, 'w') as file:
    for r in results:
        for box in r.boxes.data:
            # Write each tensor output to the file
            file.write(f"tensor({box.tolist()})\n")  # .tolist() converts tensor to a list that can be written

# Output file name
print(f"Predictions have been written to {output_file}")
