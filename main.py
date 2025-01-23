    # -*- coding: utf-8 -*-

# Cell 1: Install dependencies if not already installed
!pip install torch torchvision scikit-learn
# Cell 2: Import libraries
import os
import torchvision.transforms as transforms
import torch
from PIL import Image
from torchvision.transforms import functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
import torch.optim as optim

!mkdir Brain_Data_Organised_Extracted

!unzip "/content/Brain_Data_Organised-20241125T044401Z-001.zip" -d "/content/Brain_Data_Organised_Extracted"

# Cell 3: Define the dataset class
class StrokeDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.classes = {"Normal": 0, "Stroke": 1}
        self.images = []
        self.annotations = []

        for label, class_id in self.classes.items():
            class_dir = os.path.join(root, label)
            for image in os.listdir(class_dir):
                if image.endswith(('.jpg', '.png')):
                    self.images.append(os.path.join(class_dir, image))
                    self.annotations.append({
                        "boxes": [[0, 0, 224, 224]],  # Whole image treated as the region
                        "labels": [class_id]
                    })

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        target = self.annotations[idx]

        # Convert annotations to tensors
        target = {
            "boxes": torch.tensor(target["boxes"], dtype=torch.float32),
            "labels": torch.tensor(target["labels"], dtype=torch.int64),
        }

        if self.transform:
            image = self.transform(image)

        return image, target

# Cell 4: Split dataset and define DataLoaders
# Path to your dataset folder
dataset_path = "/content/Brain_Data_Organised_Extracted/Brain_Data_Organised"

# Define dataset
dataset = StrokeDataset(root=dataset_path)

# Split dataset
train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)

# Define train and validation datasets using the indices
train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Cell 5: Define the Faster R-CNN model
# Load pre-trained Faster R-CNN
model = fasterrcnn_resnet50_fpn(pretrained=True)

# Modify the classifier for 2 classes (background + stroke/normal)
num_classes = 2  # background + 2 categories
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Cell 6: Define the training loop with progress bar for Colab
from tqdm.notebook import tqdm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop with progress bar
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    # Display progress bar for batches in the current epoch
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch", leave=False)

    for images, targets in progress_bar:
        images = [transforms.ToTensor()(img).to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backpropagation
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

        # Update the progress bar with the current loss
        progress_bar.set_postfix({"Batch Loss": losses.item()})

    print(f"Epoch {epoch+1}/{num_epochs} completed. Average Loss: {total_loss / len(train_loader):.4f}")

    # Optional: Validation (optional, add later if needed)
    model.eval()
    # Add validation logic here, if necessary

# Cell 7: Save the trained model
torch.save(model.state_dict(), "faster_rcnn_stroke_model.pth")

# Install matplotlib and scikit-learn if not already installed
!pip install matplotlib scikit-learn

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Load the trained model
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Reinitialize the model with the same configuration
model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=2)  # Adjust `num_classes` as needed
model.load_state_dict(torch.load("faster_rcnn_stroke_model.pth"))
model.to(device)
model.eval()  # Switch to evaluation mode

# Cell: Calculate metrics
y_true = []  # True labels
y_pred = []  # Predicted labels

model.eval()
with torch.no_grad():
    for images, targets in val_loader:
        images = [transforms.ToTensor()(img).to(device) for img in images]
        outputs = model(images)

        for output, target in zip(outputs, targets):
            pred_label = output['labels'][output['scores'] > 0.5].cpu().numpy()
            true_label = target['labels'].cpu().numpy()

            # Use the first predicted box if available, else classify as background
            if len(pred_label) > 0:
                y_pred.append(pred_label[0])
            else:
                y_pred.append(0)  # Assume background if no detection

            y_true.append(true_label[0])

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='binary')
recall = recall_score(y_true, y_pred, average='binary')
f1 = f1_score(y_true, y_pred, average='binary')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Cell: Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Stroke"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Cell: Plot training loss and accuracy
epochs = range(1, num_epochs + 1)
training_loss= [0.0252, 0.0151, 0.0139, 0.0153, 0.0168, 0.0110, 0.0111, 0.0107, 0.0115, 0.0114]
training_accuracy=[0.4139, 0.4139, 0.4139, 0.4139, 0.4139, 0.4139, 0.4139, 0.4139, 0.4139, 0.4139]

# Plot training loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, training_loss, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()

# Plot training accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, training_accuracy, label="Training Accuracy", color="orange")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

# Cell 8: Inference
def classify_image(model, image_path):
    model.eval()
    image_path="/content/brain.jpeg"
    image = Image.open(image_path).convert("RGB")
    transform = T.ToTensor()
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        boxes = output[0]['boxes']
        labels = output[0]['labels']
        scores = output[0]['scores']

        for i in range(len(scores)):
            if scores[i] > 0.5:  # Confidence threshold
                label = "Stroke" if labels[i] == 1 else "Normal"
                print(f"Prediction: {label}, Score: {scores[i].item()}")

# Load the trained model
model.load_state_dict(torch.load("faster_rcnn_stroke_model.pth"))
model.to(device)

# Test on a new image
classify_image(model, "path/to/test/image.jpg")

# Install matplotlib and scikit-learn if not already installed
!pip install matplotlib scikit-learn

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Cell: Modified Training Loop
training_loss = []  # List to store training loss for each epoch
training_accuracy = []  # List to store training accuracy for each epoch

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    # Progress bar
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch", leave=False)

    for images, targets in progress_bar:
        #images = [img.to(device) for img in images]
        images = [transforms.ToTensor()(img).to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backpropagation
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

        # Track predictions
        model.eval()  # Switch to evaluation mode for predictions
        with torch.no_grad():
            outputs = model(images)
            for output, target in zip(outputs, targets):
                # Use the label with the highest confidence
                pred_label = output['labels'][output['scores'] > 0.5].cpu().numpy()
                true_label = target['labels'].cpu().numpy()
                if len(pred_label) > 0:  # If predictions exist
                    correct_predictions += (pred_label[0] == true_label[0])  # First box
                total_samples += 1

        model.train()  # Switch back to training mode

    training_loss.append(total_loss / len(train_loader))
    training_accuracy.append(correct_predictions / total_samples)

    print(f"Epoch {epoch+1} - Loss: {training_loss[-1]:.4f}, Accuracy: {training_accuracy[-1]:.4f}")
    
