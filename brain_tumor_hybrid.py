import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision import models
from transformers import SwinForImageClassification, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, confusion_matrix
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# Set random seed
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Dataset paths
train_dir = 'Training'
test_dir = 'Testing'

# Data transformations (standardized to 224x224 for both models)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = ImageFolder(train_dir, transform=train_transforms)
test_dataset = ImageFolder(test_dir, transform=test_transforms)

# Data loaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# Class info
class_names = train_dataset.classes
num_classes = len(class_names)
print(f"Classes: {class_names}")
print(f"Training samples: {len(train_dataset)}")
print(f"Testing samples: {len(test_dataset)}")

# Class weights
class_counts = np.bincount([train_dataset.targets[i] for i in range(len(train_dataset))])
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
class_weights = class_weights / class_weights.sum() * len(class_counts)
class_weights = class_weights.to(device)
print(f"Class weights: {class_weights}")

# Loss and optimizer settings (shared)
criterion = nn.CrossEntropyLoss(weight=class_weights)
num_epochs = 10
num_training_steps = len(train_loader) * num_epochs

# Hybrid Model Definition
class HybridModel(nn.Module):
    def __init__(self, swin_path, resnet_path, num_classes):
        super(HybridModel, self).__init__()
        # Load Swin model from saved pretrained directory
        self.swin = SwinForImageClassification.from_pretrained(
            swin_path,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        # Load ResNet50
        self.resnet = models.resnet50(pretrained=False)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        self.resnet.load_state_dict(torch.load(resnet_path))
        self.swin.to(device)
        self.resnet.to(device)

    def forward(self, x):
        with torch.no_grad():
            swin_logits = self.swin(x).logits
            resnet_logits = self.resnet(x)
        # Average the logits
        hybrid_logits = (swin_logits + resnet_logits) / 2
        return hybrid_logits

# ============== Train Swin Model ==============
print("\n" + "="*50)
print("Training Swin Tiny Model")
print("="*50)

swin_model = SwinForImageClassification.from_pretrained(
    'microsoft/swin-tiny-patch4-window7-224',
    num_labels=num_classes,
    ignore_mismatched_sizes=True
)
swin_model.to(device)

swin_optimizer = AdamW(swin_model.parameters(), lr=2e-5, weight_decay=0.01)
swin_scheduler = get_cosine_schedule_with_warmup(swin_optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Tracking metrics for Swin
swin_train_losses, swin_test_losses = [], []
swin_train_accs, swin_test_accs = [], []
swin_train_precisions, swin_train_recalls = [], []
swin_test_precisions, swin_test_recalls = [], []

best_swin_acc = 0.0
swin_best_path = r'D:\brain_tumor_project_deep_learning\swin_tiny_best.pth'
swin_save_dir = r'D:\brain_tumor_project_deep_learning\swin_tiny_model'



for epoch in range(num_epochs):
    swin_model.train()
    total_train_loss = 0.0
    train_preds, train_labels = [], []
    
    for batch in tqdm(train_loader, desc=f"Swin Epoch {epoch+1}/{num_epochs}"):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        outputs = swin_model(images).logits
        loss = criterion(outputs, labels)
        swin_optimizer.zero_grad()
        loss.backward()
        swin_optimizer.step()
        swin_scheduler.step()
        total_train_loss += loss.item()
        _, preds = torch.max(outputs, dim=1)
        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())
    
    # Training metrics
    train_acc = accuracy_score(train_labels, train_preds)
    train_precision = precision_score(train_labels, train_preds, average='weighted', zero_division=0)
    train_recall = recall_score(train_labels, train_preds, average='weighted', zero_division=0)
    avg_train_loss = total_train_loss / len(train_loader)
    swin_train_losses.append(avg_train_loss)
    swin_train_accs.append(train_acc)
    swin_train_precisions.append(train_precision)
    swin_train_recalls.append(train_recall)
    
    # Evaluation
    swin_model.eval()
    total_test_loss = 0.0
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Swin Evaluating"):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            outputs = swin_model(images).logits
            loss = criterion(outputs, labels)
            total_test_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    # Test metrics
    test_acc = accuracy_score(test_labels, test_preds)
    test_precision = precision_score(test_labels, test_preds, average='weighted', zero_division=0)
    test_recall = recall_score(test_labels, test_preds, average='weighted', zero_division=0)
    avg_test_loss = total_test_loss / len(test_loader)
    swin_test_losses.append(avg_test_loss)
    swin_test_accs.append(test_acc)
    swin_test_precisions.append(test_precision)
    swin_test_recalls.append(test_recall)
    
    print(f"Swin Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, "
          f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_acc:.4f}, "
          f"Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}")
    
    if test_acc > best_swin_acc:
        best_swin_acc = test_acc
        torch.save(swin_model.state_dict(), swin_best_path)
        swin_model.save_pretrained(swin_save_dir)

# ============== Train ResNet50 Model ==============
print("\n" + "="*50)
print("Training ResNet50 Model")
print("="*50)

resnet_model = models.resnet50(pretrained=True)
num_features = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_features, num_classes)
resnet_model.to(device)

resnet_optimizer = AdamW(resnet_model.parameters(), lr=2e-5, weight_decay=0.01)
resnet_scheduler = get_cosine_schedule_with_warmup(resnet_optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Tracking metrics for ResNet
resnet_train_losses, resnet_test_losses = [], []
resnet_train_accs, resnet_test_accs = [], []
resnet_train_precisions, resnet_train_recalls = [], []
resnet_test_precisions, resnet_test_recalls = [], []

best_resnet_acc = 0.0
resnet_best_path = r'D:\brain_tumor_project_deep_learning\resnet50_best.pth'



for epoch in range(num_epochs):
    resnet_model.train()
    total_train_loss = 0.0
    train_preds, train_labels = [], []
    
    for batch in tqdm(train_loader, desc=f"ResNet Epoch {epoch+1}/{num_epochs}"):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        outputs = resnet_model(images)
        loss = criterion(outputs, labels)
        resnet_optimizer.zero_grad()
        loss.backward()
        resnet_optimizer.step()
        resnet_scheduler.step()
        total_train_loss += loss.item()
        _, preds = torch.max(outputs, dim=1)
        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())
    
    # Training metrics
    train_acc = accuracy_score(train_labels, train_preds)
    train_precision = precision_score(train_labels, train_preds, average='weighted', zero_division=0)
    train_recall = recall_score(train_labels, train_preds, average='weighted', zero_division=0)
    avg_train_loss = total_train_loss / len(train_loader)
    resnet_train_losses.append(avg_train_loss)
    resnet_train_accs.append(train_acc)
    resnet_train_precisions.append(train_precision)
    resnet_train_recalls.append(train_recall)
    
    # Evaluation
    resnet_model.eval()
    total_test_loss = 0.0
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="ResNet Evaluating"):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            outputs = resnet_model(images)
            loss = criterion(outputs, labels)
            total_test_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    # Test metrics
    test_acc = accuracy_score(test_labels, test_preds)
    test_precision = precision_score(test_labels, test_preds, average='weighted', zero_division=0)
    test_recall = recall_score(test_labels, test_preds, average='weighted', zero_division=0)
    avg_test_loss = total_test_loss / len(test_loader)
    resnet_test_losses.append(avg_test_loss)
    resnet_test_accs.append(test_acc)
    resnet_test_precisions.append(test_precision)
    resnet_test_recalls.append(test_recall)
    
    print(f"ResNet Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, "
          f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_acc:.4f}, "
          f"Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}")
    
    if test_acc > best_resnet_acc:
        best_resnet_acc = test_acc
        torch.save(resnet_model.state_dict(), resnet_best_path)

# ============== Hybrid Model Evaluation ==============
print("\n" + "="*50)
print("Evaluating Hybrid Model")
print("="*50)

# Create hybrid model using best saved models
hybrid_model = HybridModel(swin_save_dir, resnet_best_path, num_classes)
hybrid_model.eval()

# Final evaluation with hybrid
test_preds, test_labels = [], []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Hybrid Final Evaluation"):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        outputs = hybrid_model(images)
        _, preds = torch.max(outputs, dim=1)
        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

# Hybrid metrics
hybrid_acc = accuracy_score(test_labels, test_preds)
hybrid_precision = precision_score(test_labels, test_preds, average='weighted', zero_division=0)
hybrid_recall = recall_score(test_labels, test_preds, average='weighted', zero_division=0)

print(f"\nHybrid Final Test Accuracy: {hybrid_acc:.4f}")
print(f"Hybrid Final Test Precision: {hybrid_precision:.4f}")
print(f"Hybrid Final Test Recall: {hybrid_recall:.4f}")
print("\nHybrid Classification Report:")
print(classification_report(test_labels, test_preds, target_names=class_names))

# Compare with individual best
swin_final_acc = max(swin_test_accs)
resnet_final_acc = max(resnet_test_accs)
print(f"\nComparison:")
print(f"Swin Best Test Acc: {swin_final_acc:.4f}")
print(f"ResNet Best Test Acc: {resnet_final_acc:.4f}")
print(f"Hybrid Test Acc: {hybrid_acc:.4f}")

# Class distribution
class_counts_dict = dict(zip(class_names, class_counts))
print(f"\nClass Distribution: {class_counts_dict}")

# Hybrid Confusion matrix
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('Truth Label')
plt.title('Hybrid Confusion Matrix')
plt.savefig(r'D:\brain_tumor_project_deep_learning\hybrid_confusion_matrix.png')

plt.show()

# Visualize hybrid predictions
def denormalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    img = img.clone()
    for i in range(3):
        img[i] = img[i] * std[i] + mean[i]
    img = img.permute(1, 2, 0)
    img = img.clamp(0, 1)
    return img

classes = train_dataset.classes
images, labels = next(iter(test_loader))

with torch.no_grad():
    images_dev = images.to(device)
    outputs = hybrid_model(images_dev)
    _, preds = torch.max(outputs, dim=1)
    preds = preds.cpu().numpy()

plt.figure(figsize=(20, 20))
for i in range(min(16, len(images))):
    plt.subplot(4, 4, i + 1)
    img = denormalize(images[i]).numpy()
    plt.imshow(img)
    plt.title(classes[preds[i]], color='k', fontsize=15)
    plt.axis('off')
plt.tight_layout()

plt.savefig(r'D:\brain_tumor_project_deep_learning\hybrid_test_batch_visualization.png')


plt.show()

# Plot metrics for both models (separate figures)
plt.style.use('fivethirtyeight')

# Swin Metrics
plt.figure(figsize=(20, 12))
# Loss
index_loss = np.argmin(swin_test_losses)
plt.subplot(2, 2, 1)
plt.plot(range(1, num_epochs + 1), swin_train_losses, 'r', label='Training loss')
plt.plot(range(1, num_epochs + 1), swin_test_losses, 'g', label='Validation loss')
plt.scatter(index_loss + 1, swin_test_losses[index_loss], s=150, c='blue', label=f'Best epoch = {index_loss+1}')
plt.title('Swin Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Accuracy
index_acc = np.argmax(swin_test_accs)
plt.subplot(2, 2, 2)
plt.plot(range(1, num_epochs + 1), swin_train_accs, 'r', label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), swin_test_accs, 'g', label='Validation Accuracy')
plt.scatter(index_acc + 1, swin_test_accs[index_acc], s=150, c='blue', label=f'Best epoch = {index_acc+1}')
plt.title('Swin Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Precision
index_precision = np.argmax(swin_test_precisions)
plt.subplot(2, 2, 3)
plt.plot(range(1, num_epochs + 1), swin_train_precisions, 'r', label='Training Precision')
plt.plot(range(1, num_epochs + 1), swin_test_precisions, 'g', label='Validation Precision')
plt.scatter(index_precision + 1, swin_test_precisions[index_precision], s=150, c='blue', label=f'Best epoch = {index_precision+1}')
plt.title('Swin Training and Validation Precision')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.legend()
plt.grid(True)

# Recall
index_recall = np.argmax(swin_test_recalls)
plt.subplot(2, 2, 4)
plt.plot(range(1, num_epochs + 1), swin_train_recalls, 'r', label='Training Recall')
plt.plot(range(1, num_epochs + 1), swin_test_recalls, 'g', label='Validation Recall')
plt.scatter(index_recall + 1, swin_test_recalls[index_recall], s=150, c='blue', label=f'Best epoch = {index_recall+1}')
plt.title('Swin Training and Validation Recall')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()
plt.grid(True)

plt.suptitle('Swin Model Training Metrics Over Epochs', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(r'D:\brain_tumor_project_deep_learning\swin_metrics_plot.png')


plt.show()

# ResNet Metrics
plt.figure(figsize=(20, 12))
# Loss
index_loss = np.argmin(resnet_test_losses)
plt.subplot(2, 2, 1)
plt.plot(range(1, num_epochs + 1), resnet_train_losses, 'r', label='Training loss')
plt.plot(range(1, num_epochs + 1), resnet_test_losses, 'g', label='Validation loss')
plt.scatter(index_loss + 1, resnet_test_losses[index_loss], s=150, c='blue', label=f'Best epoch = {index_loss+1}')
plt.title('ResNet Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Accuracy
index_acc = np.argmax(resnet_test_accs)
plt.subplot(2, 2, 2)
plt.plot(range(1, num_epochs + 1), resnet_train_accs, 'r', label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), resnet_test_accs, 'g', label='Validation Accuracy')
plt.scatter(index_acc + 1, resnet_test_accs[index_acc], s=150, c='blue', label=f'Best epoch = {index_acc+1}')
plt.title('ResNet Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Precision
index_precision = np.argmax(resnet_test_precisions)
plt.subplot(2, 2, 3)
plt.plot(range(1, num_epochs + 1), resnet_train_precisions, 'r', label='Training Precision')
plt.plot(range(1, num_epochs + 1), resnet_test_precisions, 'g', label='Validation Precision')
plt.scatter(index_precision + 1, resnet_test_precisions[index_precision], s=150, c='blue', label=f'Best epoch = {index_precision+1}')
plt.title('ResNet Training and Validation Precision')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.legend()
plt.grid(True)

# Recall
index_recall = np.argmax(resnet_test_recalls)
plt.subplot(2, 2, 4)
plt.plot(range(1, num_epochs + 1), resnet_train_recalls, 'r', label='Training Recall')
plt.plot(range(1, num_epochs + 1), resnet_test_recalls, 'g', label='Validation Recall')
plt.scatter(index_recall + 1, resnet_test_recalls[index_recall], s=150, c='blue', label=f'Best epoch = {index_recall+1}')
plt.title('ResNet Training and Validation Recall')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()
plt.grid(True)

plt.suptitle('ResNet Model Training Metrics Over Epochs', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(r'D:\brain_tumor_project_deep_learning\resnet_metrics_plot.png')

plt.show()