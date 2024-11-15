import torch
import os
import numpy as np
import random
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import evaluate
from sklearn.metrics import roc_auc_score, confusion_matrix

# Argument parser for hyperparameters and configurations
parser = argparse.ArgumentParser(description="Train text-only model for classification")
parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
parser.add_argument('--wd', type=float, default=0.01, help='Weight decay')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train')
parser.add_argument('--input_dim', type=int, default=384, help='Dimension of the input features')
parser.add_argument('--hidden_dim1', type=int, default=256, help='Text features hidden dimension 1')
parser.add_argument('--hidden_dim2', type=int, default=128, help='Text features hidden dimension 2')
parser.add_argument('--parent_dir', type=str, required=True, help='Parent directory for fine-tuned VideoMAE features and SBERT features')
parser.add_argument('--epoch_num', type=int, default=4, help='Epoch number to specify file paths')
parser.add_argument('--label_dim', type=int, default=2, help='Number of output labels (binary by default)')
parser.add_argument('--run_name', type=str, required=True, help='Directory name to save models')
args = parser.parse_args()

# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Function to generate paths for loading features and labels
def generate_paths(parent_dir, epoch_num):
    """Generate paths for train/test features and labels based on the parent directory and epoch number."""
    train_features = os.path.join(parent_dir, f'train_features_epoch{epoch_num}.pt')
    test_features = os.path.join(parent_dir, f'test_features_epoch{epoch_num}.pt')
    train_labels = os.path.join(parent_dir, f'train_labels_epoch{epoch_num}.pt')
    test_labels = os.path.join(parent_dir, f'test_labels_epoch{epoch_num}.pt')
    return train_features, test_features, train_labels, test_labels

# Function to load features and labels
def load_features_and_labels(features_path, labels_path):
    """Load features and labels from the provided paths."""
    features = torch.load(features_path)
    labels = torch.load(labels_path)
    dataset = TensorDataset(features, torch.from_numpy(labels))
    return dataset

# Define a general text-based model with two hidden layers
class TextModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(TextModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.output = nn.Linear(hidden_dim2, output_dim)

    def forward(self, features):
        x = F.relu(self.fc1(features))
        x = F.relu(self.fc2(x))
        logits = self.output(x)
        return logits

# Function to compute metrics for predictions
def compute_metrics(predictions, labels):
    """Compute classification metrics: accuracy, precision, recall, F1, AUROC, and confusion matrix."""
    probabilities = F.softmax(torch.tensor(predictions), dim=1).numpy()
    predicted_labels = np.argmax(probabilities, axis=1)
    accuracy = accuracy_metric.compute(predictions=predicted_labels, references=labels)
    precision = precision_metric.compute(predictions=predicted_labels, references=labels, average='binary')
    recall = recall_metric.compute(predictions=predicted_labels, references=labels, average='binary')
    f1 = f1_metric.compute(predictions=predicted_labels, references=labels, average='binary')
    auroc = roc_auc_score(labels, probabilities[:, 1])
    tn, fp, fn, tp = confusion_matrix(labels, predicted_labels).ravel()
    metrics = {
        'accuracy': accuracy['accuracy'],
        'precision': precision['precision'],
        'recall': recall['recall'],
        'f1': f1['f1'],
        'auroc': auroc,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp
    }
    return metrics

# Training function for one epoch
def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    for batch in tqdm(dataloader, desc="Training", leave=False):
        optimizer.zero_grad()
        features, labels = batch
        features, labels = features.to(device), labels.to(device)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        all_predictions.append(logits.cpu().detach().numpy())
        all_labels.append(labels.cpu().detach().numpy())
    
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    metrics = compute_metrics(all_predictions, all_labels)
    
    return total_loss / len(dataloader), metrics

# Test function for one epoch
def test_epoch(model, dataloader, criterion, device):
    """Evaluate the model for one epoch."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing", leave=False):
            features, labels = batch
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            all_predictions.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    metrics = compute_metrics(all_predictions, all_labels)
    
    return total_loss / len(dataloader), metrics

# Ensure the directory to save models exists
os.makedirs(args.run_name, exist_ok=True)

# Generate paths for the dataset
train_features_path, test_features_path, train_labels_path, test_labels_path = generate_paths(args.parent_dir, args.epoch_num)

# Load datasets
train_dataset = load_features_and_labels(train_features_path, train_labels_path)
test_dataset = load_features_and_labels(test_features_path, test_labels_path)

# Create DataLoader for training and testing
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

# Initialize the model
model = TextModel(
    input_dim=args.input_dim,
    hidden_dim1=args.hidden_dim1,
    hidden_dim2=args.hidden_dim2,
    output_dim=args.label_dim
).to(device)

# Define optimizer, scheduler, and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

# Initialize metrics
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")


# Training loop
for epoch in range(args.num_epochs):
    print(f"Epoch {epoch + 1}/{args.num_epochs}")
    train_loss, train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
    test_loss, test_metrics = test_epoch(model, test_loader, criterion, device)
    scheduler.step(train_metrics["accuracy"]) 
    
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Metrics: {train_metrics}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Metrics: {test_metrics}")

    # Save the model after each epoch
    torch.save(model.state_dict(), os.path.join(args.run_name, f"model_epoch_{epoch + 1}.pth"))

# Final evaluation on the test set
final_test_metrics = test_epoch(model, test_loader, criterion, device)
print("Final Test Metrics:")
print(final_test_metrics)
