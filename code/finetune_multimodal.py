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

# Argument parser for hyperparameters
parser = argparse.ArgumentParser(description="Evaluate combined model for video and text features")
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--wd', type=float, default=0.01, help='Weight decay')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs to train')
parser.add_argument('--vid_hidden_dim1', type=int, default=256, help='First video features hidden dimension')
parser.add_argument('--vid_hidden_dim2', type=int, default=128, help='Second video features hidden dimension')
parser.add_argument('--text_hidden_dim1', type=int, default=128, help='First text features hidden dimension')
parser.add_argument('--text_hidden_dim2', type=int, default=64, help='Second text features hidden dimension')
parser.add_argument('--comb_hidden_dim1', type=int, default=256, help='First combined features hidden dimension')
parser.add_argument('--comb_hidden_dim2', type=int, default=128, help='Second combined features hidden dimension')
parser.add_argument('--parent_dir', type=str, required=True, help='Parent directory for fine-tuned VideoMAE features and SBERT features')
parser.add_argument('--epoch_num', type=int, default=4, help='Epoch number to specify file paths')
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

# Function to generate paths based on the parent directory and epoch number
def generate_paths(parent_dir, epoch_num):
    """Generate paths for video features, text features, and labels based on directory and epoch number."""
    video_features = os.path.join(parent_dir, f'test_video_epoch{epoch_num}.pt')
    text_features = os.path.join(parent_dir, f'test_text_epoch{epoch_num}.pt')
    labels = os.path.join(parent_dir, f'test_labels_epoch{epoch_num}.pt')
    return video_features, text_features, labels

# Function to load the dataset
def load_features_and_labels(video_features_path, text_features_path, labels_path):
    """Load the video features, text features, and labels from the given paths."""
    video_features = torch.load(video_features_path)
    text_features = torch.load(text_features_path)
    labels = torch.load(labels_path)
    dataset = TensorDataset(video_features, text_features, torch.tensor(labels, dtype=torch.long))
    return dataset

# Function to compute metrics
def compute_metrics(predictions, labels):
    """Compute classification metrics such as accuracy, precision, recall, F1, AUROC, and confusion matrix."""
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

# Define the multimodal model
class MultimodalModel(nn.Module):
    """A multimodal model that processes video and text features separately and then merges them."""
    def __init__(self, vid_hidden_dim1, vid_hidden_dim2, text_hidden_dim1, text_hidden_dim2, comb_hidden_dim1, comb_hidden_dim2, num_labels):
        super(MultimodalModel, self).__init__()
        self.video_fc1 = nn.Linear(768, vid_hidden_dim1)
        self.video_fc2 = nn.Linear(vid_hidden_dim1, vid_hidden_dim2)
        self.text_fc1 = nn.Linear(384, text_hidden_dim1)
        self.text_fc2 = nn.Linear(text_hidden_dim1, text_hidden_dim2)
        self.comb_fc1 = nn.Linear(vid_hidden_dim2 + text_hidden_dim2, comb_hidden_dim1)
        self.comb_fc2 = nn.Linear(comb_hidden_dim1, comb_hidden_dim2)
        self.output = nn.Linear(comb_hidden_dim2, num_labels)
        
    def forward(self, video_features, text_features):
        # Video path
        video_out = F.relu(self.video_fc1(video_features))
        video_out = F.relu(self.video_fc2(video_out))
        
        # Text path
        text_out = F.relu(self.text_fc1(text_features))
        text_out = F.relu(self.text_fc2(text_out))
        
        # Combine
        combined = torch.cat((video_out, text_out), dim=1)
        combined = F.relu(self.comb_fc1(combined))
        combined = F.relu(self.comb_fc2(combined))
        
        # Output
        logits = self.output(combined)
        return logits

# Training function
def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train the model for a single epoch."""
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    for batch in tqdm(dataloader, desc="Training", leave=False):
        optimizer.zero_grad()
        video_features, text_features, labels = batch
        video_features, text_features, labels = video_features.to(device), text_features.to(device), labels.to(device)
        
        logits = model(video_features, text_features)
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

# Evaluation function
def evaluate_epoch(model, dataloader, criterion, device):
    """Evaluate the model for a single epoch."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            video_features, text_features, labels = batch
            video_features, text_features, labels = video_features.to(device), text_features.to(device), labels.to(device)
            
            logits = model(video_features, text_features)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            all_predictions.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    metrics = compute_metrics(all_predictions, all_labels)
    
    return total_loss / len(dataloader), metrics


# Load the test dataset
train_features_path, test_features_path, train_labels_path, test_labels_path = generate_paths(args.parent_dir, args.epoch_num)

# Create the datasets
train_dataset = load_features_and_labels(train_features_path, train_labels_path)
test_dataset = load_features_and_labels(test_features_path, test_labels_path)

# Initialize the model
num_labels = 2  # Binary classification
model = MultimodalModel(
    vid_hidden_dim1=args.vid_hidden_dim1, 
    vid_hidden_dim2=args.vid_hidden_dim2, 
    text_hidden_dim1=args.text_hidden_dim1, 
    text_hidden_dim2=args.text_hidden_dim2, 
    comb_hidden_dim1=args.comb_hidden_dim1, 
    comb_hidden_dim2=args.comb_hidden_dim2, 
    num_labels=num_labels
).to(device)

# Define optimizer, scheduler, and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

# Initialize evaluation metrics
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")


save_path = args.run_name

# Training loop
for epoch in range(args.num_epochs):
    print(f"Epoch {epoch + 1}/{args.num_epochs}")
    train_loss, train_metrics = train_epoch(model, train_dataset, optimizer, criterion, device)
    test_loss, test_metrics = evaluate_epoch(model, test_dataset, criterion, device)
    scheduler.step(train_metrics["accuracy"])
    
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Metrics: {train_metrics}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Metrics: {test_metrics}")

    # Save the model after each epoch
    torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{epoch + 1}.pth"))

# Evaluate the final model on the test set
final_test_metrics = evaluate_epoch(model, test_dataset, criterion, device)
print("Final Test Metrics:")
print(final_test_metrics)

# Save the final model
model_save_path = os.path.join(save_path, f"final_model.pth")
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}.")
