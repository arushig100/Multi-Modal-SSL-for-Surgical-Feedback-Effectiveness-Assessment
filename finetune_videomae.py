import torch
import os
import numpy as np
import random
from torch.utils.data import DataLoader, random_split, Subset
from transformers import VideoMAEModel
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import evaluate
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.utils import resample
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import UniformTemporalSubsample
from torch.utils.data import Dataset

# Argument parser for hyperparameters
parser = argparse.ArgumentParser(description="Train video model for feature prediction")
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--wd', type=float, default=0.01, help='Weight decay')
parser.add_argument('--mean', type=bool, default=True, help='Use mean vs full features')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train')
parser.add_argument('--predict_feature', type=str, default="trainee_behavior_change", help='Feature to predict')
parser.add_argument('--dim1', type=int, default=512, help='Dimension 1 size')
parser.add_argument('--dim2', type=int, default=256, help='Dimension 2 size')

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

# Dataset class
class VideoTextDataset(Dataset):
    def __init__(self, data, video_dir, processor, sbert_model, text_col="Dialogue", video_col="cvid",
                 num_frames=16, features_to_save=None, trigger=False):
        """
        Args:
            data (DataFrame): Data with video and text information.
            video_dir (str): Directory path for videos.
            processor (VideoMAEImageProcessor): Processor for video frames.
            sbert_model (SentenceTransformer): Pretrained SBERT model for text embedding.
            text_col (str): Column name for text data in DataFrame.
            video_col (str): Column name for video filenames in DataFrame.
            num_frames (int): Number of video frames to sample.
            features_to_save (list): List of features to extract from the data (i.e., trainee behavior change labels)
            trigger (bool): Whether to use alternative video directories for triggers.
        """
        self.data = data
        self.video_dir = video_dir
        self.processor = processor
        self.sbert_model = sbert_model
        self.text_col = text_col  # Generalized text column
        self.video_col = video_col  # Generalized video column
        self.num_frames = num_frames
        self.subsampler = UniformTemporalSubsample(self.num_frames)
        self.trigger = trigger
        self.features_to_save = features_to_save or []
        self.videos = []
        self.text_features = []
        self.extra_features = []
        self.load_data()

    def load_data(self):
        # Load and process text features using SBERT
        texts = self.data[self.text_col].tolist()
        self.text_features = self.sbert_model.encode(texts, convert_to_tensor=True, device=device)

        for _, row in self.data.iterrows():
            video_name = row[self.video_col]
            video_path = os.path.join(self.video_dir, video_name)

            try:
                video = EncodedVideo.from_path(video_path)
                video_duration = video.duration
                video_data = video.get_clip(start_sec=0, end_sec=video_duration)["video"]
                subsampled_frames = self.subsampler(video_data)
                video_data_np = subsampled_frames.numpy().transpose(1, 0, 2, 3)
                video_tensor = self.processor([video_data_np], return_tensors="pt")["pixel_values"].squeeze()
            except Exception as e:
                print(f"Error loading video {video_name}: {e}")
                video_tensor = torch.zeros((self.num_frames, 3, 224, 224))  # Dummy tensor if loading fails

            self.videos.append(video_tensor)

            # Extract any additional features to save
            extra_feature_values = {feature: row[feature] for feature in self.features_to_save}
            self.extra_features.append(extra_feature_values)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_tensor = self.videos[idx]
        text_tensor = self.text_features[idx]
        extra_features = self.extra_features[idx]
        return {"video": video_tensor, "text": text_tensor, **extra_features}


# Function to balance the dataset
def balance_classes(dataset, predict_feature):
    """
    Upsample the minority class in the dataset to balance class distribution.
    """
    labels = [sample[predict_feature] for sample in dataset]
    indices_class_0 = [i for i, label in enumerate(labels) if label == 0]
    indices_class_1 = [i for i, label in enumerate(labels) if label == 1]

    if len(indices_class_0) > len(indices_class_1):
        minority_indices = indices_class_1
        majority_class_size = len(indices_class_0)
    else:
        minority_indices = indices_class_0
        majority_class_size = len(indices_class_1)

    additional_upsampled_indices = resample(minority_indices, replace=True, 
                                            n_samples=(majority_class_size - len(minority_indices)), 
                                            random_state=seed)
    balanced_indices = minority_indices + additional_upsampled_indices + (indices_class_0 if len(indices_class_0) > len(indices_class_1) else indices_class_1)
    return Subset(dataset, balanced_indices)


# Fine-tuning Model for VideoMAE
class VideoMAEFineTuningModel(nn.Module):
    """
    Fine-tuning model for VideoMAE with fully connected layers for classification.
    """
    def __init__(self, video_model, video_hidden_dim, video_hidden_dim2, num_labels):
        super(VideoMAEFineTuningModel, self).__init__()
        self.video_model = video_model
        self.video_fc = nn.Linear(video_model.config.hidden_size, video_hidden_dim)
        self.video_fc2 = nn.Linear(video_hidden_dim, video_hidden_dim2)
        self.output = nn.Linear(video_hidden_dim2, num_labels)

    def forward(self, video):
        # Extract video features
        with torch.no_grad():
            video_outputs = self.video_model(video)
            video_features = video_outputs[0].mean(1) if args.mean else video_outputs[0][:, 0] 
        # Pass through fully connected layers
        video_features2 = F.relu(self.video_fc(video_features))
        video_features3 = F.relu(self.video_fc2(video_features2))
        logits = self.output(video_features3)
        return logits, video_features


# Metrics computation function
def compute_metrics(predictions, labels):
    """
    Compute classification metrics like accuracy, precision, recall, f1, AUROC, and confusion matrix.
    """
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


# Save features function for each epoch
def save_features_epoch(features, epoch, dataset_type, feature_type, prefix, run_name):
    """
    Save extracted features from video or text for a given epoch.
    """
    save_path = os.path.join(prefix + run_name, f"{dataset_type}_{feature_type}_epoch{epoch}.pt")
    torch.save(features, save_path)


# Save labels function for each epoch
def save_labels_epoch(labels, epoch, dataset_type, prefix, run_name):
    """
    Save labels for each epoch.
    """
    save_path = os.path.join(prefix + run_name, f"{dataset_type}_labels_epoch{epoch}.pt")
    torch.save(labels, save_path)


# Training function for a single epoch
def train_epoch(model, dataloader, optimizer, criterion, device, epoch, prefix, run_name):
    """
    Train the model for a single epoch.
    """
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    all_video_features = []
    all_text_features = []
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        optimizer.zero_grad()
        video = batch["video"].to(device)
        labels = batch[args.predict_feature].type(torch.LongTensor).to(device)
        text_features = batch["text"]

        logits, video_features = model(video)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        all_predictions.append(logits.cpu().detach().numpy())
        all_labels.append(labels.cpu().detach().numpy())
        all_video_features.append(video_features.cpu().detach())  
        all_text_features.append(text_features.cpu().detach())  

    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    metrics = compute_metrics(all_predictions, all_labels)
    all_video_features = torch.cat(all_video_features)  
    all_text_features = torch.cat(all_text_features)  
    
    save_features_epoch(all_video_features, epoch, 'train', 'video', prefix, run_name)
    save_features_epoch(all_text_features, epoch, 'train', 'text', prefix, run_name)
    save_labels_epoch(all_labels, epoch, 'train', prefix, run_name)

    return total_loss / len(dataloader), metrics


# Test function for a single epoch (replacing validation)
def test_epoch(model, dataloader, criterion, device, epoch, prefix, run_name):
    """
    Evaluate the model for a single epoch.
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    all_video_features = []
    all_text_features = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Testing", leave=False)):
            video = batch["video"].to(device)
            labels = batch[args.predict_feature].type(torch.LongTensor).to(device)
            text_features = batch["text"]

            logits, video_features = model(video)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            all_predictions.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_video_features.append(video_features.cpu().detach())  
            all_text_features.append(text_features.cpu().detach())  

    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    metrics = compute_metrics(all_predictions, all_labels)
    all_video_features = torch.cat(all_video_features)
    all_text_features = torch.cat(all_text_features)
    
    save_features_epoch(all_video_features, epoch, 'test', 'video', prefix, run_name)
    save_features_epoch(all_text_features, epoch, 'test', 'text', prefix, run_name)
    save_labels_epoch(all_labels, epoch, 'test', prefix, run_name)

    return total_loss / len(dataloader), metrics


# Loading the dataset
dataset_path = "dataset_path" 
dataset = torch.load(dataset_path)

# Split the dataset (80/20 train-test split)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed))

# Balance the classes
train_dataset = balance_classes(train_dataset, args.predict_feature)
test_dataset = balance_classes(test_dataset, args.predict_feature)

# DataLoader for training and testing
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, generator=torch.Generator().manual_seed(seed))

# Initialize VideoMAE model
videomae_model_path = "MCG-NJU/videomae-base"
video_model = VideoMAEModel.from_pretrained(videomae_model_path, output_hidden_states=True).to(device)
video_hidden_dim = args.dim1
video_hidden_dim2 = args.dim2
num_labels = 2  # Binary classification

# Initialize fine-tuning model
model = VideoMAEFineTuningModel(video_model, video_hidden_dim, video_hidden_dim2, num_labels).to(device)

# Optimizer, scheduler, and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

# Metrics loaders
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

# Define paths and experiment names
run_path = "runs/"  # Path where you want to save the runs
experiment_name = "experiment_name"  # Name for this experiment

# Ensure the directory exists
os.makedirs(os.path.join(run_path, experiment_name), exist_ok=True)

# Training loop
for epoch in range(args.num_epochs):
    print(f"Epoch {epoch + 1}/{args.num_epochs}")
    train_loss, train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, epoch, run_path, experiment_name)
    test_loss, test_metrics = test_epoch(model, test_loader, criterion, device, epoch, run_path, experiment_name)
    scheduler.step(train_metrics["accuracy"])

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Metrics: {train_metrics}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Metrics: {test_metrics}")

    # Save model state
    torch.save(model.state_dict(), os.path.join(run_path, experiment_name, f"model_epoch_{epoch + 1}.pth"))

# Evaluate the final model on the test set
final_test_metrics = test_epoch(model, test_loader, criterion, device, args.num_epochs, run_path, experiment_name)
print("Final Test Metrics:", final_test_metrics)
