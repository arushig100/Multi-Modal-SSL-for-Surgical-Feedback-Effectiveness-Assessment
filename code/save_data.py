import torch
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import VideoMAEModel, VideoMAEImageProcessor
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import UniformTemporalSubsample
from torch.utils.data import Dataset
import os
import gc

# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set a seed for reproducibility
seed = 42
torch.manual_seed(seed)

# Load dataset
def load_data(data_path, file_name):
    """ Load dataset from a CSV file. """
    try:
        annot_df = pd.read_csv(os.path.join(data_path, file_name))
        # Example: Clean the data if necessary
        annot_df = annot_df.dropna()  # Example: Drop missing data rows
        return annot_df
    except FileNotFoundError:
        print("Data file not found. Please provide a valid path.")
        return pd.DataFrame()

# Text cleaning function (to be altered/applied depending on specific dataset used)
def clean_text(text):
    """ Clean transcribed text. """
    if not isinstance(text, str):
        return ""
    
    replacements = {
        "CJ": "", "Cj": "", 'chief': '', "Dr. Aron": "he", "Kian": "", "Deborah": "her", "Luis": "", 
        "jeffrey": "", "keith": "", "randall": "", "jian": "", "!": "stop", "k": "ok", "K": "ok"
    }
    
    for key, value in replacements.items():
        text = text.replace(key, value)
    
    text = re.sub(r'^[k]( |A-z)+', 'ok ', text)
    text = re.sub(r'[ ][k]$', ' ok', text)
    
    return text.strip()

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

# Save dataset in parts for larger datasets
def save_dataset_in_parts(data, video_dir, processor, sbert_model, num_frames, features_to_save, 
                          text_col="Dialogue", video_col="cvid", save_path_prefix="dataset", num_parts=1):
    """ Save dataset in parts for large datasets. """
    total_size = len(data)
    part_size = total_size // num_parts
    for part in range(num_parts):
        start_idx = part * part_size
        end_idx = (part + 1) * part_size if part != num_parts - 1 else total_size
        dataset = VideoTextDataset(data[start_idx:end_idx], video_dir, processor, sbert_model, num_frames, 
                                   features_to_save, text_col=text_col, video_col=video_col)
        torch.save(dataset, f"{save_path_prefix}_part{part + 1}.pth")
        print(f"Saved part {part + 1}")
        del dataset
        gc.collect()

# Example usage for saving dataset
if __name__ == "__main__":
    data_path = "your_data_path"
    annot_df = load_data(data_path, "your_data_file.csv")
    annot_df['cleaned_text'] = annot_df['auto_transcription'].apply(clean_text)  # Generalized auto transcript column

    # Load models
    processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    video_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics", output_hidden_states=True)
    video_model.to(device)
    sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    sbert_model.to(device)

    # Example of saving the dataset
    save_dataset_in_parts(
        annot_df, "your_video_dir", processor, sbert_model, num_frames=16,
        features_to_save=['trainee_behavior_change'], text_col="cleaned_text", video_col="cvid", save_path_prefix="dataset"
    )
    print("Dataset saved successfully.")
