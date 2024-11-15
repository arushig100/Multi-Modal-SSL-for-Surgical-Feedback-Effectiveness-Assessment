import torch
from transformers import VideoMAEForPreTraining, AutoImageProcessor, Trainer, TrainingArguments, TrainerCallback
from torch.utils.data import DataLoader, Dataset
import wandb
import os
from pytorchvideo.transforms import UniformTemporalSubsample
from pytorchvideo.data.encoded_video import EncodedVideo

# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Video Dataset class for VideoMAE pretraining
class VideoDataset(Dataset):
    def __init__(self, data, video_dir, processor, num_frames=16):
        """
        Args:
            data (pd.DataFrame): DataFrame containing video metadata.
            video_dir (str): Directory where the videos are stored.
            processor (AutoImageProcessor): VideoMAE image processor for preprocessing.
            num_frames (int): Number of frames to subsample from the video.
        """
        self.data = data
        self.video_dir = video_dir
        self.processor = processor
        self.num_frames = num_frames
        self.subsampler = UniformTemporalSubsample(self.num_frames)
        self.videos = []
        self.load_data()

    def load_data(self):
        """Loads and preprocesses video data."""
        for idx, row in self.data.iterrows():
            video_path = os.path.join(self.video_dir, row['cvid'])
            try:
                video = EncodedVideo.from_path(video_path)
                video_data = video.get_clip(start_sec=0, end_sec=10.0)["video"]
                subsampled_frames = self.subsampler(video_data)
                video_data_np = subsampled_frames.numpy().transpose(1, 0, 2, 3)
                video_batch = [video_data_np]
                inputs = self.processor(video_batch, return_tensors="pt")
                video_tensor = inputs["pixel_values"].squeeze()
                self.videos.append(video_tensor)
            except Exception as e:
                print(f"Error loading video {video_path}: {e}")
                continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"video": self.videos[idx]}

# Define the collate function
def collate_fn(examples, mask_ratio=0.85):
    """
    Collate function to prepare batches for VideoMAE pretraining.
    
    Args:
        examples (list): List of examples returned from the dataset.
        mask_ratio (float): The proportion of patches to be masked for pretraining.
    
    Returns:
        dict: Dictionary containing pixel values and masked positions.
    """
    pixel_values = torch.stack([example['video'] for example in examples])

    # Calculate sequence length and mask
    num_frames = pixel_values.size(1)
    model = VideoMAEForPreTraining.from_pretrained("MCG-NJU/videomae-base")
    num_patches_per_frame = (model.config.image_size // model.config.patch_size) ** 2
    seq_length = (num_frames // model.config.tubelet_size) * num_patches_per_frame
    num_ones = int(mask_ratio * seq_length)
    num_zeros = seq_length - num_ones
    mask = torch.cat([torch.ones(num_ones), torch.zeros(num_zeros)]).bool()

    # Repeat and shuffle mask for each sample in the batch
    bool_masked_pos = torch.stack([mask[torch.randperm(seq_length)] for _ in range(pixel_values.size(0))])

    return {"pixel_values": pixel_values, "bool_masked_pos": bool_masked_pos}

# WandbCallback for logging
class WandbCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Custom callback to log to Wandb during training."""
        if logs:
            wandb.log(logs)

# Function to initialize the VideoMAE model and processor
def initialize_model_and_processor(model_name="MCG-NJU/videomae-base"):
    """
    Initializes the VideoMAE model and processor.
    
    Args:
        model_name (str): Pretrained model name.
    
    Returns:
        tuple: VideoMAE model and processor.
    """
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = VideoMAEForPreTraining.from_pretrained(model_name)
    model.to(device)
    return model, processor

# Function to create a DataLoader
def create_dataloader(dataset, batch_size, collate_fn):
    """
    Creates a DataLoader from a dataset.
    
    Args:
        dataset (Dataset): Dataset object.
        batch_size (int): Batch size.
        collate_fn (function): Collate function for preprocessing.
    
    Returns:
        DataLoader: PyTorch DataLoader object.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Function to initialize the Trainer
def initialize_trainer(model, dataset, training_args, collate_fn, last_checkpoint=None):
    """
    Initializes the Hugging Face Trainer.
    
    Args:
        model (nn.Module): VideoMAE model.
        dataset (Dataset): Dataset for training.
        training_args (TrainingArguments): TrainingArguments object.
        collate_fn (function): Collate function for preprocessing.
        last_checkpoint (str): Checkpoint directory to resume from (optional).
    
    Returns:
        Trainer: Hugging Face Trainer object.
    """
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=collate_fn,
        callbacks=[WandbCallback()],
        resume_from_checkpoint=last_checkpoint
    )

# Function to define the training arguments
def define_training_args(output_dir, batch_size, num_epochs, steps_per_epoch, lr, weight_decay):
    """
    Defines training arguments for the Trainer.
    
    Args:
        output_dir (str): Output directory.
        batch_size (int): Batch size.
        num_epochs (int): Number of epochs.
        steps_per_epoch (int): Number of steps per epoch.
        lr (float): Learning rate.
        weight_decay (float): Weight decay.
    
    Returns:
        TrainingArguments: Hugging Face TrainingArguments object.
    """
    return TrainingArguments(
        output_dir=output_dir,
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        logging_dir='./logs',
        logging_strategy="steps",
        logging_steps=steps_per_epoch//2,
        report_to="wandb",
        save_total_limit=3
    )

# Main function to run pretraining
def run_pretraining(data, video_dir, output_dir, batch_size, num_epochs, lr, weight_decay, mask_ratio=0.85):
    """
    Main function to run VideoMAE pretraining.
    
    Args:
        data (pd.DataFrame): DataFrame with video metadata.
        video_dir (str): Directory where videos are stored.
        output_dir (str): Output directory for saving models.
        batch_size (int): Batch size.
        num_epochs (int): Number of epochs.
        lr (float): Learning rate.
        weight_decay (float): Weight decay.
        mask_ratio (float): Mask ratio for pretraining.
    """
    # Initialize the model and processor
    model, processor = initialize_model_and_processor()
    
    # Create the dataset and dataloader
    dataset = VideoDataset(data, video_dir, processor)
    dataloader = create_dataloader(dataset, batch_size, lambda x: collate_fn(x, mask_ratio))
    
    # Define training arguments
    num_samples = len(dataset)
    steps_per_epoch = num_samples // batch_size + (1 if num_samples % batch_size != 0 else 0)
    training_args = define_training_args(output_dir, batch_size, num_epochs, steps_per_epoch, lr, weight_decay)
    
    # Initialize trainer
    last_checkpoint = None
    if os.path.isdir(output_dir) and any(f.startswith("checkpoint") for f in os.listdir(output_dir)):
        last_checkpoint = os.path.join(output_dir, sorted(os.listdir(output_dir))[-2])
        print(f"Resuming training from checkpoint: {last_checkpoint}")
    
    trainer = initialize_trainer(model, dataset, training_args, collate_fn, last_checkpoint)
    
    # Start training
    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_state()

    # Save evaluation results
    eval_results = trainer.evaluate()
    trainer.log_metrics("train", eval_results)
    trainer.save_metrics("train", eval_results)
    trainer.save_state()
    
    print("Pretraining complete.")

# Example usage
if __name__ == "__main__":
    # Load your dataset (data should be a DataFrame with video metadata)
    data_path = "/path/to/data"
    video_dir = "/path/to/videos"
    output_dir = "/path/to/save/models"
    
    # Run the pretraining process
    run_pretraining(data_path, video_dir, output_dir, batch_size=24, num_epochs=500, lr=5e-7, weight_decay=0.05)

