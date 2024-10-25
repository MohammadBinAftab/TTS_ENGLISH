import os
import zipfile
import glob
import torch
import torchaudio
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, Trainer, TrainingArguments

# Paths to your zip files
txt_zip_path = 'path/to/txt_files.zip'
wav_zip_path = 'path/to/wav_files.zip'

# Extract txt files
with zipfile.ZipFile(txt_zip_path, 'r') as txt_zip:
    txt_zip.extractall('txt_files/')

# Extract wav files
with zipfile.ZipFile(wav_zip_path, 'r') as wav_zip:
    wav_zip.extractall('wav_files/')

# Load processor and model
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

# Load file paths
txt_files = sorted(glob.glob('txt_files/*.txt'))
wav_files = sorted(glob.glob('wav_files/*.wav'))

# Check if both lists are of equal length and correspond to each other
if len(txt_files) != len(wav_files):
    raise ValueError("Mismatch between the number of txt and wav files.")

# Function to load and process each text-audio pair
class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, txt_files, wav_files, processor):
        self.txt_files = txt_files
        self.wav_files = wav_files
        self.processor = processor

    def __len__(self):
        return len(self.txt_files)

    def __getitem__(self, idx):
        # Load transcription from the text file
        with open(self.txt_files[idx], 'r') as f:
            transcription = f.read().strip()

        # Load audio file
        waveform, sample_rate = torchaudio.load(self.wav_files[idx])

        # Process the text and audio
        inputs = self.processor(text=transcription, sampling_rate=sample_rate, audio=waveform, return_tensors="pt")
        return inputs

# Create dataset and dataloader
dataset = SpeechDataset(txt_files, wav_files, processor)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_speechT5",
    per_device_train_batch_size=1,  # Adjust batch size based on your GPU's capacity
    num_train_epochs=3,  # Set desired epochs
    logging_dir="./logs",
    save_steps=500,  # Save model every 500 steps
    save_total_limit=2,  # Keep only the last 2 checkpoints
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_speechT5")
processor.save_pretrained("./fine_tuned_speechT5")

print("Model fine-tuning complete and saved at './fine_tuned_speechT5'")
