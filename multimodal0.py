'''
import mne
import torch
from mne.io import read_raw_snirf
from mne_nirs.channels import get_long_channels
from scipy import interpolate
import numpy as np
import os
import joblib  # For loading your model


# Constants and Configuration 
data_dir = '/home/jobbe/Desktop/Thesis_Mind-fMRI/2021-fNIRS-Analysis-Methods-Passive-Auditory/2021-fNIRS-Analysis-Methods-Passive-Auditory'
subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10', 'sub-11', 'sub-12', 'sub-13', 'sub-14', 'sub-15', 'sub-16', 'sub-17']  # Include all subjects
task_name = 'AudioSpeechNoise'


# Function for padding (same as before)
def pad_to_length(x, length=1000):
    # ... (same as before) ...
'''
import mne
import torch
import torch.nn as nn
from mne.io import read_raw_snirf
from mne_nirs.channels import get_long_channels
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import time



# Constants and Configuration
data_dir = '/home/jobbe/Desktop/Thesis_Mind-fMRI/2021-fNIRS-Analysis-Methods-Passive-Auditory/2021-fNIRS-Analysis-Methods-Passive-Auditory'
subjects = ['sub-01']  # Start with one subject for testing
task_name = 'AudioSpeechNoise'

# Load data from all subjects  

# Existing loop where the error occurred
for subject in subjects:
    # Process each subject
    pass  # Replace with actual processing code
# Load data from all subjects
all_subject_data = []
all_subject_labels = []


# Custom Dataset class
class FNIRSDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def pad_to_length(x, length=1000):  # Default length is 1000
    assert x.ndim == 2
    assert x.shape[-1] <= length
    if x.shape[-1] == length:
        return x
    return np.pad(x, ((0, 0), (0, length - x.shape[1])), mode='constant')

def augmentation(data, aug_times=2, interpolation_ratio=0.5):
    num_to_generate = int((aug_times - 1) * len(data))
    if num_to_generate == 0:
        return data
    pairs_idx = np.random.choice(len(data), size=(num_to_generate, 2), replace=True)
    data_aug = []
    for i in pairs_idx:
        z = interpolate_voxels(data[i[0]], data[i[1]], interpolation_ratio)
        data_aug.append(np.expand_dims(z, axis=0))
    data_aug = np.concatenate(data_aug, axis=0)
    return np.concatenate([data, data_aug], axis=0)

def interpolate_voxels(x, y, ratio=0.5):
    values = np.stack((x, y))
    points = (np.r_[0, 1], np.arange(len(x)))
    xi = np.c_[np.full((len(x)), ratio), np.arange(len(x)).reshape(-1, 1)]
    z = interpolate.interpn(points, values, xi)
    return z

def preprocess_subject(subject):
    """Preprocesses and augments fNIRS data for a single subject."""

    file_path = f'{data_dir}/{subject}/ses-01/nirs/{subject}_ses-01_task-{task_name}_nirs.snirf'
    raw_intensity = read_raw_snirf(file_path)

    # Preprocessing
    raw_intensity = get_long_channels(raw_intensity)
    raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)

    # Create a new Annotations object
    annotations = mne.Annotations(onset=raw_intensity.annotations.onset,
                                 duration=raw_intensity.annotations.duration,
                                 description=raw_intensity.annotations.description)

    # Set the annotations to the raw_od object
    raw_od.set_annotations(annotations)

    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od)

    # Filtering
    raw_haemo = raw_haemo.filter(l_freq=0.01, h_freq=0.2)

    # Epoch Extraction 
    events, event_id = mne.events_from_annotations(raw_haemo)
    epochs = mne.Epochs(raw_haemo, events, event_id=event_id, 
                        tmin=-5, tmax=20, baseline=(None, 0), preload=True)

    # Data Augmentation
    data = epochs.get_data()

    # Pad data to desired length
    data = np.array([pad_to_length(epoch) for epoch in data])
    
    augmented_data = augmentation(data, aug_times=2)  # Use your augmentation function
    
    # Convert to PyTorch tensors
    data_tensor = torch.tensor(augmented_data)
    labels_tensor = torch.tensor(np.repeat(epochs.events[:, -1], 2))

    return data_tensor, labels_tensor

# Preprocess all subjects
for subject in subjects:
    data, labels = preprocess_subject(subject)
    all_subject_data.append(data)
    all_subject_labels.append(labels)


# Combine data from all subjects and create dataloader
data = torch.cat(all_subject_data, axis=0)
labels = torch.cat(all_subject_labels, axis=0)
print(f"Data type before model: {data.dtype}")
print(f"Data shape before model: {data.shape}")


# Split data into training, validation, and test sets
train_val_data, test_data, train_val_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
train_data, val_data, train_labels, val_labels = train_test_split(train_val_data, train_val_labels, test_size=0.2, random_state=42)
print(f"Train data shape: {train_data.shape}")
print(f"Validation data shape: {val_data.shape}")
print(f"Test data shape: {test_data.shape}")

# Create DataLoader for training and validation sets
train_dataset = FNIRSDataset(train_data, train_labels)
val_dataset = FNIRSDataset(val_data, val_labels)
test_dataset = FNIRSDataset(test_data, test_labels)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)  # For testing phase

# Save the data and labels if needed
torch.save(train_data, 'train_data.pt')
torch.save(train_labels, 'train_labels.pt')
torch.save(val_data, 'val_data.pt')
torch.save(val_labels, 'val_labels.pt')
torch.save(test_data, 'test_data.pt')
torch.save(test_labels, 'test_labels.pt')

# ... (Rest of the code for model training, evaluation, and visualization) ...

# ... The rest of the code (model definition, training loop, visualization) remains the same ...
'''


def preprocess_subject(subject):
    """Preprocesses fNIRS data for a single subject, matching the training data format."""

    file_path = f'{data_dir}/{subject}/ses-01/nirs/{subject}_ses-01_task-{task_name}_nirs.snirf'
    raw_intensity = read_raw_snirf(file_path)

    # Preprocessing (same as before)
    raw_intensity = get_long_channels(raw_intensity)
    raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)

    # Create a new Annotations object
    annotations = mne.Annotations(onset=raw_intensity.annotations.onset,
                                 duration=raw_intensity.annotations.duration,
                                 description=raw_intensity.annotations.description)

    # Set the annotations to the raw_od object
    raw_od.set_annotations(annotations)

    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od)
    raw_haemo = raw_haemo.filter(l_freq=0.01, h_freq=0.2)

    # Epoch Extraction 
    events, _ = mne.events_from_annotations(raw_haemo)
    event_dict = {'Speech': 1, 'Noise': 2}  # Adjust if needed based on your data
    epochs = mne.Epochs(raw_haemo, events, event_id=event_dict, 
                        tmin=-5, tmax=20, baseline=(None, 0), preload=True)

    # Convert to PyTorch tensors and pad to match training data
    data_tensor = torch.tensor(np.array([pad_to_length(epoch) for epoch in epochs.get_data()]))  
    labels_tensor = torch.tensor(epochs.events[:, -1])
    return data_tensor, labels_tensor

# Load your trained model
model_path = '/home/jobbe/transfer_learning/jina-ai/autoencoder_model_2_1_3epoch.pth'  # Replace with the actual path
model = joblib.load(model_path)

# Preprocess all subjects
all_data = []
all_labels = []
for subject in subjects:
    data, labels = preprocess_subject(subject)
    all_data.append(data)
    all_labels.append(labels)

all_data_tensor = torch.cat(all_data)
all_labels_tensor = torch.cat(all_labels)

# GPU Acceleration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
all_data_tensor = all_data_tensor.to(device)
all_labels_tensor = all_labels_tensor.to(device)

# Use your trained model for inference (prediction)
with torch.no_grad():
    predictions = model(all_data_tensor)

# Analyze your predictions
# calculate accuracy, visualize results
def calculate_accuracy(predictions, labels):
    # Your accuracy calculation logic here
    pass

accuracy = calculate_accuracy(predictions, all_labels_tensor)
print(f"Accuracy: {accuracy}")
# ... (rest of your analysis) ...

# Save predictions if needed
torch.save(predictions, 'predictions.pt')

# ... (rest of your code) ...
# Example of how to use the predictions

'''