import torch
import tonic
from tonic import DiskCachedDataset
from torchvision import transforms
import torch.nn.functional as F
import os
import glob

class CustomEventDataset(torch.utils.data.Dataset):
    """
    A custom dataset for your own event camera recording.
    This version correctly finds all .aedat4 files in subdirectories.
    """
    def __init__(self, raw_data_path, transform=None):
        """
        Args:
            raw_data_path (str): The path to the directory containing class sub-folders of .aedat4 files.
            transform (callable, optional): A transform to be applied to a sample.
        """
        self.transform = transform
        self.classes = sorted([d.name for d in os.scandir(raw_data_path) if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = []
        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            class_dir = os.path.join(raw_data_path, class_name)
            for filepath in glob.glob(os.path.join(class_dir, '*.aedat4')):
                self.samples.append((filepath, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, target = self.samples[idx]
        events = tonic.io.read_aedat4(filepath)
        if self.transform:
            events = self.transform(events)
        return events, target

# --- Main execution block ---
if __name__ == '__main__':
    # --------------------------------------------------------------------------
    # TODO: USER CONFIGURATION REQUIRED
    # --------------------------------------------------------------------------
    # 1. Set the path to the folder containing your class sub-directories.
    #    e.g., "data/my_recordings/class_0/recording1.aedat4"
    my_data_path = "data/my_recordings"

    # 2. YOU MUST CHANGE THIS VALUE to match your camera's resolution.
    #    This is the most common source of errors.
    #    The format is (Width, Height, Polarity_Channels=2).
    sensor_size = (346, 260, 2)
    # -------------------------------------
    # clear-------------------------------------

    # 3. Define the pipeline for converting raw events to a tensor.
    step = 10 # Number of time steps
    size = 48 # Final spatial size (e.g., 48x48)

    event_to_frame_transform = transforms.Compose([
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=step),
    ])

    # 4. Create an instance of your raw dataset.
    print("Initializing custom raw dataset...")
    raw_dataset = CustomEventDataset(
        raw_data_path=my_data_path,
        transform=event_to_frame_transform
    )
    print(f"Found {len(raw_dataset)} samples in {len(raw_dataset.classes)} classes.")

    # 5. Define the final transforms to be applied after loading from cache.
    final_transforms = transforms.Compose([
        lambda x: torch.tensor(x, dtype=torch.float),
        lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
    ])

    # 6. Wrap your raw dataset with DiskCachedDataset for performance.
    cache_path = os.path.join(my_data_path, "custom_cache")
    cached_dataset = DiskCachedDataset(raw_dataset,
                                       cache_path=cache_path,
                                       transform=final_transforms)

    # 7. You can now wrap this with a PyTorch DataLoader.
    print("Creating DataLoader...")
    data_loader = torch.utils.data.DataLoader(cached_dataset,
                                              batch_size=4, # Small batch for testing
                                              shuffle=True)
    print("DataLoader created.")

    # 8. Get one batch of data to verify everything works.
    print("Attempting to get one batch of data... (This may take a moment on the first run as it processes and caches the data)")
    first_batch, first_labels = next(iter(data_loader))
    print(f"\nSuccess!")
    print(f"Batch shape: {first_batch.shape}")
    print(f"Labels in batch: {first_labels}")
    print(f"This confirms the entire data pipeline is working for your custom data.") 