import torch
import tonic
import numpy as np
import argparse
import os
from torchvision import transforms
import torch.nn.functional as F

import braincog
from braincog.model_zoo.vgg_snn import VGG_SNN
from braincog.base.node import *

# --- Main Inference Logic ---

def run_inference(model_path, input_file, sensor_size_str, device_str, step, node_type_str):
    """
    Loads a trained model, processes a single event file, and prints the predicted class.
    """
    # --- 1. Setup Environment ---
    device = torch.device(device_str)
    
    # The class names must match the folders created by the organization script.
    # The order is determined alphabetically, which is how the DataLoader reads them.
    idx_to_class = sorted(['Fire_Safety', 'First_Aid', 'Mandatory', 'Prohibition', 'Safe_Condition', 'Warning'])
    num_classes = len(idx_to_class)

    try:
        sensor_w, sensor_h, _ = map(int, sensor_size_str.strip('[]').split(','))
        sensor_size = (sensor_w, sensor_h, 2)
    except Exception:
        raise ValueError("Invalid sensor_size format. Expected '[width,height,2]', e.g., '[346,260,2]'")

    # --- 2. Load the Model ---
    print(f"Loading model from {model_path}...")
    node_type = getattr(braincog.base.node, node_type_str)

    # Re-create the model with the same architecture as during training
    model = VGG_SNN(
        num_classes=num_classes,
        step=step,
        node_type=node_type,
        dataset='my_custom_data'  # This is needed to pass the internal check
    )

    # Load the saved weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    print("Model loaded successfully.")

    # --- 3. Load and Preprocess the Input File ---
    print(f"Loading and processing input file: {input_file}")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"The specified input file does not exist: {input_file}")

    # The preprocessing pipeline must be IDENTICAL to the one used for the test set during training.
    transform = transforms.Compose([
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=step),
        lambda x: torch.tensor(x, dtype=torch.float),
        lambda x: F.interpolate(x, size=[48, 48], mode='bilinear', align_corners=True),
    ])

    # Load the event data based on file extension
    if input_file.endswith('.aedat4'):
        events = tonic.io.read_aedat4(input_file)
    elif input_file.endswith('.csv'):
        events_np = np.loadtxt(input_file, delimiter=',', skiprows=1, dtype=np.int64)
        events = np.core.records.fromarrays(events_np.T, names='t,x,y,p', formats='i8,i2,i2,i2')
    else:
        raise ValueError("Unsupported file type. Please provide a .aedat4 or .csv file.")

    # Apply transformations and add a batch dimension (B, T, C, H, W)
    processed_data = transform(events).unsqueeze(0).to(device)

    # --- 4. Perform Inference ---
    print("Running inference...")
    with torch.no_grad():  # Disable gradient calculation for efficiency
        output = model(processed_data)
        
    # Get the index of the class with the highest score
    pred_idx = output.argmax(dim=1).item()
    predicted_class = idx_to_class[pred_idx]

    # --- 5. Display Result ---
    print("\n--- Inference Result ---")
    print(f"The model predicts that the sign is: '{predicted_class}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SNN Inference Script')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint (.pth file).')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input event file (.aedat4 or .csv).')
    parser.add_argument('--sensor_size', type=str, default='[346,260,2]', help="Camera resolution as a string, e.g., '[346,260,2]'")
    parser.add_argument('--device', type=str, default='cpu', help="Device to use ('cpu' or 'cuda:0').")
    parser.add_argument('--step', type=int, default=8, help='SNN simulation time steps (must match training).')
    parser.add_argument('--node_type', type=str, default='LIFNode', help='Neuron type used during training.')
    
    args = parser.parse_args()
    
    run_inference(args.model_path, args.input_file, args.sensor_size, args.device, args.step, args.node_type) 