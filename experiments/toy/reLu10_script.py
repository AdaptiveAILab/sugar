# Imports
import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import OrderedDict
import argparse
import logging
import sys
import time
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from relu_fgi import perform_relu_fgi


# Output folders (models_retrained/ keeps the provided paper networks in models/ untouched)
MODEL_DIR, RESULTS_DIR = "models_retrained", "results_retrained"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Logging config
tm = time.strftime("%Y%m%d_%H-%M-%S")
log_file = f"{RESULTS_DIR}/{tm}_ReLUexperiment.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
)

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Global constants and variables
input_dim = 1
# hidden_dim = 2
output_dim = 1
learning_rate = 5e-3
num_epochs = 250
n_runs = 1 # Number of independent runs

# Default values for input and output
fgi = "Custom" # Our B-SiLU formulation
analyze_freq = 1
fw_func = "NONE"
layer_func = nn.ReLU()
layer_type = nn.ReLU
target_func = "absfunc1"

# Arguments being parsed to the script
parser = argparse.ArgumentParser(description="ReLU10 FGI Experiment")

parser.add_argument("--fgi", type=str, default=fgi, help="Set the FGI type (possible: 'NONE', 'LeakyReLU', 'ELU', 'CeLU', 'GeLU', 'SeLU', 'Mish', 'Custom')")
parser.add_argument("--analyze_freq", type=int, default=analyze_freq, help="Set the analysis frequency (default: 1)")
parser.add_argument("--hidden_dim", type=int, default=2, help="Set the hidden dim (default: 2)")
parser.add_argument("--num_epochs", type=int, default=num_epochs, help="Set the number of training epochs")
# The "NONE" argument refers to the usage of the standard ReLU function. It's name stems from the usage of the fgi helper function.
parser.add_argument("--fw_func", type=str, default=fw_func, help="Set the FW function (possible: 'NONE', 'LeakyReLU', 'ELU', 'CeLU', 'GeLU', 'SeLU', 'Mish')")
parser.add_argument("--seed", type=int, default=42, help="Seed for data, initialisation and batch order")
parser.add_argument("--target_func", type=str, default=target_func, help="Set the target function(absfunc1, sinefunc2, sinefunc3, absfunc4)")

args = parser.parse_args()

# Update variables
fgi = args.fgi
analyze_freq = args.analyze_freq
num_epochs = args.num_epochs
fw_func = args.fw_func
target_func = args.target_func
hidden_dim = args.hidden_dim
torch.manual_seed(args.seed)

if fw_func == "NONE":
    layer_func = nn.ReLU()
    layer_type = nn.ReLU
elif fw_func == "LeakyReLU":
    layer_func = nn.LeakyReLU()
    layer_type = nn.LeakyReLU
elif fw_func == "ELU":
    layer_func = nn.ELU()
    layer_type = nn.ELU
elif fw_func == "CeLU":
    layer_func = nn.CELU()
    layer_type = nn.CELU
elif fw_func == "GeLU":
    layer_func = nn.GELU()
    layer_type = nn.GELU
elif fw_func == "SELU":
    layer_func = nn.SELU()
    layer_type = nn.SELU
elif fw_func == "Mish":
    layer_func = nn.Mish()
    layer_type = nn.Mish

# File names as loaded by plot_collapse_split.py
model_file = f"{MODEL_DIR}/relu10_{num_epochs}_epochs_fw_func_{fw_func}_fgi_{fgi}_target_func_{target_func}.pth"
results_file = f"{RESULTS_DIR}/relu10_{num_epochs}_epochs_fw_func_{fw_func}_fgi_{fgi}_target_func_{target_func}.npy"

print(f"Using params: {results_file}")

class SimpleReLUNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleReLUNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.layer7 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.layer8 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.layer9 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.layer10 = nn.Sequential(
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        return out

# Data sampling
train_size = 3000
batch_size = 64
x = torch.empty(train_size, 1).uniform_(-np.sqrt(3), np.sqrt(3))

# For the 3D-task absfunc4, a second x input is needed
x_2 = torch.empty(train_size, 1).uniform_(-np.sqrt(3), np.sqrt(3))

if target_func == 'absfunc1':
    y = torch.abs(x)
elif target_func == 'sinefunc2':
    y = x * torch.sin(5 * x)
elif target_func == 'sinefunc3':
    t = torch.where(x > 0, 1.0, 0.0)
    y = t + 0.2 * torch.sin(5 * x)
elif target_func == 'absfunc4':
    y_1 = torch.abs(x + x_2)
    y_2 = torch.abs(x - x_2)

    x = torch.cat((x, x_2), dim=1) # Size([3000, 2])
    y = torch.cat((y_1, y_2), dim=1) # Size([3000, 2])

    # In order for the network to be able to process the new 2D-input, the dims need to be changed
    input_dim, output_dim = 2, 2

# Create DataLoader
x, y = x.to(device), y.to(device)
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

criterion = nn.MSELoss(reduction='sum').to(device)

# Train, test, tracking etc.
def train_and_evaluate(input_dim, hidden_dim, output_dim, learning_rate, epochs, dataloader) -> tuple[torch.nn.Module, float, dict]:
    model = SimpleReLUNetwork(input_dim, hidden_dim, output_dim)
    model, layer_type = perform_relu_fgi(model, fgi)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150, 200, 225])

    loss_dict = OrderedDict()

    for epoch in range(epochs):
        if epoch % 50 == 0:
            print(f"Epoch {epoch}/{num_epochs}")

        epoch_loss = 0
        num_train_batches = 0
        # Training loop
        model.train()

        for batch_x, batch_y in dataloader:
            optimizer.zero_grad() # Reset gradients
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            epoch_loss += loss.item()
            num_train_batches += 1

            # Backward pass
            loss.backward()
            optimizer.step()

        scheduler.step()

        loss_dict[epoch] = epoch_loss / num_train_batches


        run_dict = OrderedDict()
        run_dict["loss"] = loss_dict

    return model, loss.item(), run_dict

def meta_train(n_runs, input_dim, hidden_dim, output_dim, learning_rate, epochs, dataloader):
    losses = []
    min_loss = 1e6
    analysis_dict = OrderedDict()
    for run in range(n_runs):
        print(f"Starting Run {run}")

        model, final_loss, run_dict = train_and_evaluate(input_dim, hidden_dim, output_dim, learning_rate, epochs, dataloader)
        analysis_dict[run] = run_dict

        if run % 250 == 0:
            timestamp = datetime.now().strftime("%m-%d %H:%M:%S")
            log_message = f"[{timestamp}] Run {run}, Final Loss: {final_loss:.4f}"
            logging.info(log_message)

        print(f"Run {run + 1}/{n_runs}, Final Loss: {final_loss:.4f}")

    np.save(results_file, analysis_dict)
    # print number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, model_file)  # CPU tensors: loadable anywhere

    return losses

def main():
    meta_train(n_runs, input_dim, hidden_dim, output_dim, learning_rate, num_epochs, dataloader)

if __name__ == '__main__':
    main()
