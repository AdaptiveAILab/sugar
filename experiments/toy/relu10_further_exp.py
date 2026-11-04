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
from rai_init import apply_rai

# Make sure output directories exist before anything writes to them
for _d in ("outputs", "models_sweep", "results"):
    os.makedirs(_d, exist_ok=True)

# Logging is configured further down, once the per-config vars (fgi, target_func,
# widths_tag) are known, so each config writes to its own uniquely named log file
# instead of colliding on a shared second-resolution timestamp under parallelism.
tm = time.strftime("%Y%m%d_%H-%M-%S")

# Device config
device = torch.device("cpu")  # "'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Global constants and variables
input_dim = 1
output_dim = 1
DEFAULT_LR = 5e-3
num_epochs = 250
n_runs = 1000  # Number of independent runs

# Number of hidden (ReLU-activated) layers. The network then has NUM_HIDDEN + 1
# linear blocks in total (the final block has no activation).
NUM_HIDDEN = 9

# Default values for input and output
fgi = "Custom"  # Our B-SiLU formulation
analyze_freq = 1
fw_func = "NONE"
layer_func = nn.ReLU()
layer_type = nn.ReLU
target_func = "absfunc1"

# Arguments being parsed to the script
parser = argparse.ArgumentParser(description="ReLU10 FGI Experiment")

parser.add_argument("--fgi", type=str, default=fgi, help="Set the FGI type (possible: 'NONE', 'LeakyReLU', 'ELU', 'CeLU', 'GeLU', 'SeLU', 'Mish', 'Custom')")
parser.add_argument("--analyze_freq", type=int, default=analyze_freq, help="Set the analysis frequency (default: 1)")
parser.add_argument("--hidden_dim", type=int, default=2, help="Uniform hidden width, used only when --widths is not given (default: 2)")
parser.add_argument("--widths", type=str, default=None,
                    help=f"Comma-separated per-layer hidden widths ({NUM_HIDDEN} values), "
                         f"e.g. '4,4,2,2,2,2,2,2,2'. Overrides --hidden_dim. "
                         f"Lets you change a single layer's width in isolation.")
parser.add_argument("--num_epochs", type=int, default=num_epochs, help="Set the number of training epochs")
parser.add_argument("--batch_size", type=int, default=64, help="Set the batchsize")
parser.add_argument("--n_runs", type=int, default=n_runs, help="Number of independent runs (default: 1000)")
parser.add_argument("--optimize", type=str, default="adam", help="which optimizer (default: adam)")
parser.add_argument("--lr", type=float, default=DEFAULT_LR)
parser.add_argument("--no_lr_scheduling", action="store_true", help="lr scheduler turned off if this is given")
parser.add_argument("--momentum", type=float, default=0.0)
# The "NONE" argument refers to the usage of the standard ReLU function. Its name stems from the usage of the fgi helper function.
parser.add_argument("--fw_func", type=str, default=fw_func, help="Set the FW function (possible: 'NONE', 'LeakyReLU', 'ELU', 'CeLU', 'GeLU', 'SeLU', 'Mish')")
parser.add_argument("--target_func", type=str, default=target_func, help="Set the target function(absfunc1, sinefunc2, sinefunc3, absfunc4)")
parser.add_argument("--seed", type=int, default=42,
                    help="Base seed. Data/init/shuffle seeds derive from this but "
                         "NOT from --fgi, so fgi=NONE and fgi=Custom are paired "
                         "(same data, same init, same batch order) per capacity/run.")
parser.add_argument("--rai", action="store_true", help="Initializes all intermediate linear weights of the network to RAI, the first one is he-initialized")

args = parser.parse_args()

# Update variables
fgi = args.fgi
analyze_freq = args.analyze_freq
num_epochs = args.num_epochs
n_runs = args.n_runs
fw_func = args.fw_func
target_func = args.target_func
base_seed = args.seed

# Stable per-target offsets (do NOT use hash(), which is salted per process).
# Keeps the dataset identical across fgi types and capacities for a given target,
# while different targets get different data.
TARGET_SEED_OFFSET = {"absfunc1": 0, "sinefunc2": 1, "sinefunc3": 2, "absfunc4": 3}


def make_run_seed(run):
    """Per-run seed for weight init and batch order.

    Depends on base_seed and the run index only -- deliberately NOT on fgi -- so
    run r of fgi=NONE and run r of fgi=Custom (at the same capacity) start from
    identical weights and see the same shuffle order. Capacity is not folded in
    either, so a given run index also starts from the same init across capacities.
    """
    return base_seed + 1 + run

# Resolve the per-layer hidden widths.
# If --widths is given it wins; otherwise fall back to a uniform --hidden_dim.
if args.widths is not None:
    hidden_widths = [int(w) for w in args.widths.split(",") if w.strip() != ""]
else:
    hidden_widths = [args.hidden_dim] * NUM_HIDDEN
    
num_layers = len(hidden_widths) + 1
# Compact tag used in output filenames so different width configs never
# overwrite each other's results.
widths_tag = "-".join(str(w) for w in hidden_widths)

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

if args.rai:
    rai_tag = "_rai_init"
else:
    rai_tag = ""

if args.optimize == "sgd":
    opt_tag = f"_opt-{args.optimize}_lr{args.lr:g}_mom{args.momentum:g}"
else:
    if args.lr != DEFAULT_LR:
        opt_tag = f"_opt-{args.optimize}_lr{args.lr:g}"
    else:
        opt_tag = ""


model_file = f"models_sweep/relu{num_layers}_{num_epochs}_epochs_fw_func_{fw_func}_fgi_{fgi}_target_func_{target_func}{rai_tag}_widths_{widths_tag}{opt_tag}.pth"
results_file = f"results/relu{num_layers}_{num_epochs}_epochs_fw_func_{fw_func}_fgi_{fgi}_target_func_{target_func}{rai_tag}_widths_{widths_tag}{opt_tag}.npy"
log_file = f"outputs/{tm}_fgi_{fgi}_target_{target_func}{rai_tag}_widths_{widths_tag}{opt_tag}_pid{os.getpid()}.log"
# Configure logging now that the config is known. Naming the log file by config
# (plus PID as a final tie-breaker) guarantees each parallel process writes to a
# distinct, mappable file instead of sharing one second-resolution timestamp.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
)

print(f"Using params: {results_file}")
print(f"Hidden widths: {hidden_widths}")
logging.info(f"Base seed: {base_seed} (fgi={fgi}, target={target_func}, widths={widths_tag})")


class SimpleReLUNetwork(nn.Module):
    """MLP built from an explicit list of hidden-layer widths.

    `hidden_widths` gives the output width of each hidden block (layer1..layerN,
    each Linear + ReLU). A final linear block (no activation) maps the last
    hidden width to `output_dim`. The layer1..layer{N+1} attribute names are kept
    so perform_relu_fgi locates the same modules it did in the original script.
    """
    def __init__(self, input_dim, hidden_widths, output_dim):
        super(SimpleReLUNetwork, self).__init__()
        dims = [input_dim] + list(hidden_widths)
        self.num_hidden = len(hidden_widths)
        for i in range(1, self.num_hidden + 1):
            setattr(self, f"layer{i}",
                    nn.Sequential(nn.Linear(dims[i - 1], dims[i]), nn.ReLU()))
        # Final linear block, no ReLU.
        self.final_idx = self.num_hidden + 1
        setattr(self, f"layer{self.final_idx}",
                nn.Sequential(nn.Linear(dims[-1], output_dim)))

    def forward(self, x):
        out = x
        for i in range(1, self.final_idx + 1):
            out = getattr(self, f"layer{i}")(out)
        return out


# Data sampling
train_size = 3000
batch_size = args.batch_size

# Seed data generation by (base_seed, target) only, so the dataset is identical
# across fgi types and capacities for a given target function.
data_seed = base_seed * 100 + TARGET_SEED_OFFSET.get(target_func, 0)
torch.manual_seed(data_seed)

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

    x = torch.cat((x, x_2), dim=1)  # Size([3000, 2])
    y = torch.cat((y_1, y_2), dim=1)  # Size([3000, 2])

    # In order for the network to be able to process the new 2D-input, the dims need to be changed
    input_dim, output_dim = 2, 2

# Create DataLoader with a dedicated generator. Reseeding this generator per run
# (see train_and_evaluate) makes the batch order reproducible and, crucially,
# independent of how much RNG model construction / perform_relu_fgi consumed --
# so fgi=NONE and fgi=Custom get the same shuffle even if Custom adds parameters.
x, y = x.to(device), y.to(device)
dataset = TensorDataset(x, y)
loader_generator = torch.Generator()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=loader_generator)

criterion = nn.MSELoss(reduction='sum').to(device)


def count_parameters(model):
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# Log the parameter count only once per invocation (it's identical across runs
# for a given width config), rather than spamming it on every run.
_params_logged = False


# Train, test, tracking etc.
def train_and_evaluate(input_dim, hidden_widths, output_dim, learning_rate, epochs, dataloader, run=0, optimize: str="adam", momentum=0.0, rai=False, no_lr_scheduling=False) -> tuple[torch.nn.Module, float, dict]:
    global _params_logged

    # Seed for this run. Same seed for fgi=NONE and fgi=Custom (seed ignores fgi),
    # so the Linear weights below are initialized identically for both. The loader
    # generator gets its own seed so the batch order matches too, regardless of any
    # extra RNG that perform_relu_fgi draws for Custom.
    run_seed = make_run_seed(run)
    torch.manual_seed(run_seed)
    loader_generator.manual_seed(run_seed + 100000)

    model = SimpleReLUNetwork(input_dim, hidden_widths, output_dim)  # Linear init drawn here
    model, layer_type = perform_relu_fgi(model, fgi)
    if rai:
        rai_gen = torch.Generator()                     # CPU generator (model is on CPU)
        rai_gen.manual_seed(run_seed + 200000)          # distinct offset from the shuffle
        apply_rai(model, generator=rai_gen)   
    model.to(device)
    # print(model)
    # Counted after perform_relu_fgi so any params the FGI adds are included.
    if not _params_logged:
        total, trainable = count_parameters(model)
        logging.info(f"Num params: {total} total, {trainable} trainable "
                     f"(widths={hidden_widths}, fgi={fgi})")
        _params_logged = True
        
    if optimize == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        
    if not no_lr_scheduling:
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
            # Move tensors to the configured device
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()  # Reset gradients

            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            epoch_loss += loss.item()
            num_train_batches += 1

            # Backward pass
            loss.backward()
            optimizer.step()

        if not no_lr_scheduling:
            scheduler.step()

        loss_dict[epoch] = epoch_loss / num_train_batches

        run_dict = OrderedDict()
        run_dict["loss"] = loss_dict

    return model, loss.item(), run_dict


def meta_train(n_runs, input_dim, hidden_widths, output_dim, learning_rate, epochs, dataloader):
    losses = []
    min_loss = 1e6
    analysis_dict = OrderedDict()
    model = None
    for run in range(n_runs):
        print(f"Starting Run {run}")

        model, final_loss, run_dict = train_and_evaluate(input_dim, 
                                                         hidden_widths, 
                                                         output_dim, learning_rate, 
                                                         epochs, dataloader, run=run, 
                                                         optimize=args.optimize, momentum=args.momentum, 
                                                         rai=args.rai, no_lr_scheduling=args.no_lr_scheduling)
        analysis_dict[run] = run_dict

        if run % 250 == 0:
            timestamp = datetime.now().strftime("%m-%d %H:%M:%S")
            log_message = f"[{timestamp}] Run {run}, Final Loss: {final_loss:.4f}"
            logging.info(log_message)

        print(f"Run {run + 1}/{n_runs}, Final Loss: {final_loss:.4f}", flush=True)
    np.save(results_file, analysis_dict)
    torch.save(model.state_dict(), model_file)  

    return losses


def main():
    meta_train(n_runs, input_dim, hidden_widths, output_dim, args.lr, num_epochs, dataloader)


if __name__ == '__main__':
    main()