import json
import os
import random
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
from tqdm import tqdm

import cnn.hooks as hooks
from cnn.hook_injector import HookInjector
from cnn.resnet18 import resnet18_cifar
from cnn.sugar import activation_function, apply_sugar
from cnn.vgg import VGG16

_AVAILABLE_MODELS = {
    "vgg16": lambda nc, **kw: VGG16(num_classes=nc, **kw),
    "resnet18": lambda nc, **kw: resnet18_cifar(num_classes=nc, **kw),
}
FGI_WITH_ALPHA = {"nelu"}


def replace_layers(module, old_layer_type, new_layer_fn):
    for name, child in module.named_children():
        if isinstance(child, old_layer_type):
            setattr(module, name, new_layer_fn())
        else:
            replace_layers(child, old_layer_type, new_layer_fn)


def _disable_inplace_relu(model: torch.nn.Module):
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False


def run_analysis_epoch(
    epoch: int,
    num_epochs: int,
    model: nn.Module,
    train_loader,
    criterion,
    injector,
    hooks,
    device,
    layer_type,
    analyze_freq: int,
    containers: Dict[str, Dict[int, Any]],
) -> None:
    if epoch % analyze_freq != 0:
        return
    if len(train_loader) == 0:
        raise ValueError("Cannot collect analysis statistics from an empty data loader")
    act_minmax, grad_minmax = ([], [])
    act_hist, grad_hist = ({}, {})
    nonzero_cnt, nonzero_neuron = ({}, {})
    try:
        injector.inject_forward(
            hooks.get_fwd_nonzero_count, hooks.is_layer_type(layer_type), nonzero_cnt
        )
        injector.inject_forward(
            hooks.get_fwd_acts_nonzero_act_cnt,
            hooks.is_layer_type(layer_type),
            nonzero_neuron,
        )
        injector.inject_forward(
            hooks.get_fwd_acts_minmax, hooks.is_layer_type(layer_type), act_minmax
        )
        handles = injector.inject_input_gradient(
            hooks.get_input_grads_minmax, hooks.is_layer_type(layer_type), grad_minmax
        )
        if not handles:
            raise RuntimeError("No matching activation layers found for analysis")
        total_step = len(train_loader)
        model.train(False)
        for step, (images, labels) in enumerate(train_loader, 1):
            images, labels = (images.to(device), labels.to(device))
            loss = criterion(model(images), labels)
            loss.backward()
            if step % 100 == 0:
                print(
                    f"Analyzing Pre Step Epoch[{epoch + 1}/{num_epochs}], Step [{step}/{total_step}]"
                )
        injector.remove_all_forward()
        injector.remove_all_backward()
        if len(act_minmax) != 2 or len(grad_minmax) != 2:
            raise RuntimeError(
                "Analysis collected no activation values or input gradients. "
                "Check that the selected layers are used and gradients are enabled."
            )
        injector.inject_forward(
            hooks.get_fwd_acts_hist,
            hooks.is_layer_type(layer_type),
            act_hist,
            act_minmax[0],
            act_minmax[1],
        )
        injector.inject_input_gradient(
            hooks.get_input_grads_hist,
            hooks.is_layer_type(layer_type),
            grad_hist,
            grad_minmax[0],
            grad_minmax[1],
        )
        model.zero_grad()
        for step, (images, labels) in enumerate(train_loader, 1):
            images, labels = (images.to(device), labels.to(device))
            loss = criterion(model(images), labels)
            loss.backward()
            if step % 100 == 0:
                print(
                    f"Analyzing Histogram Epoch[{epoch + 1}/{num_epochs}], Step [{step}/{total_step}]"
                )
    finally:
        injector.remove_all_forward()
        injector.remove_all_backward()
    nonzero_neuron_cpu = {
        lyr: tensor.data.cpu().numpy() for lyr, tensor in nonzero_neuron.items()
    }
    containers["acts_hists"][epoch] = act_hist
    containers["grads_hists"][epoch] = grad_hist
    containers["nonzero_counts"][epoch] = nonzero_cnt
    containers["acts_minmax"][epoch] = act_minmax
    containers["grads_minmax"][epoch] = grad_minmax
    containers["nonzero_neuron_acts"][epoch] = nonzero_neuron_cpu


class Experiment:
    def __init__(
        self,
        dataset="cifar10",
        model_name="vgg16",
        use_fgi=True,
        data_dir="../datasets",
        batch_size=128,
        valid_size=0.1,
        seed=42,
        num_workers=4,
        pin_memory=True,
        fgi_type="NONE",
        num_epochs=250,
        learning_rate=0.001,
        label_smoothing_train=0.0,
        label_smoothing_val=0.0,
        milestones=20,
        optimizer_name="SGD",
        alpha=0.1,
        results_dir="./vgg16_results",
        gpu_id=0,
        reg=False,
    ):
        self.base_results_dir = results_dir
        self.orig_results_dir = results_dir
        self.dataset = dataset.lower()
        self.data_dir = data_dir
        self.use_fgi = use_fgi
        self.model_name = model_name
        self.batch_size = batch_size
        self.valid_size = valid_size
        self.seed = seed
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.fgi_type = fgi_type
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.label_smoothing_train = label_smoothing_train
        self.label_smoothing_val = label_smoothing_val
        self.milestones = milestones
        self.optimizer_name = optimizer_name
        self.alpha = alpha
        os.makedirs(self.base_results_dir, exist_ok=True)
        self._set_seed(self.seed)
        self.gpu_id = gpu_id
        self.reg = reg
        self.device = torch.device(
            f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu"
        )

    def _set_seed(self, seed):
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    def _uses_alpha(self):
        return self.fgi_type in FGI_WITH_ALPHA

    def _build_model(self, num_classes: int):
        if self.model_name not in _AVAILABLE_MODELS:
            raise ValueError(f"Unknown model '{self.model_name}'")
        return _AVAILABLE_MODELS[self.model_name](num_classes)

    def _make_results_path(self) -> str:
        tag = "fgi" if self.use_fgi else "non_fgi"
        parts = [
            self.base_results_dir,
            self.dataset,
            self.model_name,
            tag,
            self.fgi_type,
        ]
        if self._uses_alpha():
            parts.append(f"alpha_{self.alpha}")
        parts.append(f"seed_{self.seed}")
        path = os.path.join(*parts)
        os.makedirs(path, exist_ok=True)
        return path

    def _prepare_model(self, num_classes: int):
        model = self._build_model(num_classes).to(self.device)
        if self.use_fgi:
            model, layer_type = apply_sugar(model, self.fgi_type, alpha=self.alpha)
            return (model, layer_type)
        layer_type = nn.ReLU
        function = activation_function(self.fgi_type, self.alpha)

        class Surrogate(nn.Module):
            def forward(self, x):
                return function(x)

        layer_type = Surrogate
        replace_layers(model, nn.ReLU, lambda: Surrogate())
        return (model, layer_type)

    def get_data_loaders(self):
        if self.dataset == "cifar10":
            mean, std = ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.201])
            DatasetClass = datasets.CIFAR10
            resize_op = transforms.Lambda(lambda x: x)
        elif self.dataset == "cifar100":
            mean, std = ([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])
            DatasetClass = datasets.CIFAR100
            resize_op = transforms.Lambda(lambda x: x)
        elif self.dataset in ("tinyimg", "tiny-imagenet", "tinyimagenet"):
            mean, std = ([0.4802, 0.4481, 0.3975], [0.277, 0.2691, 0.2821])
            DatasetClass = datasets.ImageFolder
            resize_op = transforms.Resize(64)
        else:
            raise ValueError(f"Unknown dataset '{self.dataset}'")
        tf_train = transforms.Compose(
            [
                resize_op,
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        tf_test = transforms.Compose(
            [resize_op, transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
        if "tiny" in self.dataset:
            root = Path(self.data_dir) / "tiny-imagenet-200"
            train_ds = DatasetClass(root / "train", transform=tf_train)
            val_root = root / "val"
            valid_ds = DatasetClass(val_root, transform=tf_test)
            test_ds = valid_ds
            train_sampler = valid_sampler = None
        else:
            full_train = DatasetClass(
                self.data_dir, train=True, download=True, transform=tf_train
            )
            num_train = len(full_train)
            split = int(self.valid_size * num_train)
            indices = list(range(num_train))
            rng = np.random.RandomState(self.seed)
            rng.shuffle(indices)
            train_idx, valid_idx = (indices[split:], indices[:split])
            train_ds = full_train
            valid_ds = DatasetClass(
                self.data_dir, train=True, download=False, transform=tf_test
            )
            test_ds = DatasetClass(
                self.data_dir, train=False, download=False, transform=tf_test
            )
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)
        g = torch.Generator().manual_seed(self.seed)

        def seed_worker(worker_id):
            np.random.seed(self.seed + worker_id)
            random.seed(self.seed + worker_id)

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=seed_worker,
            generator=g,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=valid_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=seed_worker,
            generator=g,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=seed_worker,
            generator=g,
        )
        return (train_loader, valid_loader, test_loader)

    def log_args(self):
        cfg = {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_") and k != "base_results_dir"
        }
        if "device" in cfg:
            cfg["device"] = str(cfg["device"])
        if "orig_results_dir" in cfg:
            cfg["results_dir"] = self.orig_results_dir
        print("Experiment configuration:")
        print(json.dumps(cfg, indent=2))

    def train(self, analyze_freq: int = 1):
        train_loader, valid_loader, test_loader = self.get_data_loaders()
        NUM_CLASSES = {
            "cifar10": 10,
            "cifar100": 100,
            "tinyimg": 200,
            "tiny-imagenet": 200,
        }
        num_classes = NUM_CLASSES[self.dataset]
        model, layer_type = self._prepare_model(num_classes)
        _disable_inplace_relu(model)
        injector = HookInjector(model)
        if self.reg:
            self.label_smoothing_train = 0.05
        crit_train = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing_train)
        crit_val = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing_val)
        if self.reg:
            opt = getattr(optim, self.optimizer_name)(
                model.parameters(), lr=self.learning_rate, weight_decay=0.005
            )
        else:
            opt = getattr(optim, self.optimizer_name)(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=0.0001,
                nesterov=True,
                momentum=0.9,
            )
        sched = MultiStepLR(opt, milestones=self.milestones)
        logs = {"train": OrderedDict(), "val": OrderedDict()}
        analysis_containers: Dict[str, Dict[int, Any]] = {
            "acts_hists": {},
            "acts_minmax": {},
            "grads_hists": {},
            "grads_minmax": {},
            "nonzero_counts": {},
            "nonzero_neuron_acts": {},
        }
        for ep in range(self.num_epochs):
            model.train()
            tl = 0.0
            for x, y in tqdm(
                train_loader,
                desc=f"Epoch {ep + 1}/{self.num_epochs}",
                disable=not sys.stdout.isatty(),
            ):
                x, y = (x.to(self.device), y.to(self.device))
                loss = crit_train(model(x), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tl += loss.item()
            logs["train"][ep] = tl / len(train_loader)
            model.eval()
            vl = 0.0
            with torch.no_grad():
                for x, y in valid_loader:
                    x, y = (x.to(self.device), y.to(self.device))
                    vl += crit_val(model(x), y).item()
            logs["val"][ep] = vl / len(valid_loader)
            sched.step()
            print(
                f"Epoch {ep + 1}: Train={logs['train'][ep]:.4f}, Val={logs['val'][ep]:.4f}"
            )
            prev = torch.are_deterministic_algorithms_enabled()
            torch.use_deterministic_algorithms(False)
            run_analysis_epoch(
                epoch=ep,
                num_epochs=self.num_epochs,
                model=model,
                train_loader=train_loader,
                criterion=crit_train,
                injector=injector,
                hooks=hooks,
                device=self.device,
                layer_type=layer_type,
                analyze_freq=analyze_freq,
                containers=analysis_containers,
            )
            torch.use_deterministic_algorithms(prev)
        model.eval()
        correct_fgi, total = (0, 0)
        with torch.no_grad():
            for x, y in test_loader:
                x, y = (x.to(self.device), y.to(self.device))
                preds = model(x).argmax(dim=1)
                correct_fgi += (preds == y).sum().item()
                total += y.size(0)
        fgi_acc = 100 * correct_fgi / total
        function = activation_function(self.fgi_type, self.alpha)

        class Surrogate(nn.Module):
            def forward(self, x):
                return function(x)

        replace_layers(model, layer_type, lambda: Surrogate())
        model.eval()
        correct_nonfgi = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = (x.to(self.device), y.to(self.device))
                preds = model(x).argmax(dim=1)
                correct_nonfgi += (preds == y).sum().item()
        nfgi_acc = 100 * correct_nonfgi / total
        print(f"Non‑FGI Test Accuracy: {nfgi_acc:.2f}%")
        out_dir = self._make_results_path()
        results: Dict[str, Any] = {
            "dataset": self.dataset,
            "fgi_type": self.fgi_type,
            "alpha": self.alpha,
            "seed": self.seed,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer_name,
            "milestones": self.milestones,
            "train_losses": logs["train"],
            "val_losses": logs["val"],
            "test_accuracy": fgi_acc,
            "non_fgi_test_accuracy": nfgi_acc,
            "model_state_dict": model.state_dict(),
            "acts_hists": analysis_containers["acts_hists"],
            "acts_minmax": analysis_containers["acts_minmax"],
            "grads_hists": analysis_containers["grads_hists"],
            "grads_minmax": analysis_containers["grads_minmax"],
            "nonzero_counts": analysis_containers["nonzero_counts"],
            "nonzero_neuron_acts": analysis_containers["nonzero_neuron_acts"],
        }
        filename = f"run_acc_{fgi_acc:.2f}.pt"
        torch.save(results, os.path.join(out_dir, filename))
        print(f"Saved all results + model to {os.path.join(out_dir, filename)}")
        return results
