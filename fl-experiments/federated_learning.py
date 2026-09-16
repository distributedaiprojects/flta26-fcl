import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import copy
import importlib.util

import csv
import os
from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, average_precision_score, auc
from sklearn.preprocessing import label_binarize
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import sys
from collections import deque
import random
from typing import List, Dict, OrderedDict, Tuple
import pandas as pd
from datetime import datetime

SEED = 200
datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Use relative path based on script location
results_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(results_dir, exist_ok=True)
results_info_path = os.path.join(results_dir, f"results_info_{datetime_str}_exp{str(sys.argv[1])}.txt")

class SimpleNet(nn.Module):
    def __init__(self, input_channels=1, input_size=28, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(input_channels*input_size*input_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.fc(self.flatten(x))

class FedAvgClient:
    def __init__(self, model, data_loader, device, sample_selection=False, selection_ratio=0.8, fedprox_mu=0.0, global_model=None):
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.optimizer = optim.SGD(model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.sample_selection = sample_selection
        self.selection_ratio = selection_ratio
        self.sample_history = deque(maxlen=100)  # Store sample loss history
        self.probability_vector = None  # Store empirical probability vector
        self.fedprox_mu = fedprox_mu  # FedProx proximal coefficient
        self.global_model = global_model  # Global model for FedProx
    
    def generate_sample_guidance(self, dataset, global_model, selection_ratio=0.7, method='adaptive'):
        """Generate sample selection guidance using global model"""
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        global_model.eval()
        sample_difficulties = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = global_model(data)
                
                # Calculate prediction confidence (lower = harder sample)
                probs = torch.softmax(output, dim=1)
                max_probs = torch.max(probs, dim=1)[0]
                
                for i, confidence in enumerate(max_probs):
                    sample_idx = batch_idx * data.size(0) + i
                    sample_difficulties.append((sample_idx, 1.0 - confidence.item()))
        
        # Sort by difficulty (higher = more difficult)
        sample_difficulties.sort(key=lambda x: x[1], reverse=True)
        num_samples = len(sample_difficulties)
        
        if method == 'adaptive':
            # Adaptive selection: mix of hard and medium difficulty samples
            num_select = max(1, int(num_samples * selection_ratio))
            hard_ratio = 0.6  # 60% hard samples
            medium_ratio = 0.4  # 40% medium samples
            
            num_hard = int(num_select * hard_ratio)
            num_medium = num_select - num_hard
            
            # Select hardest samples
            hard_indices = [idx for idx, _ in sample_difficulties[:num_hard]]
            
            # Select medium difficulty samples (middle range)
            mid_start = num_samples // 3
            mid_end = 2 * num_samples // 3
            medium_samples = sample_difficulties[mid_start:mid_end]
            medium_indices = [idx for idx, _ in medium_samples[:num_medium]]
            
            selected_indices = hard_indices + medium_indices
            
        elif method == 'hard':
            # Select hardest samples only
            num_select = max(1, int(num_samples * selection_ratio))
            selected_indices = [idx for idx, _ in sample_difficulties[:num_select]]
            
        else:  # random
            num_select = max(1, int(num_samples * selection_ratio))
            all_indices = [idx for idx, _ in sample_difficulties]
            selected_indices = random.sample(all_indices, num_select)
        
        return selected_indices
    
    def select_samples(self, dataset):
        """Select samples based on loss-based prioritization"""
        if not self.sample_selection or len(self.sample_history) == 0:
            return dataset
        
        # Calculate sample priorities based on historical losses
        sample_losses = {}
        for sample_info in self.sample_history:
            idx, loss = sample_info['idx'], sample_info['loss']
            if idx not in sample_losses:
                sample_losses[idx] = []
            sample_losses[idx].append(loss)
        
        # Calculate average loss per sample
        sample_priorities = {}
        for idx, losses in sample_losses.items():
            avg_loss = np.mean(losses)
            sample_priorities[idx] = avg_loss
        
        # Select samples with higher average loss (harder samples)
        if len(sample_priorities) > 0:
            sorted_samples = sorted(sample_priorities.items(), key=lambda x: x[1], reverse=True)
            num_select = max(1, int(len(sorted_samples) * self.selection_ratio))
            selected_indices = [idx for idx, _ in sorted_samples[:num_select]]
            
            # Add random samples if not enough historical data
            remaining = len(dataset) - len(selected_indices)
            if remaining > 0:
                all_indices = set(range(len(dataset)))
                available = list(all_indices - set(selected_indices))
                additional = random.sample(available, min(remaining, int(len(dataset) * (1 - self.selection_ratio))))
                selected_indices.extend(additional)
            
            return Subset(dataset, selected_indices)
        
        return dataset
    
    def compute_data_smoothing(self, dataset, num_classes=10):
        """
        Compute smoothed empirical probability vector using Laplace correction.
        
        Args:
            dataset: The dataset to compute class probabilities for
            num_classes: Number of classes (k in the formula)
            
        Returns:
            np.ndarray: Smoothed empirical probability vector of shape (num_classes,)
        """
        # Count examples per class
        class_counts = np.zeros(num_classes)
        
        # Count class occurrences in the dataset
        for i in range(len(dataset)):
            _, label = dataset[i]
            class_counts[label] += 1
        
        # Total number of examples
        total_examples = len(dataset)
        
        # Apply Laplace smoothing: p̂_i(D) = (n_i + 1) / (|D| + k)
        smoothed_probs = (class_counts + 1) / (total_examples + num_classes)
        
        return smoothed_probs
    
    def get_empirical_probability_vector(self, dataset, num_classes=10):
        """
        Get the empirical probability vector for the dataset.
        
        Returns both raw and smoothed probability vectors.
        
        Args:
            dataset: The dataset
            num_classes: Number of classes
            
        Returns:
            dict: Contains 'raw' and 'smoothed' probability vectors
        """
        # Compute raw empirical probabilities
        class_counts = np.zeros(num_classes)
        for i in range(len(dataset)):
            _, label = dataset[i]
            class_counts[label] += 1
        
        raw_probs = class_counts / len(dataset)
        
        # Compute smoothed probabilities using Laplace correction
        smoothed_probs = self.compute_data_smoothing(dataset, num_classes)
        
        return {
            'raw': raw_probs,
            'smoothed': smoothed_probs,
            'class_counts': class_counts,
            'total_examples': len(dataset)
        }
    
    def train(self, epochs=1):
        self.model.train()
        
        for _ in range(epochs):
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Add FedProx proximal term if enabled
                if self.fedprox_mu is not None and self.global_model is not None:
                    proximal_term = 0.0
                    for param, global_param in zip(self.model.parameters(), self.global_model.parameters()):
                        proximal_term += torch.norm(param - global_param, p=2)**2
                    loss += (self.fedprox_mu / 2) * proximal_term
                
                loss.backward()
                self.optimizer.step()
                
                # Store sample losses for future selection
                if self.sample_selection:
                    batch_size = data.size(0)
                    individual_losses = nn.CrossEntropyLoss(reduction='none')(output, target)
                    for i, sample_loss in enumerate(individual_losses):
                        sample_idx = batch_idx * batch_size + i
                        self.sample_history.append({
                            'idx': sample_idx,
                            'loss': sample_loss.item()
                        })
        
        return self.model.state_dict()

class FedAvgServer:
    def __init__(self, model, is_client_server=False, client_ids=None):
        self.global_model = model
        self.is_client_server = is_client_server
        self.client_ids = client_ids if client_ids is not None else []
    
    def aggregate(self, client_models, client_weights):
        global_dict = self.global_model.state_dict()
        for key in global_dict.keys():
            global_dict[key] = torch.stack([
                client_models[i][key] * client_weights[i] 
                for i in range(len(client_models))
            ]).sum(dim=0)
        self.global_model.load_state_dict(global_dict)
        return self.global_model.state_dict()
    
    def select_clients(self, client_accuracies, client_datasets, device, selection_method, gradient_diversity_method, selection_ratio, input_channels=1, input_size=28, num_classes=10):
        # Use consistent seed for all servers
        server_seed = SEED
        
        if selection_method == 'accuracy':
            return select_clients(client_accuracies, selection_ratio)
        elif selection_method == 'gradient_diversity':
            if gradient_diversity_method == 'cosine':
                return select_clients_by_gradient_diversity(
                    self.global_model, client_datasets, device, selection_ratio, server_seed, input_channels, input_size, num_classes)
            elif gradient_diversity_method == 'euclidean':
                return select_clients_by_gradient_diversity_euclidean(
                    self.global_model, client_datasets, device, selection_ratio, server_seed, input_channels, input_size, num_classes)
        elif selection_method == 'random':
            return select_clients_randomly(len(client_datasets), selection_ratio)
        return list(range(len(client_datasets)))
    
def create_client_data(dataset, num_clients=10, iid=True, classes_per_client=2, round_num=0, dynamic_redistribution=False, client_label_config=None, num_classes=10):
    if iid:
        # IID: Random shuffle and equal split
        indices = torch.randperm(len(dataset)).tolist()
        data_per_client = len(dataset) // num_clients
        client_datasets = []
        
        for i in range(num_clients):
            start_idx = i * data_per_client
            end_idx = start_idx + data_per_client if i < num_clients - 1 else len(dataset)
            client_indices = indices[start_idx:end_idx]
            client_datasets.append(Subset(dataset, client_indices))
    else:
        # Non-IID: Each client gets limited classes
        # Handle different dataset types
        if hasattr(dataset, 'targets'):
            labels = np.array(dataset.targets)
        elif isinstance(dataset, TensorDataset):
            labels = dataset.tensors[1].numpy()
        else:
            raise ValueError("Unsupported dataset type")
            
        client_datasets = []
        
        for i in range(num_clients):
            # Use manual configuration if provided
            if client_label_config and round_num in client_label_config and i in client_label_config[round_num]:
                client_classes = client_label_config[round_num][i]
            else:
                # Calculate class offset based on round for dynamic redistribution
                if dynamic_redistribution:
                    cycle_position = (round_num // 5) % 2  # Cycles between 0 and 1
                    round_offset = cycle_position * classes_per_client
                else:
                    round_offset = 0
                
                # Assign classes to client (cycling through classes with round offset)
                client_classes = [(i * classes_per_client + j + round_offset) % num_classes for j in range(classes_per_client)]
            client_indices = []
            
            for class_id in client_classes:
                class_indices = np.where(labels == class_id)[0]
                samples_per_class = len(class_indices) // (num_clients // classes_per_client + 1)
                start = (i // classes_per_client) * samples_per_class
                end = start + samples_per_class
                client_indices.extend(class_indices[start:end])
            
            client_datasets.append(Subset(dataset, client_indices))
    
    # Compute probability vectors for each client and combine into unified list
    client_data = []
    for i, client_ds in enumerate(client_datasets):
        class_counts = np.zeros(num_classes)
        for idx in client_ds.indices:
            if hasattr(dataset, 'targets'):
                label = dataset.targets[idx]
            elif isinstance(dataset, TensorDataset):
                label = dataset.tensors[1][idx].item()
            class_counts[label] += 1
        
        total_examples = len(client_ds)
        raw_probs = class_counts / total_examples if total_examples > 0 else class_counts
        smoothed_probs = (class_counts + 1) / (total_examples + num_classes)
        
        probability_vector = {
            'raw': raw_probs.tolist(),
            'smoothed': smoothed_probs.tolist(),
            'class_counts': class_counts.tolist(),
            'total_examples': int(total_examples)
        }
        
        client_data.append({
            'dataset': client_ds,
            'probability_vector': probability_vector
        })
    
    return client_data

def get_client_labels(client_datasets, dataset):
    """Extract labels present in each client's dataset"""
    client_labels = {}
    for i, client_ds in enumerate(client_datasets):
        labels = set()
        for idx in client_ds.indices:
            if hasattr(dataset, 'targets'):
                labels.add(int(dataset.targets[idx]))
            elif isinstance(dataset, TensorDataset):
                labels.add(int(dataset.tensors[1][idx].item()))
        client_labels[i] = sorted(list(labels))
    return client_labels

def visualize_label_distribution(client_datasets, dataset):
    labels = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i, client_ds in enumerate(client_datasets):
        label_counts = np.zeros(10)
        for idx in client_ds.indices:
            label = dataset.targets[idx]
            label_counts[label] += 1
        
        axes[i].bar(range(10), label_counts)
        axes[i].set_title(f'Client {i+1}')
        axes[i].set_xticks(range(10))
        axes[i].set_xticklabels([l[:3] for l in labels], rotation=45)
    
    plt.tight_layout()
    plt.savefig('client_label_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()

def save_round_to_csv(result, filename=None):
    if filename is None:
        filename = f'{results_info_path.replace(".txt", "")}_federated_results_automated_tmp.csv'
    file_exists = os.path.exists(filename)
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = ['experiment_id','dataset','iid_setting', 'classes_per_client', 'round', 'global_accuracy', 'client_accuracies', 'selected_clients', 'num_selected_clients', 'selection_method', 'gradient_diversity_method', 'client_selection_ratio', 'server_type', 'server_clients', 'selection_counts', 'aggregation_client', 'selection_client', 'server_selections', 'client_sample_counts', 'use_sample_selection', 'sample_selection_ratio', 'sample_selection_method', 'server_guidance_method', 'dynamic_redistribution', 'client_labels', 'client_label_metrics']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(result)

def compute_gradient_parallel(args):
    client_id, global_state_dict, client_dataset, device_str, input_channels, input_size, num_classes = args
    device = torch.device(device_str)
    
    # Create model and load global state
    model = SimpleNet(input_channels=input_channels, input_size=input_size, num_classes=num_classes).to(device)
    model.load_state_dict(global_state_dict)
    
    # Use only a subset of client data for gradient computation
    subset_size = min(100, len(client_dataset))  # Use max 100 samples
    subset_indices = list(range(subset_size))
    subset_dataset = Subset(client_dataset, subset_indices)
    
    # Create data loader (no shuffle for consistent gradients)
    data_loader = DataLoader(subset_dataset, batch_size=32, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    
    # Compute gradient on one batch
    model.train()
    data, target = next(iter(data_loader))
    data, target = data.to(device), target.to(device)
    
    model.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    
    # Extract gradients
    gradients = []
    for param in model.parameters():
        if param.grad is not None:
            gradients.append(param.grad.flatten())
    
    gradient_vector = torch.cat(gradients)
    return client_id, gradient_vector.cpu()

def select_clients_by_gradient_diversity_euclidean(global_model, client_datasets, device, selection_ratio=0.7, seed=None, input_channels=1, input_size=28, num_classes=10):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    num_clients = len(client_datasets)
    num_selected = max(1, int(num_clients * selection_ratio))
    
    # Compute gradients for all clients
    global_state_dict = global_model.state_dict()
    gradient_args = [(i, global_state_dict, client_datasets[i], str(device), input_channels, input_size, num_classes) 
                    for i in range(num_clients)]
    
    gradient_results = [compute_gradient_parallel(args) for args in gradient_args]
    
    # Extract gradients in order
    gradients = [None] * num_clients
    for client_id, grad in gradient_results:
        gradients[client_id] = grad
    
    # Select diverse clients using Euclidean distance
    start_client = int(np.random.randint(0, num_clients)) if seed is not None else 0
    selected_clients = [start_client]  # Start with random client
    
    for _ in range(num_selected - 1):
        max_diversity = -1
        candidates = []  # Track ties
        
        for i in range(num_clients):
            if i in selected_clients:
                continue
            
            # Compute minimum Euclidean distance with selected clients
            min_distance = float('inf')
            for selected in selected_clients:
                distance = torch.norm(gradients[i] - gradients[selected], p=2)
                min_distance = min(min_distance, distance.item())
            
            # Track candidates with maximum diversity
            if min_distance > max_diversity:
                max_diversity = min_distance
                candidates = [i]
            elif abs(min_distance - max_diversity) < 1e-6:  # Handle ties
                candidates.append(i)
        
        if candidates:
            best_client = int(np.random.choice(candidates)) if seed is not None else candidates[0]
            selected_clients.append(best_client)
    
    return selected_clients

def select_clients_by_gradient_diversity(global_model, client_datasets, device, selection_ratio=0.7, seed=None, input_channels=1, input_size=28, num_classes=10):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    num_clients = len(client_datasets)
    num_selected = max(1, int(num_clients * selection_ratio))
    
    # Compute gradients for all clients
    global_state_dict = global_model.state_dict()
    gradient_args = [(i, global_state_dict, client_datasets[i], str(device), input_channels, input_size, num_classes) 
                    for i in range(num_clients)]
    
    gradient_results = [compute_gradient_parallel(args) for args in gradient_args]
    
    # Extract gradients in order
    gradients = [None] * num_clients
    for client_id, grad in gradient_results:
        gradients[client_id] = grad
    
    # Select diverse clients using cosine similarity
    start_client = int(np.random.randint(0, num_clients)) if seed is not None else 0
    selected_clients = [start_client]  # Start with random client
    
    for _ in range(num_selected - 1):
        max_diversity = -1
        candidates = []  # Track ties
        
        for i in range(num_clients):
            if i in selected_clients:
                continue
            
            # Compute minimum cosine similarity with selected clients
            min_similarity = float('inf')
            for selected in selected_clients:
                similarity = torch.cosine_similarity(gradients[i], gradients[selected], dim=0)
                min_similarity = min(min_similarity, similarity.item())
            
            # Track candidates with maximum diversity (minimum similarity)
            diversity = -min_similarity
            if diversity > max_diversity:
                max_diversity = diversity
                candidates = [i]
            elif abs(diversity - max_diversity) < 1e-6:  # Handle ties
                candidates.append(i)
        
        if candidates:
            best_client = int(np.random.choice(candidates)) if seed is not None else candidates[0]
            selected_clients.append(best_client)
    
    return selected_clients


def select_clients_randomly(num_clients, selection_ratio=0.7):
    num_selected = max(1, int(num_clients * selection_ratio))
    selected_clients = np.random.choice(num_clients, num_selected, replace=False).tolist()
    return sorted(selected_clients)

def select_clients(client_accuracies, selection_ratio=0.7):
    num_clients = len(client_accuracies)
    num_selected = max(1, int(num_clients * selection_ratio))
    
    # Select top performing clients
    client_scores = [(i, acc) for i, acc in enumerate(client_accuracies)]
    client_scores.sort(key=lambda x: x[1], reverse=True)
    selected_indices = [idx for idx, _ in client_scores[:num_selected]]
    
    return selected_indices

def train_client_parallel(args):
    client_id, global_state_dict, client_dataset, device_str, epochs, sample_selection, selection_ratio, selection_method, server_guidance_method, teachers, input_channels, input_size, num_classes, fedprox_mu, kd_ll_ratio, kd_gl_ratio, kd_temperature, kd_weight, use_cfed, cfed_alpha, cfed_temperature = args
    device = torch.device(device_str)
    
    # Create model and load global state
    model = SimpleNet(input_channels=input_channels, input_size=input_size, num_classes=num_classes).to(device)
    model.load_state_dict(global_state_dict)
    
    # Create global model for FedProx
    global_model = SimpleNet(input_channels=input_channels, input_size=input_size, num_classes=num_classes).to(device)
    global_model.load_state_dict(global_state_dict)
    global_model.eval()  # Global model should not be trained

    old_global_model = None
    if use_cfed:
        old_global_model = SimpleNet(input_channels=input_channels, input_size=input_size, num_classes=num_classes).to(device)
        old_global_model.load_state_dict(global_state_dict)
        old_global_model.eval()
    
    # Check if teachers are available for this client
    has_teachers = teachers is not None and client_id in teachers
    teacher_ll = None
    teacher_gl = None
    
    if has_teachers:
        # Load LL (Local Leader) teacher if available
        if 'LL' in teachers[client_id]:
            teacher_ll = SimpleNet(input_channels=input_channels, input_size=input_size, num_classes=num_classes).to(device)
            teacher_ll.load_state_dict(teachers[client_id]['LL'])
            teacher_ll.eval()
        
        # Load GL (Global Leader) teacher if available
        if 'GL' in teachers[client_id]:
            teacher_gl = SimpleNet(input_channels=input_channels, input_size=input_size, num_classes=num_classes).to(device)
            teacher_gl.load_state_dict(teachers[client_id]['GL'])
            teacher_gl.eval()
    
    original_size = len(client_dataset)
    selected_size = original_size
    
    # Apply sample selection if enabled
    if sample_selection:
        if selection_method == 'server_assisted':
            # Client generates its own guidance using global model
            client = FedAvgClient(model, None, device)
            selected_indices = client.generate_sample_guidance(
                client_dataset, model, selection_ratio, server_guidance_method)
            
        elif selection_method == 'hard':
            # Hard sample mining: select samples with higher loss
            data_loader = DataLoader(client_dataset, batch_size=32, shuffle=False)
            model.eval()
            sample_losses = []
            
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(data_loader):
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    losses = nn.CrossEntropyLoss(reduction='none')(output, target)
                    for i, loss in enumerate(losses):
                        sample_idx = batch_idx * data.size(0) + i
                        sample_losses.append((sample_idx, loss.item()))
            
            # Select hardest samples
            sample_losses.sort(key=lambda x: x[1], reverse=True)
            num_select = max(1, int(len(sample_losses) * selection_ratio))
            selected_indices = [idx for idx, _ in sample_losses[:num_select]]
            
        elif selection_method == 'random':
            # Random sample selection
            num_select = max(1, int(original_size * selection_ratio))
            selected_indices = random.sample(range(original_size), num_select)
        
        selected_size = len(selected_indices)
        selected_dataset = Subset(client_dataset, selected_indices)
        data_loader = DataLoader(selected_dataset, batch_size=32, shuffle=True)
        dataset_used_for_stats = selected_dataset
    else:
        data_loader = DataLoader(client_dataset, batch_size=32, shuffle=True)
        dataset_used_for_stats = client_dataset
    
    # Train
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    temperature = kd_temperature  # Use grid search parameter

    model.train()
    total_loss = 0.0
    total_steps = 0
    
    for _ in range(epochs):
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            
            # Standard cross-entropy loss
            loss = criterion(output, target)
            
            # If teachers are available, add knowledge distillation loss
            if has_teachers:
                distill_loss_teacher1 = torch.zeros((), device=output.device, dtype=output.dtype)
                distill_loss_teacher2 = torch.zeros((), device=output.device, dtype=output.dtype)
                num_teachers = 0
                student_logp = nn.functional.log_softmax(output / temperature, dim=1)

                # Knowledge distillation from Local Leader teacher
                if teacher_ll is not None:
                    with torch.no_grad():
                        t = teacher_ll(data)
                        t_prob = nn.functional.softmax(t / temperature, dim=1)
                    distill_loss_teacher1 += nn.functional.kl_div(student_logp, t_prob, reduction="batchmean")
                    num_teachers += 1
                
                # Knowledge distillation from Global Leader teacher
                if teacher_gl is not None:
                    with torch.no_grad():
                        t = teacher_gl(data)
                        t_prob = nn.functional.softmax(t / temperature, dim=1)
                    distill_loss_teacher2 += nn.functional.kl_div(student_logp, t_prob, reduction="batchmean")
                    num_teachers += 1
               
                # Weighted combination of CE and distillation losses
                if num_teachers > 0:
                    # Combine both teachers: kd_ll_ratio% LL + kd_gl_ratio% GL
                    combined_distill = (kd_ll_ratio * distill_loss_teacher1 + kd_gl_ratio * distill_loss_teacher2)
                    # combined_distill = kd_ll_ratio * distill_loss_teacher1
                    # KD weight: (1 - kd_weight)% CE + kd_weight% * T² * distillation
                    ce_weight = 1.0 - kd_weight
                    loss = ce_weight * loss + kd_weight * (temperature ** 2) * combined_distill
                    debug_msg = f"Client {client_id}: CE Loss={loss.item():.4f}, LL Distill Loss={distill_loss_teacher1.item():.4f}, GL Distill Loss={distill_loss_teacher2.item():.4f}"
                    # print(debug_msg)
                    # with open(results_info_path, 'a') as f:
                    #     f.write(debug_msg + '\n')

            # If CFeD mode is enabled, distill from the previous global model
            if use_cfed and old_global_model is not None:
                with torch.no_grad():
                    old_output = old_global_model(data)
                student_logp = nn.functional.log_softmax(output / cfed_temperature, dim=1)
                teacher_prob = nn.functional.softmax(old_output / cfed_temperature, dim=1)
                distill_loss = nn.KLDivLoss(reduction="batchmean")(student_logp, teacher_prob)
                loss = (1.0 - cfed_alpha) * loss + cfed_alpha * (cfed_temperature ** 2) * distill_loss

            # Add FedProx proximal term if enabled
            if fedprox_mu != None:
                proximal_term = 0.0
                for param, global_param in zip(model.parameters(), global_model.parameters()):
                    proximal_term += torch.norm(param - global_param, p=2)**2
                loss += (fedprox_mu / 2) * proximal_term
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_steps += 1
        
        client_loss = total_loss / total_steps if total_steps > 0 else 0.0

    return client_id, model.state_dict(), selected_size, client_loss

def evaluate_per_label_metrics(model, test_dataset, label_set, device):
    """Compute accuracy and PR-AUC for a given set of labels on the test dataset."""
    if hasattr(test_dataset, 'targets'):
        all_labels = np.array(test_dataset.targets)
    else:
        all_labels = test_dataset.tensors[1].numpy()
    indices = np.where(np.isin(all_labels, label_set))[0]
    if len(indices) == 0:
        return 0.0, 0.0
    subset = Subset(test_dataset, indices)
    loader = DataLoader(subset, batch_size=256, shuffle=False)
    model.eval()
    all_targets, all_probs = [], []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            probs = torch.softmax(model(data), dim=1)
            all_targets.append(target.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)
    predicted = all_probs.argmax(axis=1)
    accuracy = round(float(100.0 * (predicted == all_targets).mean()), 4)
    # PR-AUC: one-vs-rest for each label in label_set, then average
    pr_aucs = []
    for lbl in label_set:
        binary = (all_targets == lbl).astype(int)
        if binary.sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(binary, all_probs[:, lbl])
        pr_aucs.append(auc(recall, precision))
    pr_auc = round(float(np.mean(pr_aucs)) if pr_aucs else 0.0, 4)
    return accuracy, pr_auc

def evaluate_client_accuracies(model, client_loaders, device):
    client_accuracies = []
    for loader in client_loaders:
        accuracy = evaluate_model(model, loader, device)
        client_accuracies.append(round(accuracy, 2))
    return client_accuracies

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total

def _as_array(prob_vectors: List[List[float]]) -> np.ndarray:
    X = np.asarray(prob_vectors, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got {X.shape}")
    if X.shape[0] < 1:
        raise ValueError("Need at least 1 sample.")
    return X

def kmeans_optimal_k_elbow_keep_models(
    prob_vectors: List[List[float]],
    k_min: int = 1,
    k_max: int = 10,
    n_init: int = 100,
    random_state: int = 42,
    improvement_threshold: float = 0.1,
) -> Tuple[int, np.ndarray, KMeans, Dict[int, float]]:
    """
    Finds optimal k via elbow method using inertia (within-cluster sum of squares).
    Selects k where the relative improvement in inertia drops below improvement_threshold.
    Includes k=1 for comparison.

    Returns:
      best_k
      best_labels
      best_model   (fitted KMeans)
      inertia_by_k
    """
    X = _as_array(prob_vectors)
    n = X.shape[0]

    if k_max is None:
        k_max = min(10, n)  # Allow up to n clusters

    k_min = max(1, k_min)  # Allow k=1
    k_max = min(k_max, n)  # But not more than n

    if k_min > k_max:
        raise ValueError("Invalid k range")

    inertia_by_k: Dict[int, float] = {}
    models_by_k: Dict[int, KMeans] = {}
    labels_by_k: Dict[int, np.ndarray] = {}

    # Fit models for all k
    for k in range(k_min, k_max + 1):
        model = KMeans(
            n_clusters=k,
            n_init=n_init,
            random_state=random_state,
        )
        labels = model.fit_predict(X)
        inertia = model.inertia_
        inertia_by_k[k] = inertia
        models_by_k[k] = model
        labels_by_k[k] = labels

    # Find elbow point using improvement threshold
    best_k = k_min
    prev_inertia = inertia_by_k[k_min]

    for k in range(k_min + 1, k_max + 1):
        current_inertia = inertia_by_k[k]
        if prev_inertia == 0:
            improvement = 1.0  # Avoid division by zero
        else:
            improvement = (prev_inertia - current_inertia) / prev_inertia

        if improvement < improvement_threshold:
            best_k = k - 1  # Use the previous k
            break
        else:
            best_k = k
            prev_inertia = current_inertia

    # If we reached max_k, use it
    if best_k > k_max:
        best_k = k_max

    return best_k, labels_by_k[best_k], models_by_k[best_k], inertia_by_k

def find_local_global_leaders(df_clustering: pd.DataFrame):
    """ Find local and global leaders based on clustering of client probability vectors."""           
    df_clustering["local_leader"] = (
    df_clustering.index.isin(
        df_clustering.groupby("cluster")["loss"].idxmin()
    ).astype(int)
)
    
def find_global_leader(df_clustering: pd.DataFrame):
    """ Find global leader based on clustering of client probability vectors."""           
    df_clustering["global_leader"] = 0
    global_leader_idx = df_clustering["loss"].idxmin()
    df_clustering.at[global_leader_idx, "global_leader"] = 1

def define_local_teacher_for_all_clients(df_clustering: pd.DataFrame, client_model_weights: List[Dict[str, any]]):
    """ Define local teachers: aggregated model per client that includes:
        - a current local leader
        - a current client's model
        - a randomly selected client from the same cluster
    """           
    teachers = {}
    for _, row in df_clustering.iterrows():
        client_id = int(row["client_id"])
        teachers[client_id] = {}
        # Get the local leader in the same cluster
        local_leader_id = df_clustering[(df_clustering["cluster"] == row["cluster"]) & (df_clustering["local_leader"] == 1)]["client_id"].values[0]
        # Get all clients in same cluster
        cluster_clients = df_clustering[df_clustering["cluster"] == row["cluster"]]["client_id"].values
        # Pick random client from same cluster
        random_client_id = random.choice(cluster_clients)
        # get model weights for a local leader, random client, and current client
        random_client_model_weights = client_model_weights[random_client_id]
        local_leader_model_weights = client_model_weights[local_leader_id]
        current_client_model_weights = client_model_weights[client_id] 

        if cluster_clients.shape[0] >= 3:
            teachers[client_id]['LL'] = weighted_average_state_dicts([random_client_model_weights, local_leader_model_weights, current_client_model_weights])

        print_string = f"Client {client_id}: Cluster {row['cluster']} local teacher = LL(Local leader {local_leader_id}, random client {random_client_id})"
        with open(results_info_path, 'a') as f:
            f.write(print_string + '\n')
        print(print_string)
    return teachers

def define_global_teacher_for_all_clients(df_clustering: pd.DataFrame, teachers: Dict[int, Dict[str, any]], client_model_weights: List[Dict[str, any]]):
    """ Define global teachers: aggregated model per client that includes:
        - a current global leader
        - a current client's model
        - a randomly selected client from all clients
    """                     
    for _, row in df_clustering.iterrows():
        client_id = int(row["client_id"])
        # Get the global leader
        global_leader_id = df_clustering[df_clustering["global_leader"] == 1]['client_id'].values[0]
        # Pick random client from same cluster
        random_client_id = random.choice(df_clustering["client_id"].values)
        # get model weights for a local leader, random client, and current client
        random_client_model_weights = client_model_weights[random_client_id]
        global_leader_model_weights = client_model_weights[global_leader_id]
        current_client_model_weights = client_model_weights[client_id] 
        teachers[client_id]['GL'] = weighted_average_state_dicts([random_client_model_weights, global_leader_model_weights, current_client_model_weights])
        
        print_string = f"Client {client_id}: Cluster {row['cluster']} global teacher = GL(Global leader: {global_leader_id}, random client: {random_client_id})"
        with open(results_info_path, 'a') as f:
            f.write(print_string + '\n')
        print(print_string)
    return teachers

def weighted_average_state_dicts(state_dicts):
    avg_sd = OrderedDict()

    for key in state_dicts[0].keys():
        avg_sd[key] = sum(
            sd[key]
            for sd in state_dicts
        ) / len(state_dicts)
    return avg_sd

def main():

    spec = importlib.util.spec_from_file_location("config", os.path.join(os.path.dirname(__file__), "conf", "exp"+str(sys.argv[1])+".py"))
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    
    # Debug: print what's in config
    config_attrs = [attr for attr in dir(config) if not attr.startswith('_')]
    print(f"[DEBUG] Config attributes: {config_attrs[:10]}...")  # Show first 10
    
    # Check if grid search parameters are provided
    grid_params = getattr(config, 'GRID_SEARCH_PARAMS', None)
    print(f"[DEBUG] GRID_SEARCH_PARAMS found: {grid_params is not None}")
    if grid_params:
        print(f"[DEBUG] GRID_SEARCH_PARAMS value: {grid_params}")
    
    if grid_params:
        KD_LL_RATIO = grid_params.get('ll_ratio', 0.9)
        KD_GL_RATIO = 1.0 - KD_LL_RATIO
        KD_TEMPERATURE = grid_params.get('temperature', 1.5)
        KD_WEIGHT = grid_params.get('kd_weight', 0.05)
    else:
        KD_LL_RATIO = 0.9
        KD_GL_RATIO = 1.0 - KD_LL_RATIO
        KD_TEMPERATURE = 1.5 
        KD_WEIGHT = 0.05

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # Print KD configuration
    if grid_params:
        print(f"\n[GRID SEARCH MODE]")
        print(f"KD Parameters: LL={KD_LL_RATIO:.2f}, GL={KD_GL_RATIO:.2f}, T={KD_TEMPERATURE}, KD_w={KD_WEIGHT}")
    else:
        print(f"\n[STANDARD MODE]")
        print(f"KD Parameters: LL={KD_LL_RATIO:.2f}, GL={KD_GL_RATIO:.2f}, T={KD_TEMPERATURE}, KD_w={KD_WEIGHT}\n")
    #server_client_ids_args = sys.argv[1].split(",",-1)  # Reset command line arguments
    #server_client_ids_args_intlist = [int(_) for _ in server_client_ids_args]

    # Use configuration from exp18.py
    num_clients = getattr(config, 'num_clients', 10)
    iid_setting = config.iid_setting
    classes_per_client = config.classes_per_client
    use_client_selection = config.use_client_selection
    selection_method = config.selection_method
    selection_ratio = config.selection_ratio
    client_as_server = config.client_as_server
    server_client_ids = config.server_client_ids
    aggregation_client_id = config.aggregation_client_id
    
    dynamic_redistribution = config.dynamic_redistribution
    client_label_config = config.client_label_config
    
    use_sample_selection = config.use_sample_selection
    sample_selection_ratio = config.sample_selection_ratio
    sample_selection_method = config.sample_selection_method
    server_guidance_method = config.server_guidance_method
    
    server_gradient_methods = config.server_gradient_methods
    experiment_id = config.experiment_id
    num_rounds = config.num_rounds
    use_smo = config.use_smo
    fedprox_mu = config.fedprox_mu  # FedProx proximal coefficient

    use_cfed = getattr(config, 'use_cfed', False)
    cfed_alpha = getattr(config, 'cfed_alpha', 0.1)
    cfed_temperature = getattr(config, 'cfed_temperature', 2.0)

    # Load dataset based on configuration
    dataset_name = getattr(config, 'dataset', 'fashion_mnist').lower()
    
    if dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR10('data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10('data', train=False, download=True, transform=transform)
        input_channels = 3
        input_size = 32
        num_classes = 10
    elif dataset_name == 'pokemon':
        # Load Pokemon CSV dataset
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'pokemon.csv'))
        # Drop non-numeric and identifier columns
        feature_cols = [col for col in df.columns if col not in ['id', 'user_id', 'QoD_model', 'QoD_os-version', 'MOS']]
        X = df[feature_cols].values.astype(np.float32)
        y = (df['MOS'].values - 1).astype(np.int64)  # MOS is 1-5, convert to 0-4
        
        # Normalize features
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        
        # Split into train/test
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Convert to tensors and reshape for CNN input (add channel and spatial dims)
        n_features = X_train.shape[1]
        input_size = int(np.ceil(np.sqrt(n_features)))
        input_channels = 1
        
        # Pad features to make square
        pad_size = input_size * input_size - n_features
        X_train_padded = np.pad(X_train, ((0, 0), (0, pad_size)), mode='constant')
        X_test_padded = np.pad(X_test, ((0, 0), (0, pad_size)), mode='constant')
        
        # Reshape to (batch, channels, height, width)
        X_train_tensor = torch.FloatTensor(X_train_padded).reshape(-1, 1, input_size, input_size)
        X_test_tensor = torch.FloatTensor(X_test_padded).reshape(-1, 1, input_size, input_size)
        y_train_tensor = torch.LongTensor(y_train)
        y_test_tensor = torch.LongTensor(y_test)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        num_classes = 5  # MOS scores 1-5
    else:  # fashion_mnist
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = datasets.FashionMNIST('data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST('data', train=False, download=True, transform=transform)
        input_channels = 1
        input_size = 28
        num_classes = 10
    
    # Create initial client datasets
    client_data = create_client_data(train_dataset, num_clients=num_clients, iid=iid_setting, classes_per_client=classes_per_client, round_num=0, dynamic_redistribution=dynamic_redistribution, client_label_config=client_label_config, num_classes=num_classes)
    client_datasets = [c['dataset'] for c in client_data]
    client_loaders = [DataLoader(ds, batch_size=32, shuffle=True) for ds in client_datasets]
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Visualize label distribution
    #print(f"Data distribution: {'IID' if iid_setting else 'Non-IID'}")
    #visualize_label_distribution(client_datasets, train_dataset)
    
    # Initialize multiple servers for client-server mode
    global_model = SimpleNet(input_channels=input_channels, input_size=input_size, num_classes=num_classes).to(device)
    if client_as_server:
        servers = {}
        for server_id in server_client_ids:
            servers[server_id] = FedAvgServer(copy.deepcopy(global_model), is_client_server=True, client_ids=[server_id])
        main_server = servers[aggregation_client_id]
        print(f"Clients {server_client_ids} acting as servers")
    else:
        main_server = FedAvgServer(global_model)
        servers = {}
        print("Dedicated server")
    
    clients = [FedAvgClient(copy.deepcopy(global_model), loader, device, use_sample_selection, sample_selection_ratio, fedprox_mu, copy.deepcopy(global_model)) 
               for loader in client_loaders]
    
    # Federated training
    client_weights = [len(ds) / len(train_dataset) for ds in client_datasets]
    previous_client_accuracies = None
    client_label_history = {i: [] for i in range(num_clients)}  # ordered list of distinct label-set snapshots
    
    print(f"Starting Federated Learning with {selection_method} client selection: {use_client_selection}...")
    for round_num in range(num_rounds):
        with open(results_info_path, 'a') as f:
            f.write(f"\n--- Round {round_num + 1} ---\n")
        if round_num == 0:
            teachers = {}
        # Update client datasets if dynamic redistribution is enabled
        # if dynamic_redistribution and round_num > 0 and round_num % 5 == 0:
        #     print(f"\n--- Round {round_num + 1}: Redistributing client data ---")
        #     client_datasets = create_client_data(train_dataset, num_clients=10, iid=iid_setting, classes_per_client=classes_per_client, round_num=round_num, dynamic_redistribution=dynamic_redistribution, client_label_config=None)
        #     client_loaders = [DataLoader(ds, batch_size=32, shuffle=True) for ds in client_datasets]
        # Apply manual client label configuration if specified for this round
        if dynamic_redistribution and round_num in client_label_config:
            print(f"\n--- Round {round_num + 1}: Applying manual client configurations ---")
            client_data = create_client_data(train_dataset, num_clients=num_clients, iid=iid_setting, classes_per_client=classes_per_client, round_num=round_num, dynamic_redistribution=False, client_label_config=client_label_config, num_classes=num_classes)
            client_datasets = [c['dataset'] for c in client_data]
            client_loaders = [DataLoader(ds, batch_size=32, shuffle=True) for ds in client_datasets]
        # Client selection (each server client performs selection separately)
        if use_client_selection and round_num > 0:
            if client_as_server:
                # Each server client performs selection independently
                all_selections = []
                server_selections = {}  # Track selections per server
                
                print(f"\n--- Round {round_num + 1} Client Selection ---")
                for server_id in server_client_ids:
                    server_method = server_gradient_methods.get(server_id, 'cosine')
                    selected = servers[server_id].select_clients(
                        previous_client_accuracies, client_datasets, device,
                        selection_method, server_method, selection_ratio, input_channels, input_size, num_classes)
                    server_selections[server_id] = selected
                    all_selections.extend(selected)
                    method_name = selection_method if selection_method != 'gradient_diversity' else server_method
                    print(f"Server {server_id} ({method_name}) selected clients: {selected}")
                
                # Count selection frequency for each client
                selection_counts = {i: 0 for i in range(len(clients))}
                for client_id in all_selections:
                    selection_counts[client_id] += 1
                
                # Select clients that were chosen by at least one server
                selected_clients = [i for i, count in selection_counts.items() if count > 0]
                print(f"Combined selected clients: {sorted(selected_clients)}")
                print(f"Selection frequency: {dict(sorted(selection_counts.items()))}")
            else:
                server_selections = {}  # Initialize for non-client-server mode
                gradient_diversity_method = 'cosine'  # Default for non-client-server
                if selection_method == 'accuracy':
                    selected_clients = select_clients(previous_client_accuracies, selection_ratio)
                elif selection_method == 'gradient_diversity':
                    if gradient_diversity_method == 'cosine':
                        selected_clients = select_clients_by_gradient_diversity(
                            main_server.global_model, client_datasets, device, selection_ratio, seed=SEED, input_channels=input_channels, input_size=input_size, num_classes=num_classes)
                    elif gradient_diversity_method == 'euclidean':
                        selected_clients = select_clients_by_gradient_diversity_euclidean(
                            main_server.global_model, client_datasets, device, selection_ratio, seed=SEED, input_channels=input_channels, input_size=input_size, num_classes=num_classes)
                elif selection_method == 'random':
                    selected_clients = select_clients_randomly(len(clients), selection_ratio)
                selection_counts = {i: 1 for i in selected_clients}
        else:
            selected_clients = list(range(len(clients)))
            selection_counts = {i: 1 for i in selected_clients}
            server_selections = {}  # Initialize for first round or no selection
            gradient_diversity_method = 'cosine'  # Default when not using client selection
        
        # Parallel client training (only selected clients)
        global_state_dict = main_server.global_model.state_dict()
        
        # Prepare arguments for parallel training
        train_args = [(i, global_state_dict, client_datasets[i], str(device), 5, use_sample_selection, sample_selection_ratio, sample_selection_method, server_guidance_method, teachers, input_channels, input_size, num_classes, fedprox_mu, KD_LL_RATIO, KD_GL_RATIO, KD_TEMPERATURE, KD_WEIGHT, use_cfed, cfed_alpha, cfed_temperature) 
                     for i in selected_clients]
        
        # Execute sequential training
        results_parallel = [train_client_parallel(args) for args in train_args]
        
        # Extract client models and weights based on selection frequency
        selected_client_models = []
        selected_client_weights = []
        client_sample_counts = {}
        client_params_for_smo = [] # {"client_id": None, "loss": None, "propability_vector": None}}}
        
        for client_id, model_state, selected_samples, client_loss in results_parallel:
            selected_client_models.append(model_state)
            client_sample_counts[client_id] = selected_samples
            # Weight based on selection frequency and data size
            base_weight = client_weights[client_id]
            selection_weight = selection_counts.get(client_id, 1)
            final_weight = base_weight * selection_weight
            selected_client_weights.append(final_weight)
            if use_smo:
                client_params_for_smo.append({
                    "client_id": client_id,
                    "loss": client_loss,
                    "propability_vector": client_data[client_id]['probability_vector']['smoothed'],
                    "client_model_weights": selected_client_models[client_id]
                })
        # Normalize weights for selected clients
        total_weight = sum(selected_client_weights)
        selected_client_weights = [w / total_weight for w in selected_client_weights]

        if use_smo:
            prob_vectors = [cp['propability_vector'] for cp in client_params_for_smo]
            best_k, best_labels, best_model, scores_by_k = kmeans_optimal_k_elbow_keep_models(prob_vectors)
            with open(results_info_path, 'a') as f:
                f.write(f"KMeans optimal k (elbow method): {best_k} with inertia values: {scores_by_k}\n")
            print(f"KMeans optimal k (elbow method): {best_k} with inertia values: {scores_by_k}")
            
            df_clustering = pd.DataFrame(prob_vectors) # columns: prob vector, client loss, local leader, global leader, LL teacher, GL teacher, cluster
            df_clustering['cluster'] = best_labels
            df_clustering['loss'] = [cp['loss'] for cp in client_params_for_smo]
            df_clustering['client_id'] = [cp['client_id'] for cp in client_params_for_smo]
            client_model_weights = [cp['client_model_weights'] for cp in client_params_for_smo]

            # define the local and global leaders based on clustering
            find_local_global_leaders(df_clustering)
            find_global_leader(df_clustering)
            teachers = define_local_teacher_for_all_clients(df_clustering, client_model_weights)
            teachers = define_global_teacher_for_all_clients(df_clustering, teachers, client_model_weights)
            with open(results_info_path, 'a') as f:
                f.write(df_clustering.to_string() + '\n')
            print(df_clustering)
        
        # Server aggregation (handled by aggregation_client_id)
        if client_as_server and aggregation_client_id in server_client_ids:
            main_server.aggregate(selected_client_models, selected_client_weights)
            # Update all server models with the aggregated result
            for server_id in server_client_ids:
                servers[server_id].global_model.load_state_dict(main_server.global_model.state_dict())
            print(f"Client {aggregation_client_id} performed aggregation with selection-weighted models")
        else:
            main_server.aggregate(selected_client_models, selected_client_weights)
        
        # Evaluation
        global_accuracy = evaluate_model(main_server.global_model, test_loader, device)
        client_accuracies = evaluate_client_accuracies(main_server.global_model, client_loaders, device)
        previous_client_accuracies = client_accuracies
        
        # Get client labels for this round
        client_labels = get_client_labels(client_datasets, train_dataset)
        for cid, lbls in client_labels.items():
            current = sorted(lbls)
            if not client_label_history[cid] or client_label_history[cid][-1] != current:
                client_label_history[cid].append(current)
        
        # Evaluate accuracy and PR-AUC per client for each historical label-set snapshot independently
        client_label_metrics = {}
        for cid in range(num_clients):
            entries = []
            for lbl_set in client_label_history[cid]:
                acc, pr_auc = evaluate_per_label_metrics(main_server.global_model, test_dataset, lbl_set, device)
                entries.append({'labels': lbl_set, 'accuracy': acc, 'precision_recall_auc': pr_auc})
            client_label_metrics[cid] = entries
        
        # Store experience for replay
        if selection_method == 'experience_replay':
            experience = {
                'round': round_num + 1,
                'global_accuracy': global_accuracy,
                'client_accuracies': client_accuracies.copy(),
                'selected_clients': selected_clients.copy()
            }
        
        print(f"Round {round_num + 1}: Global Accuracy = {global_accuracy:.2f}%, Selected {len(selected_clients)}/{len(clients)} clients ({str(server_gradient_methods) if (selection_method == 'gradient_diversity' and client_as_server) else (gradient_diversity_method if selection_method == 'gradient_diversity' else selection_method)})")
        with open(results_info_path, 'a') as f:
            f.write(f"Round {round_num + 1}: Global Accuracy = {global_accuracy:.2f}%\n")

        # Save result immediately
        result = {
            'experiment_id': experiment_id,
            'dataset': dataset_name,
            'iid_setting': iid_setting,
            'classes_per_client': classes_per_client if not iid_setting else 'N/A',
            'round': round_num + 1,
            'global_accuracy': round(global_accuracy, 2),
            'client_accuracies': str([round(acc, 2) for acc in client_accuracies]),
            'selected_clients': str(selected_clients) if use_client_selection else 'All',
            'num_selected_clients': len(selected_clients),
            'selection_method': selection_method if use_client_selection else 'None',
            'gradient_diversity_method': str(server_gradient_methods) if (selection_method == 'gradient_diversity' and use_client_selection and client_as_server) else (gradient_diversity_method if (selection_method == 'gradient_diversity' and use_client_selection) else 'N/A'),
            'client_selection_ratio': str(selection_ratio),
            'server_type': 'client_server' if client_as_server else 'dedicated',
            'server_clients': str(server_client_ids) if client_as_server else 'N/A',
            'selection_counts': str(dict(sorted(selection_counts.items()))) if use_client_selection else 'N/A',
            'aggregation_client': aggregation_client_id if client_as_server else 'N/A',
            'selection_client': str(server_client_ids) if client_as_server else 'N/A',
            'server_selections': str(server_selections) if (client_as_server and use_client_selection and round_num > 0) else 'N/A',
            'client_sample_counts': str(dict(sorted(client_sample_counts.items()))),
            'use_sample_selection': use_sample_selection,
            'sample_selection_ratio': sample_selection_ratio,
            'sample_selection_method': sample_selection_method,
            'server_guidance_method': server_guidance_method if sample_selection_method == 'server_assisted' else 'N/A',
            'dynamic_redistribution': dynamic_redistribution,
            'client_labels': str(dict(sorted(client_labels.items()))),
            'client_label_metrics': str(dict(sorted(client_label_metrics.items())))
        }
        save_round_to_csv(result)
        print(f"Round {round_num + 1} results saved to CSV")
    
    print("Federated Learning completed!")

    # Compute and save final AP score per experiment
    print("Computing final Average Precision for the experiment...")
    all_targets = []
    all_probs = []
    main_server.global_model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = main_server.global_model(data)
            probs = torch.softmax(outputs, dim=1)
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    y_test_bin = label_binarize(all_targets, classes=range(num_classes))
    ap_score = average_precision_score(y_test_bin, all_probs, average='macro')
    print(f"Final Average Precision Score (Macro): {ap_score:.4f}")
    
    # Save to CSV (append mode for multiple experiments)
    ap_csv_path = os.path.join(results_dir, f'{datetime_str}_exp_{experiment_id}_ap_scores_per_experiment.csv')
    file_exists = os.path.exists(ap_csv_path)
    with open(ap_csv_path, 'a', newline='') as csvfile:
        fieldnames = ['experiment_id', 'ap_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({'experiment_id': experiment_id, 'ap_score': ap_score})

if __name__ == "__main__":
    main()
