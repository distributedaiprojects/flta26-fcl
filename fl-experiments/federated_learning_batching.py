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
from sklearn.metrics import precision_recall_curve, average_precision_score, auc
from sklearn.preprocessing import label_binarize
import sys
from collections import deque, OrderedDict
import random
from typing import List, Dict, Tuple, Any
import pandas as pd
from datetime import datetime

SEED = 200
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_info_path = f"~/results/results_info_{datetime_str}_exp{str(sys.argv[1])}.txt"


class SimpleNet(nn.Module):
    def __init__(self, input_channels=1, input_size=28, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(input_channels * input_size * input_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
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
        self.sample_history = deque(maxlen=100)
        self.probability_vector = None
        self.fedprox_mu = fedprox_mu
        self.global_model = global_model

    def generate_sample_guidance(self, dataset, global_model, selection_ratio=0.7, method='adaptive'):
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        global_model.eval()
        sample_difficulties = []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = global_model(data)
                probs = torch.softmax(output, dim=1)
                max_probs = torch.max(probs, dim=1)[0]
                for i, confidence in enumerate(max_probs):
                    sample_idx = batch_idx * data.size(0) + i
                    sample_difficulties.append((sample_idx, 1.0 - confidence.item()))

        sample_difficulties.sort(key=lambda x: x[1], reverse=True)
        num_samples = len(sample_difficulties)
        num_select = max(1, int(num_samples * selection_ratio))

        if method == 'adaptive':
            num_hard = int(num_select * 0.6)
            num_medium = num_select - num_hard
            hard_indices = [idx for idx, _ in sample_difficulties[:num_hard]]
            mid_start = num_samples // 3
            mid_end = 2 * num_samples // 3
            medium_samples = sample_difficulties[mid_start:mid_end]
            medium_indices = [idx for idx, _ in medium_samples[:num_medium]]
            selected_indices = hard_indices + medium_indices
        elif method == 'hard':
            selected_indices = [idx for idx, _ in sample_difficulties[:num_select]]
        else:
            all_indices = [idx for idx, _ in sample_difficulties]
            selected_indices = random.sample(all_indices, num_select)

        return selected_indices

    def train(self, epochs=1):
        self.model.train()
        for _ in range(epochs):
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)

                if self.fedprox_mu is not None and self.global_model is not None:
                    proximal_term = 0.0
                    for param, global_param in zip(self.model.parameters(), self.global_model.parameters()):
                        proximal_term += torch.norm(param - global_param, p=2) ** 2
                    loss += (self.fedprox_mu / 2) * proximal_term

                loss.backward()
                self.optimizer.step()

                if self.sample_selection:
                    batch_size = data.size(0)
                    individual_losses = nn.CrossEntropyLoss(reduction='none')(output, target)
                    for i, sample_loss in enumerate(individual_losses):
                        sample_idx = batch_idx * batch_size + i
                        self.sample_history.append({'idx': sample_idx, 'loss': sample_loss.item()})
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
        server_seed = SEED
        if selection_method == 'accuracy':
            return select_clients(client_accuracies, selection_ratio)
        if selection_method == 'gradient_diversity':
            if gradient_diversity_method == 'cosine':
                return select_clients_by_gradient_diversity(self.global_model, client_datasets, device, selection_ratio, server_seed, input_channels, input_size, num_classes)
            if gradient_diversity_method == 'euclidean':
                return select_clients_by_gradient_diversity_euclidean(self.global_model, client_datasets, device, selection_ratio, server_seed, input_channels, input_size, num_classes)
        if selection_method == 'random':
            return select_clients_randomly(len(client_datasets), selection_ratio)
        return list(range(len(client_datasets)))


def _labels_array(dataset):
    if hasattr(dataset, 'targets'):
        return np.array(dataset.targets)
    if isinstance(dataset, TensorDataset):
        return dataset.tensors[1].numpy()
    raise ValueError("Unsupported dataset type")


def _select_round_batch(class_indices, round_num, client_id, class_id, samples_per_class_per_client, seed, batch_strategy):
    """
    Returns a different subset of samples for the same class across rounds.

    round_robin: deterministic non-random window over a fixed permutation.
    random: deterministic random batch per round.
    """
    class_indices = np.asarray(class_indices)
    if len(class_indices) == 0:
        return []

    # Stable permutation per client/class so batches are reproducible.
    base_rng = np.random.default_rng(seed + client_id * 10_000 + class_id * 100)
    permuted = base_rng.permutation(class_indices)

    n = min(samples_per_class_per_client, len(permuted))

    if batch_strategy == 'random':
        round_rng = np.random.default_rng(seed + round_num * 100_000 + client_id * 1_000 + class_id)
        selected = round_rng.choice(permuted, size=n, replace=False)
        return selected.tolist()

    # Default: round-robin through the permuted class samples.
    start = (round_num * n) % len(permuted)
    end = start + n
    if end <= len(permuted):
        selected = permuted[start:end]
    else:
        selected = np.concatenate([permuted[start:], permuted[:end - len(permuted)]])
    return selected.tolist()


def create_client_data(
    dataset,
    num_clients=10,
    iid=True,
    classes_per_client=2,
    round_num=0,
    dynamic_redistribution=False,
    client_label_config=None,
    num_classes=10,
    samples_per_class_per_client=300,
    seed=SEED,
    batch_strategy='round_robin',
):
    """
    Creates client datasets.

    Important change for concept drift / forgetting:
    - label combinations may repeat according to client_label_config;
    - samples inside those labels change with round_num.

    This prevents measuring memorization of identical samples as if it were concept memory.
    """
    labels = _labels_array(dataset)

    if iid:
        rng = np.random.default_rng(seed + round_num)
        indices = rng.permutation(len(dataset)).tolist()
        data_per_client = len(dataset) // num_clients
        client_datasets = []
        for i in range(num_clients):
            start_idx = i * data_per_client
            end_idx = start_idx + data_per_client if i < num_clients - 1 else len(dataset)
            client_datasets.append(Subset(dataset, indices[start_idx:end_idx]))
    else:
        client_datasets = []
        for i in range(num_clients):
            if client_label_config and round_num in client_label_config and i in client_label_config[round_num]:
                client_classes = client_label_config[round_num][i]
            else:
                if dynamic_redistribution:
                    cycle_position = (round_num // 5) % 2
                    round_offset = cycle_position * classes_per_client
                else:
                    round_offset = 0
                client_classes = [(i * classes_per_client + j + round_offset) % num_classes for j in range(classes_per_client)]

            client_indices = []
            for class_id in client_classes:
                class_indices = np.where(labels == class_id)[0]
                selected = _select_round_batch(
                    class_indices=class_indices,
                    round_num=round_num,
                    client_id=i,
                    class_id=int(class_id),
                    samples_per_class_per_client=samples_per_class_per_client,
                    seed=seed,
                    batch_strategy=batch_strategy,
                )
                client_indices.extend(selected)

            client_datasets.append(Subset(dataset, client_indices))

    client_data = []
    for client_ds in client_datasets:
        class_counts = np.zeros(num_classes)
        for idx in client_ds.indices:
            class_counts[int(labels[idx])] += 1

        total_examples = len(client_ds)
        raw_probs = class_counts / total_examples if total_examples > 0 else class_counts
        smoothed_probs = (class_counts + 1) / (total_examples + num_classes)

        client_data.append({
            'dataset': client_ds,
            'probability_vector': {
                'raw': raw_probs.tolist(),
                'smoothed': smoothed_probs.tolist(),
                'class_counts': class_counts.tolist(),
                'total_examples': int(total_examples),
            },
        })

    return client_data


def get_client_labels(client_datasets, dataset):
    labels_arr = _labels_array(dataset)
    client_labels = {}
    for i, client_ds in enumerate(client_datasets):
        labels = set()
        for idx in client_ds.indices:
            labels.add(int(labels_arr[idx]))
        client_labels[i] = sorted(list(labels))
    return client_labels


def save_round_to_csv(result, filename=None):
    if filename is None:
        filename = f'{results_info_path.replace(".txt", "")}_federated_results_automated_tmp.csv'
    file_exists = os.path.exists(filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = [
            'experiment_id', 'dataset', 'iid_setting', 'classes_per_client', 'round',
            'global_accuracy', 'client_accuracies', 'selected_clients', 'num_selected_clients',
            'selection_method', 'gradient_diversity_method', 'client_selection_ratio',
            'server_type', 'server_clients', 'selection_counts', 'aggregation_client',
            'selection_client', 'server_selections', 'client_sample_counts',
            'use_sample_selection', 'sample_selection_ratio', 'sample_selection_method',
            'server_guidance_method', 'dynamic_redistribution', 'client_labels',
            'client_label_metrics'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def compute_gradient_parallel(args):
    client_id, global_state_dict, client_dataset, device_str, input_channels, input_size, num_classes = args
    device = torch.device(device_str)
    model = SimpleNet(input_channels=input_channels, input_size=input_size, num_classes=num_classes).to(device)
    model.load_state_dict(global_state_dict)

    subset_size = min(100, len(client_dataset))
    subset_dataset = Subset(client_dataset, list(range(subset_size)))
    data_loader = DataLoader(subset_dataset, batch_size=32, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    model.train()
    data, target = next(iter(data_loader))
    data, target = data.to(device), target.to(device)
    model.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()

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
    global_state_dict = global_model.state_dict()
    gradient_args = [(i, global_state_dict, client_datasets[i], str(device), input_channels, input_size, num_classes) for i in range(num_clients)]
    gradient_results = [compute_gradient_parallel(args) for args in gradient_args]

    gradients = [None] * num_clients
    for client_id, grad in gradient_results:
        gradients[client_id] = grad

    start_client = int(np.random.randint(0, num_clients)) if seed is not None else 0
    selected_clients = [start_client]

    for _ in range(num_selected - 1):
        max_diversity = -1
        candidates = []
        for i in range(num_clients):
            if i in selected_clients:
                continue
            min_distance = min(torch.norm(gradients[i] - gradients[selected], p=2).item() for selected in selected_clients)
            if min_distance > max_diversity:
                max_diversity = min_distance
                candidates = [i]
            elif abs(min_distance - max_diversity) < 1e-6:
                candidates.append(i)
        if candidates:
            selected_clients.append(int(np.random.choice(candidates)) if seed is not None else candidates[0])
    return selected_clients


def select_clients_by_gradient_diversity(global_model, client_datasets, device, selection_ratio=0.7, seed=None, input_channels=1, input_size=28, num_classes=10):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    num_clients = len(client_datasets)
    num_selected = max(1, int(num_clients * selection_ratio))
    global_state_dict = global_model.state_dict()
    gradient_args = [(i, global_state_dict, client_datasets[i], str(device), input_channels, input_size, num_classes) for i in range(num_clients)]
    gradient_results = [compute_gradient_parallel(args) for args in gradient_args]

    gradients = [None] * num_clients
    for client_id, grad in gradient_results:
        gradients[client_id] = grad

    start_client = int(np.random.randint(0, num_clients)) if seed is not None else 0
    selected_clients = [start_client]

    for _ in range(num_selected - 1):
        max_diversity = -1
        candidates = []
        for i in range(num_clients):
            if i in selected_clients:
                continue
            min_similarity = min(torch.cosine_similarity(gradients[i], gradients[selected], dim=0).item() for selected in selected_clients)
            diversity = -min_similarity
            if diversity > max_diversity:
                max_diversity = diversity
                candidates = [i]
            elif abs(diversity - max_diversity) < 1e-6:
                candidates.append(i)
        if candidates:
            selected_clients.append(int(np.random.choice(candidates)) if seed is not None else candidates[0])
    return selected_clients


def select_clients_randomly(num_clients, selection_ratio=0.7):
    num_selected = max(1, int(num_clients * selection_ratio))
    return sorted(np.random.choice(num_clients, num_selected, replace=False).tolist())


def select_clients(client_accuracies, selection_ratio=0.7):
    num_clients = len(client_accuracies)
    num_selected = max(1, int(num_clients * selection_ratio))
    client_scores = [(i, acc) for i, acc in enumerate(client_accuracies)]
    client_scores.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in client_scores[:num_selected]]


def train_client_parallel(args):
    (
        client_id, global_state_dict, client_dataset, device_str, epochs,
        sample_selection, selection_ratio, selection_method, server_guidance_method,
        teachers, input_channels, input_size, num_classes, fedprox_mu,
        kd_ll_ratio, kd_gl_ratio, kd_temperature, kd_weight
    ) = args

    device = torch.device(device_str)
    model = SimpleNet(input_channels=input_channels, input_size=input_size, num_classes=num_classes).to(device)
    model.load_state_dict(global_state_dict)

    global_model = SimpleNet(input_channels=input_channels, input_size=input_size, num_classes=num_classes).to(device)
    global_model.load_state_dict(global_state_dict)
    global_model.eval()

    has_teachers = teachers is not None and client_id in teachers
    teacher_ll = None
    teacher_gl = None

    if has_teachers:
        if 'LL' in teachers[client_id]:
            teacher_ll = SimpleNet(input_channels=input_channels, input_size=input_size, num_classes=num_classes).to(device)
            teacher_ll.load_state_dict(teachers[client_id]['LL'])
            teacher_ll.eval()
        if 'GL' in teachers[client_id]:
            teacher_gl = SimpleNet(input_channels=input_channels, input_size=input_size, num_classes=num_classes).to(device)
            teacher_gl.load_state_dict(teachers[client_id]['GL'])
            teacher_gl.eval()

    original_size = len(client_dataset)
    selected_size = original_size

    if sample_selection:
        if selection_method == 'server_assisted':
            client = FedAvgClient(model, None, device)
            selected_indices = client.generate_sample_guidance(client_dataset, model, selection_ratio, server_guidance_method)
        elif selection_method == 'hard':
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
            sample_losses.sort(key=lambda x: x[1], reverse=True)
            num_select = max(1, int(len(sample_losses) * selection_ratio))
            selected_indices = [idx for idx, _ in sample_losses[:num_select]]
        else:
            num_select = max(1, int(original_size * selection_ratio))
            selected_indices = random.sample(range(original_size), num_select)

        selected_size = len(selected_indices)
        selected_dataset = Subset(client_dataset, selected_indices)
        data_loader = DataLoader(selected_dataset, batch_size=32, shuffle=True)
    else:
        data_loader = DataLoader(client_dataset, batch_size=32, shuffle=True)

    optimizer = optim.SGD(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    temperature = kd_temperature

    model.train()
    total_loss = 0.0
    total_steps = 0

    for _ in range(epochs):
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            if has_teachers:
                distill_loss_teacher1 = torch.zeros((), device=output.device, dtype=output.dtype)
                distill_loss_teacher2 = torch.zeros((), device=output.device, dtype=output.dtype)
                num_teachers = 0
                student_logp = nn.functional.log_softmax(output / temperature, dim=1)

                if teacher_ll is not None:
                    with torch.no_grad():
                        t_prob = nn.functional.softmax(teacher_ll(data) / temperature, dim=1)
                    distill_loss_teacher1 += nn.functional.kl_div(student_logp, t_prob, reduction='batchmean')
                    num_teachers += 1

                if teacher_gl is not None:
                    with torch.no_grad():
                        t_prob = nn.functional.softmax(teacher_gl(data) / temperature, dim=1)
                    distill_loss_teacher2 += nn.functional.kl_div(student_logp, t_prob, reduction='batchmean')
                    num_teachers += 1

                if num_teachers > 0:
                    combined_distill = kd_ll_ratio * distill_loss_teacher1 + kd_gl_ratio * distill_loss_teacher2
                    loss = (1.0 - kd_weight) * loss + kd_weight * (temperature ** 2) * combined_distill

            if fedprox_mu is not None:
                proximal_term = 0.0
                for param, global_param in zip(model.parameters(), global_model.parameters()):
                    proximal_term += torch.norm(param - global_param, p=2) ** 2
                loss += (fedprox_mu / 2) * proximal_term

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_steps += 1

    client_loss = total_loss / total_steps if total_steps > 0 else 0.0
    return client_id, model.state_dict(), selected_size, client_loss


def evaluate_per_label_metrics(model, test_dataset, label_set, device):
    all_labels = _labels_array(test_dataset)
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
    return [round(evaluate_model(model, loader, device), 2) for loader in client_loaders]


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
    return 100 * correct / total if total > 0 else 0.0


def _as_array(prob_vectors: List[List[float]]) -> np.ndarray:
    X = np.asarray(prob_vectors, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got {X.shape}")
    if X.shape[0] < 1:
        raise ValueError("Need at least 1 sample.")
    return X


def kmeans_optimal_k_elbow_keep_models(prob_vectors: List[List[float]], k_min: int = 1, k_max: int = 10, n_init: int = 100, random_state: int = 42, improvement_threshold: float = 0.1) -> Tuple[int, np.ndarray, KMeans, Dict[int, float]]:
    X = _as_array(prob_vectors)
    n = X.shape[0]
    k_max = min(k_max if k_max is not None else min(10, n), n)
    k_min = max(1, k_min)
    if k_min > k_max:
        raise ValueError("Invalid k range")

    inertia_by_k = {}
    models_by_k = {}
    labels_by_k = {}

    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = model.fit_predict(X)
        inertia_by_k[k] = model.inertia_
        models_by_k[k] = model
        labels_by_k[k] = labels

    best_k = k_min
    prev_inertia = inertia_by_k[k_min]
    for k in range(k_min + 1, k_max + 1):
        current_inertia = inertia_by_k[k]
        improvement = 1.0 if prev_inertia == 0 else (prev_inertia - current_inertia) / prev_inertia
        if improvement < improvement_threshold:
            best_k = k - 1
            break
        best_k = k
        prev_inertia = current_inertia

    return best_k, labels_by_k[best_k], models_by_k[best_k], inertia_by_k


def find_local_global_leaders(df_clustering: pd.DataFrame):
    df_clustering['local_leader'] = df_clustering.index.isin(df_clustering.groupby('cluster')['loss'].idxmin()).astype(int)


def find_global_leader(df_clustering: pd.DataFrame):
    df_clustering['global_leader'] = 0
    global_leader_idx = df_clustering['loss'].idxmin()
    df_clustering.at[global_leader_idx, 'global_leader'] = 1


def weighted_average_state_dicts(state_dicts):
    avg_sd = OrderedDict()
    for key in state_dicts[0].keys():
        avg_sd[key] = sum(sd[key] for sd in state_dicts) / len(state_dicts)
    return avg_sd


def define_local_teacher_for_all_clients(df_clustering: pd.DataFrame, client_model_weights: Dict[int, Dict[str, Any]]):
    teachers = {}
    for _, row in df_clustering.iterrows():
        client_id = int(row['client_id'])
        teachers[client_id] = {}
        local_leader_id = int(df_clustering[(df_clustering['cluster'] == row['cluster']) & (df_clustering['local_leader'] == 1)]['client_id'].values[0])
        cluster_clients = df_clustering[df_clustering['cluster'] == row['cluster']]['client_id'].astype(int).values
        random_client_id = int(random.choice(cluster_clients))

        if len(cluster_clients) >= 3:
            teachers[client_id]['LL'] = weighted_average_state_dicts([
                client_model_weights[random_client_id],
                client_model_weights[local_leader_id],
                client_model_weights[client_id],
            ])

        print_string = f"Client {client_id}: Cluster {row['cluster']} local teacher = LL(Local leader {local_leader_id}, random client {random_client_id})"
        with open(results_info_path, 'a') as f:
            f.write(print_string + '\n')
        print(print_string)
    return teachers


def define_global_teacher_for_all_clients(df_clustering: pd.DataFrame, teachers: Dict[int, Dict[str, Any]], client_model_weights: Dict[int, Dict[str, Any]]):
    for _, row in df_clustering.iterrows():
        client_id = int(row['client_id'])
        global_leader_id = int(df_clustering[df_clustering['global_leader'] == 1]['client_id'].values[0])
        random_client_id = int(random.choice(df_clustering['client_id'].astype(int).values))
        teachers.setdefault(client_id, {})
        teachers[client_id]['GL'] = weighted_average_state_dicts([
            client_model_weights[random_client_id],
            client_model_weights[global_leader_id],
            client_model_weights[client_id],
        ])

        print_string = f"Client {client_id}: Cluster {row['cluster']} global teacher = GL(Global leader: {global_leader_id}, random client: {random_client_id})"
        with open(results_info_path, 'a') as f:
            f.write(print_string + '\n')
        print(print_string)
    return teachers


def load_dataset(dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train_dataset = datasets.CIFAR10('data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10('data', train=False, download=True, transform=transform)
        return train_dataset, test_dataset, 3, 32, 10

    if dataset_name == 'pokemon':
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'pokemon.csv'))
        feature_cols = [col for col in df.columns if col not in ['id', 'user_id', 'QoD_model', 'QoD_os-version', 'MOS']]
        X = df[feature_cols].values.astype(np.float32)
        y = (df['MOS'].values - 1).astype(np.int64)
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        n_features = X_train.shape[1]
        input_size = int(np.ceil(np.sqrt(n_features)))
        pad_size = input_size * input_size - n_features
        X_train_padded = np.pad(X_train, ((0, 0), (0, pad_size)), mode='constant')
        X_test_padded = np.pad(X_test, ((0, 0), (0, pad_size)), mode='constant')

        X_train_tensor = torch.FloatTensor(X_train_padded).reshape(-1, 1, input_size, input_size)
        X_test_tensor = torch.FloatTensor(X_test_padded).reshape(-1, 1, input_size, input_size)
        y_train_tensor = torch.LongTensor(y_train)
        y_test_tensor = torch.LongTensor(y_test)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        return train_dataset, test_dataset, 1, input_size, 5

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_dataset = datasets.FashionMNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST('data', train=False, download=True, transform=transform)
    return train_dataset, test_dataset, 1, 28, 10


def main():
    if len(sys.argv) < 2:
        raise ValueError("Usage: python fl_drift_batching_full.py <experiment_id>")

    spec = importlib.util.spec_from_file_location("config", os.path.join(os.path.dirname(__file__), "conf", "exp" + str(sys.argv[1]) + ".py"))
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    grid_params = getattr(config, 'GRID_SEARCH_PARAMS', None)
    if grid_params:
        KD_LL_RATIO = grid_params.get('ll_ratio', 0.9)
        KD_GL_RATIO = 1.0 - KD_LL_RATIO
        KD_TEMPERATURE = grid_params.get('temperature', 1.5)
        KD_WEIGHT = grid_params.get('kd_weight', 0.05)
    else:
        KD_LL_RATIO = 0.9
        KD_GL_RATIO = 0.1
        KD_TEMPERATURE = 1.5
        KD_WEIGHT = 0.05

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

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
    fedprox_mu = config.fedprox_mu

    samples_per_class_per_client = getattr(config, 'samples_per_class_per_client', 300)
    batch_strategy = getattr(config, 'batch_strategy', 'round_robin')

    dataset_name = getattr(config, 'dataset', 'fashion_mnist').lower()
    train_dataset, test_dataset, input_channels, input_size, num_classes = load_dataset(dataset_name)

    client_data = create_client_data(
        train_dataset,
        num_clients=num_clients,
        iid=iid_setting,
        classes_per_client=classes_per_client,
        round_num=0,
        dynamic_redistribution=dynamic_redistribution,
        client_label_config=client_label_config,
        num_classes=num_classes,
        samples_per_class_per_client=samples_per_class_per_client,
        seed=SEED,
        batch_strategy=batch_strategy,
    )
    client_datasets = [c['dataset'] for c in client_data]
    client_loaders = [DataLoader(ds, batch_size=32, shuffle=True) for ds in client_datasets]
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    global_model = SimpleNet(input_channels=input_channels, input_size=input_size, num_classes=num_classes).to(device)
    if client_as_server:
        servers = {server_id: FedAvgServer(copy.deepcopy(global_model), is_client_server=True, client_ids=[server_id]) for server_id in server_client_ids}
        main_server = servers[aggregation_client_id]
        print(f"Clients {server_client_ids} acting as servers")
    else:
        main_server = FedAvgServer(global_model)
        servers = {}
        print("Dedicated server")

    client_weights = [len(ds) / len(train_dataset) for ds in client_datasets]
    previous_client_accuracies = None
    client_label_history = {i: [] for i in range(num_clients)}
    teachers = {}

    print(f"Starting Federated Learning with {selection_method} client selection: {use_client_selection}...")
    print(f"Batching: samples_per_class_per_client={samples_per_class_per_client}, batch_strategy={batch_strategy}")

    for round_num in range(num_rounds):
        with open(results_info_path, 'a') as f:
            f.write(f"\n--- Round {round_num + 1} ---\n")

        # Recreate data every round when dynamic redistribution is enabled.
        # This is what changes the samples even when labels repeat.
        if dynamic_redistribution:
            if round_num in client_label_config:
                print(f"\n--- Round {round_num + 1}: Applying manual client configurations with fresh batches ---")
            client_data = create_client_data(
                train_dataset,
                num_clients=num_clients,
                iid=iid_setting,
                classes_per_client=classes_per_client,
                round_num=round_num,
                dynamic_redistribution=dynamic_redistribution,
                client_label_config=client_label_config,
                num_classes=num_classes,
                samples_per_class_per_client=samples_per_class_per_client,
                seed=SEED,
                batch_strategy=batch_strategy,
            )
            client_datasets = [c['dataset'] for c in client_data]
            client_loaders = [DataLoader(ds, batch_size=32, shuffle=True) for ds in client_datasets]
            client_weights = [len(ds) / sum(len(d) for d in client_datasets) for ds in client_datasets]

        if use_client_selection and round_num > 0:
            if client_as_server:
                all_selections = []
                server_selections = {}
                print(f"\n--- Round {round_num + 1} Client Selection ---")
                for server_id in server_client_ids:
                    server_method = server_gradient_methods.get(server_id, 'cosine')
                    selected = servers[server_id].select_clients(
                        previous_client_accuracies, client_datasets, device,
                        selection_method, server_method, selection_ratio,
                        input_channels, input_size, num_classes,
                    )
                    server_selections[server_id] = selected
                    all_selections.extend(selected)
                    method_name = selection_method if selection_method != 'gradient_diversity' else server_method
                    print(f"Server {server_id} ({method_name}) selected clients: {selected}")

                selection_counts = {i: 0 for i in range(num_clients)}
                for client_id in all_selections:
                    selection_counts[client_id] += 1
                selected_clients = [i for i, count in selection_counts.items() if count > 0]
                print(f"Combined selected clients: {sorted(selected_clients)}")
                print(f"Selection frequency: {dict(sorted(selection_counts.items()))}")
            else:
                server_selections = {}
                gradient_diversity_method = 'cosine'
                if selection_method == 'accuracy':
                    selected_clients = select_clients(previous_client_accuracies, selection_ratio)
                elif selection_method == 'gradient_diversity':
                    selected_clients = select_clients_by_gradient_diversity(
                        main_server.global_model, client_datasets, device, selection_ratio,
                        seed=SEED, input_channels=input_channels, input_size=input_size,
                        num_classes=num_classes,
                    )
                elif selection_method == 'random':
                    selected_clients = select_clients_randomly(len(client_datasets), selection_ratio)
                else:
                    selected_clients = list(range(num_clients))
                selection_counts = {i: 1 for i in selected_clients}
        else:
            selected_clients = list(range(num_clients))
            selection_counts = {i: 1 for i in selected_clients}
            server_selections = {}
            gradient_diversity_method = 'cosine'

        global_state_dict = main_server.global_model.state_dict()
        train_args = [
            (
                i, global_state_dict, client_datasets[i], str(device), 5,
                use_sample_selection, sample_selection_ratio, sample_selection_method,
                server_guidance_method, teachers, input_channels, input_size,
                num_classes, fedprox_mu, KD_LL_RATIO, KD_GL_RATIO,
                KD_TEMPERATURE, KD_WEIGHT,
            )
            for i in selected_clients
        ]

        results_parallel = [train_client_parallel(args) for args in train_args]

        selected_client_models = []
        selected_client_weights = []
        client_sample_counts = {}
        client_params_for_smo = []
        model_by_client_id = {}

        for client_id, model_state, selected_samples, client_loss in results_parallel:
            selected_client_models.append(model_state)
            model_by_client_id[client_id] = model_state
            client_sample_counts[client_id] = selected_samples

            base_weight = client_weights[client_id]
            selection_weight = selection_counts.get(client_id, 1)
            selected_client_weights.append(base_weight * selection_weight)

            if use_smo:
                client_params_for_smo.append({
                    'client_id': client_id,
                    'loss': client_loss,
                    'probability_vector': client_data[client_id]['probability_vector']['smoothed'],
                    'client_model_weights': model_state,
                })

        total_weight = sum(selected_client_weights)
        selected_client_weights = [w / total_weight for w in selected_client_weights]

        if use_smo and client_params_for_smo:
            prob_vectors = [cp['probability_vector'] for cp in client_params_for_smo]
            best_k, best_labels, best_model, scores_by_k = kmeans_optimal_k_elbow_keep_models(prob_vectors)
            with open(results_info_path, 'a') as f:
                f.write(f"KMeans optimal k (elbow method): {best_k} with inertia values: {scores_by_k}\n")
            print(f"KMeans optimal k (elbow method): {best_k} with inertia values: {scores_by_k}")

            df_clustering = pd.DataFrame(prob_vectors)
            df_clustering['cluster'] = best_labels
            df_clustering['loss'] = [cp['loss'] for cp in client_params_for_smo]
            df_clustering['client_id'] = [cp['client_id'] for cp in client_params_for_smo]
            client_model_weights = {cp['client_id']: cp['client_model_weights'] for cp in client_params_for_smo}

            find_local_global_leaders(df_clustering)
            find_global_leader(df_clustering)
            teachers = define_local_teacher_for_all_clients(df_clustering, client_model_weights)
            teachers = define_global_teacher_for_all_clients(df_clustering, teachers, client_model_weights)
            with open(results_info_path, 'a') as f:
                f.write(df_clustering.to_string() + '\n')
            print(df_clustering)

        if client_as_server and aggregation_client_id in server_client_ids:
            main_server.aggregate(selected_client_models, selected_client_weights)
            for server_id in server_client_ids:
                servers[server_id].global_model.load_state_dict(main_server.global_model.state_dict())
            print(f"Client {aggregation_client_id} performed aggregation with selection-weighted models")
        else:
            main_server.aggregate(selected_client_models, selected_client_weights)

        global_accuracy = evaluate_model(main_server.global_model, test_loader, device)
        client_accuracies = evaluate_client_accuracies(main_server.global_model, client_loaders, device)
        previous_client_accuracies = client_accuracies

        client_labels = get_client_labels(client_datasets, train_dataset)
        for cid, lbls in client_labels.items():
            current = sorted(lbls)
            if not client_label_history[cid] or client_label_history[cid][-1] != current:
                client_label_history[cid].append(current)

        client_label_metrics = {}
        for cid in range(num_clients):
            entries = []
            for lbl_set in client_label_history[cid]:
                acc, pr_auc = evaluate_per_label_metrics(main_server.global_model, test_dataset, lbl_set, device)
                entries.append({'labels': lbl_set, 'accuracy': acc, 'precision_recall_auc': pr_auc})
            client_label_metrics[cid] = entries

        method_display = str(server_gradient_methods) if (selection_method == 'gradient_diversity' and client_as_server) else (gradient_diversity_method if selection_method == 'gradient_diversity' else selection_method)
        print(f"Round {round_num + 1}: Global Accuracy = {global_accuracy:.2f}%, Selected {len(selected_clients)}/{num_clients} clients ({method_display})")
        with open(results_info_path, 'a') as f:
            f.write(f"Round {round_num + 1}: Global Accuracy = {global_accuracy:.2f}%\n")

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
            'client_label_metrics': str(dict(sorted(client_label_metrics.items()))),
        }
        save_round_to_csv(result)
        print(f"Round {round_num + 1} results saved to CSV")

    print("Federated Learning completed!")
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

    ap_csv_path = f'{datetime_str}_exp_{experiment_id}_ap_scores_per_experiment.csv'
    file_exists = os.path.exists(ap_csv_path)
    with open(ap_csv_path, 'a', newline='') as csvfile:
        fieldnames = ['experiment_id', 'ap_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({'experiment_id': experiment_id, 'ap_score': ap_score})


if __name__ == "__main__":
    main()
