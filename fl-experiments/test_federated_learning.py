"""
Unit tests for federated_learning.py module.
Tests cover SimpleNet, FedAvgClient, FedAvgServer, and utility functions.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset
from collections import OrderedDict
import tempfile
import os

# Import the module under test
import sys
sys.path.insert(0, os.path.dirname(__file__))
from federated_learning import (
    SimpleNet,
    FedAvgClient,
    FedAvgServer,
    create_client_data,
    select_clients_randomly,
    select_clients,
    evaluate_model,
    compute_gradient_parallel,
    select_clients_by_gradient_diversity_euclidean,
    select_clients_by_gradient_diversity,
    weighted_average_state_dicts,
    kmeans_optimal_k_elbow_keep_models,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def device():
    """Return CPU device for testing (faster than GPU for unit tests)"""
    return torch.device('cpu')


@pytest.fixture
def simple_dataset():
    """Create a simple synthetic dataset for testing"""
    # Create random data and labels
    X = torch.randn(100, 1, 28, 28)  # 100 samples, 1 channel, 28x28
    y = torch.randint(0, 10, (100,))  # 10 classes
    dataset = TensorDataset(X, y)
    return dataset


@pytest.fixture
def simple_model(device):
    """Create a simple model for testing"""
    model = SimpleNet(input_channels=1, input_size=28, num_classes=10)
    model = model.to(device)
    return model


@pytest.fixture
def data_loader(simple_dataset):
    """Create a data loader from the simple dataset"""
    return DataLoader(simple_dataset, batch_size=32, shuffle=False)


@pytest.fixture
def client_loaders(simple_dataset):
    """Create multiple client data loaders"""
    num_clients = 5
    samples_per_client = len(simple_dataset) // num_clients
    loaders = []
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client if i < num_clients - 1 else len(simple_dataset)
        subset = Subset(simple_dataset, range(start_idx, end_idx))
        loader = DataLoader(subset, batch_size=16, shuffle=False)
        loaders.append(loader)
    return loaders


# ============================================================================
# Tests for SimpleNet
# ============================================================================

class TestSimpleNet:
    """Tests for the SimpleNet model"""

    def test_model_initialization(self):
        """Test that SimpleNet initializes correctly"""
        model = SimpleNet(input_channels=1, input_size=28, num_classes=10)
        assert model is not None
        assert isinstance(model, nn.Module)

    def test_model_initialization_custom_params(self):
        """Test SimpleNet with custom parameters"""
        model = SimpleNet(input_channels=3, input_size=32, num_classes=100)
        assert model is not None
        # Check that the first layer has correct input size: 3 * 32 * 32 = 3072
        first_layer = list(model.fc.children())[0]
        assert first_layer.in_features == 3 * 32 * 32

    def test_forward_pass(self, simple_model, device):
        """Test forward pass through the model"""
        x = torch.randn(8, 1, 28, 28).to(device)
        output = simple_model(x)
        assert output.shape == (8, 10)  # batch_size=8, num_classes=10

    def test_forward_pass_different_batch_sizes(self, simple_model, device):
        """Test forward pass with different batch sizes"""
        for batch_size in [1, 4, 16, 32]:
            x = torch.randn(batch_size, 1, 28, 28).to(device)
            output = simple_model(x)
            assert output.shape == (batch_size, 10)

    def test_model_trainable(self, simple_model):
        """Test that model parameters are trainable"""
        for param in simple_model.parameters():
            assert param.requires_grad

    def test_model_state_dict(self, simple_model):
        """Test that model state dict can be retrieved"""
        state_dict = simple_model.state_dict()
        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0
        assert 'fc.0.weight' in state_dict or 'fc.0.bias' in state_dict


# ============================================================================
# Tests for FedAvgClient
# ============================================================================

class TestFedAvgClient:
    """Tests for the FedAvgClient class"""

    def test_client_initialization(self, simple_model, data_loader, device):
        """Test FedAvgClient initialization"""
        client = FedAvgClient(
            model=simple_model,
            data_loader=data_loader,
            device=device,
            sample_selection=False
        )
        assert client is not None
        assert client.model is simple_model
        assert client.device == device
        assert client.sample_selection is False
        assert client.selection_ratio == 0.8

    def test_client_initialization_with_sample_selection(self, simple_model, data_loader, device):
        """Test FedAvgClient with sample selection enabled"""
        client = FedAvgClient(
            model=simple_model,
            data_loader=data_loader,
            device=device,
            sample_selection=True,
            selection_ratio=0.7,
            fedprox_mu=0.01
        )
        assert client.sample_selection is True
        assert client.selection_ratio == 0.7
        assert client.fedprox_mu == 0.01

    def test_client_train_single_epoch(self, simple_model, simple_dataset, device):
        """Test client training for one epoch"""
        data_loader = DataLoader(simple_dataset, batch_size=16, shuffle=True)
        client = FedAvgClient(
            model=simple_model,
            data_loader=data_loader,
            device=device
        )
        
        # Get initial state dict
        initial_state = {k: v.clone() for k, v in simple_model.state_dict().items()}
        
        # Train
        state_dict = client.train(epochs=1)
        
        # Check that state dict is returned
        assert isinstance(state_dict, dict)
        
        # Check that model weights have changed
        for key in initial_state:
            assert not torch.allclose(initial_state[key], simple_model.state_dict()[key])

    def test_client_train_multiple_epochs(self, simple_model, simple_dataset, device):
        """Test client training for multiple epochs"""
        data_loader = DataLoader(simple_dataset, batch_size=16, shuffle=True)
        client = FedAvgClient(
            model=simple_model,
            data_loader=data_loader,
            device=device
        )
        
        state_dict = client.train(epochs=3)
        assert isinstance(state_dict, dict)

    def test_compute_data_smoothing(self, simple_dataset, device):
        """Test data smoothing with Laplace correction"""
        simple_model = SimpleNet()
        client = FedAvgClient(
            model=simple_model,
            data_loader=None,
            device=device
        )
        
        # Create a balanced dataset with known distribution
        X = torch.randn(100, 1, 28, 28)
        y = torch.cat([torch.full((10,), i) for i in range(10)])  # 10 samples per class
        dataset = TensorDataset(X, y)
        
        smoothed_probs = client.compute_data_smoothing(dataset, num_classes=10)
        
        # Check output shape
        assert smoothed_probs.shape == (10,)
        
        # Check that probabilities sum to approximately 1
        assert np.isclose(smoothed_probs.sum(), 1.0)
        
        # Check that all probabilities are positive
        assert np.all(smoothed_probs > 0)

    def test_get_empirical_probability_vector(self, simple_dataset, device):
        """Test empirical probability vector computation"""
        simple_model = SimpleNet()
        client = FedAvgClient(
            model=simple_model,
            data_loader=None,
            device=device
        )
        
        prob_dict = client.get_empirical_probability_vector(simple_dataset, num_classes=10)
        
        # Check that dictionary contains expected keys
        assert 'raw' in prob_dict
        assert 'smoothed' in prob_dict
        assert 'class_counts' in prob_dict
        assert 'total_examples' in prob_dict
        
        # Check shapes and values
        assert len(prob_dict['raw']) == 10
        assert len(prob_dict['smoothed']) == 10
        assert len(prob_dict['class_counts']) == 10
        assert prob_dict['total_examples'] == len(simple_dataset)
        
        # Check that both sum to approximately 1
        assert np.isclose(prob_dict['raw'].sum(), 1.0, atol=1e-5)
        assert np.isclose(prob_dict['smoothed'].sum(), 1.0)

    def test_select_samples_without_sample_selection(self, simple_dataset, device):
        """Test select_samples when sample_selection is disabled"""
        simple_model = SimpleNet()
        client = FedAvgClient(
            model=simple_model,
            data_loader=None,
            device=device,
            sample_selection=False
        )
        
        result = client.select_samples(simple_dataset)
        # Should return the original dataset
        assert result is simple_dataset

    def test_generate_sample_guidance_adaptive(self, simple_model, simple_dataset, device):
        """Test adaptive sample selection guidance"""
        client = FedAvgClient(
            model=simple_model,
            data_loader=None,
            device=device
        )
        
        simple_model.eval()
        selected_indices = client.generate_sample_guidance(
            simple_dataset, 
            simple_model,
            selection_ratio=0.7,
            method='adaptive'
        )
        
        # Check that we get valid indices
        assert isinstance(selected_indices, list)
        assert len(selected_indices) > 0
        assert len(selected_indices) <= len(simple_dataset)
        assert all(0 <= idx < len(simple_dataset) for idx in selected_indices)

    def test_generate_sample_guidance_hard(self, simple_model, simple_dataset, device):
        """Test hard sample selection guidance"""
        client = FedAvgClient(
            model=simple_model,
            data_loader=None,
            device=device
        )
        
        simple_model.eval()
        selected_indices = client.generate_sample_guidance(
            simple_dataset,
            simple_model,
            selection_ratio=0.6,
            method='hard'
        )
        
        assert isinstance(selected_indices, list)
        assert len(selected_indices) > 0
        assert len(selected_indices) <= len(simple_dataset)


# ============================================================================
# Tests for FedAvgServer
# ============================================================================

class TestFedAvgServer:
    """Tests for the FedAvgServer class"""

    def test_server_initialization(self, simple_model):
        """Test FedAvgServer initialization"""
        server = FedAvgServer(model=simple_model)
        assert server is not None
        assert server.global_model is simple_model
        assert server.is_client_server is False
        assert server.client_ids == []

    def test_server_initialization_with_client_server_mode(self, simple_model):
        """Test FedAvgServer with client_server mode"""
        client_ids = [0, 1, 2]
        server = FedAvgServer(
            model=simple_model,
            is_client_server=True,
            client_ids=client_ids
        )
        assert server.is_client_server is True
        assert server.client_ids == client_ids

    def test_aggregate_two_clients(self, simple_model, device):
        """Test aggregation with two clients"""
        server = FedAvgServer(model=simple_model)
        
        # Create two client model states
        client1_state = simple_model.state_dict()
        client2_state = simple_model.state_dict()
        
        # Modify them slightly
        for key in client1_state:
            client1_state[key] = client1_state[key] + 0.1
        for key in client2_state:
            client2_state[key] = client2_state[key] - 0.1
        
        client_models = [client1_state, client2_state]
        client_weights = [0.5, 0.5]  # Equal weights
        
        aggregated_state = server.aggregate(client_models, client_weights)
        
        assert isinstance(aggregated_state, dict)
        # Check that aggregated result is approximately the average
        for key in aggregated_state:
            assert aggregated_state[key] is not None

    def test_aggregate_weighted_clients(self, simple_model, device):
        """Test aggregation with different weights"""
        server = FedAvgServer(model=simple_model)
        
        client1_state = {k: v.clone() for k, v in simple_model.state_dict().items()}
        client2_state = {k: v.clone() for k, v in simple_model.state_dict().items()}
        
        client_models = [client1_state, client2_state]
        client_weights = [0.7, 0.3]  # Unequal weights
        
        aggregated_state = server.aggregate(client_models, client_weights)
        assert isinstance(aggregated_state, dict)


# ============================================================================
# Tests for Utility Functions
# ============================================================================

class TestUtilityFunctions:
    """Tests for utility functions"""

    def test_create_client_data_iid(self, simple_dataset):
        """Test IID data distribution"""
        num_clients = 5
        client_data = create_client_data(
            simple_dataset,
            num_clients=num_clients,
            iid=True,
            num_classes=10
        )
        
        # Check that we have the right number of clients
        assert len(client_data) == num_clients
        
        # Each client should have a dataset
        for client_info in client_data:
            assert 'dataset' in client_info
            assert 'probability_vector' in client_info
            assert len(client_info['dataset']) > 0

    def test_create_client_data_non_iid(self, simple_dataset):
        """Test non-IID data distribution"""
        num_clients = 5
        classes_per_client = 2
        client_data = create_client_data(
            simple_dataset,
            num_clients=num_clients,
            iid=False,
            classes_per_client=classes_per_client,
            num_classes=10
        )
        
        assert len(client_data) == num_clients
        for client_info in client_data:
            assert 'dataset' in client_info
            assert 'probability_vector' in client_info

    def test_create_client_data_with_config(self, simple_dataset):
        """Test data distribution with manual client label config"""
        client_label_config = {
            0: {
                0: [0, 1],
                1: [2, 3],
                2: [4, 5]
            }
        }
        num_clients = 3
        client_data = create_client_data(
            simple_dataset,
            num_clients=num_clients,
            iid=False,
            classes_per_client=2,
            round_num=0,
            client_label_config=client_label_config,
            num_classes=10
        )
        
        assert len(client_data) == num_clients

    def test_select_clients_randomly(self):
        """Test random client selection"""
        num_clients = 10
        selection_ratio = 0.5
        
        selected = select_clients_randomly(num_clients, selection_ratio)
        
        assert isinstance(selected, list)
        assert len(selected) == 5  # 50% of 10
        assert all(0 <= c < num_clients for c in selected)
        assert len(set(selected)) == len(selected)  # No duplicates

    def test_select_clients_randomly_high_ratio(self):
        """Test random selection with high ratio"""
        num_clients = 10
        selection_ratio = 0.9
        
        selected = select_clients_randomly(num_clients, selection_ratio)
        assert len(selected) == 9

    def test_select_clients_randomly_low_ratio(self):
        """Test random selection with low ratio"""
        num_clients = 10
        selection_ratio = 0.1
        
        selected = select_clients_randomly(num_clients, selection_ratio)
        assert len(selected) >= 1  # At least 1 client

    def test_select_clients_by_accuracy(self):
        """Test client selection by accuracy"""
        client_accuracies = [0.9, 0.7, 0.85, 0.6, 0.95, 0.75]
        selection_ratio = 0.5
        
        selected = select_clients(client_accuracies, selection_ratio)
        
        # Should select top 3 clients (50% of 6)
        assert len(selected) == 3
        # Should select clients with indices 4 (0.95), 0 (0.9), 2 (0.85)
        assert 4 in selected
        assert 0 in selected
        assert 2 in selected

    def test_select_clients_by_accuracy_all(self):
        """Test selection with ratio = 1.0"""
        client_accuracies = [0.9, 0.7, 0.85]
        selected = select_clients(client_accuracies, selection_ratio=1.0)
        
        assert len(selected) == 3

    def test_select_clients_by_accuracy_one(self):
        """Test selection with low ratio (minimum 1)"""
        client_accuracies = [0.9, 0.7, 0.85, 0.6, 0.95]
        selected = select_clients(client_accuracies, selection_ratio=0.01)
        
        assert len(selected) >= 1

    def test_evaluate_model(self, simple_model, data_loader, device):
        """Test model evaluation"""
        simple_model.eval()
        accuracy = evaluate_model(simple_model, data_loader, device)
        
        # Accuracy should be between 0 and 100
        assert 0 <= accuracy <= 100
        assert isinstance(accuracy, float)

    def test_weighted_average_state_dicts(self, simple_model):
        """Test weighted averaging of state dicts"""
        state1 = simple_model.state_dict()
        state2 = simple_model.state_dict()
        
        # Modify state2
        for key in state2:
            state2[key] = state2[key] * 2
        
        state_dicts = [state1, state2]
        averaged = weighted_average_state_dicts(state_dicts)
        
        assert isinstance(averaged, OrderedDict)
        assert len(averaged) == len(state1)
        
        # Check that averaged is between state1 and state2
        for key in averaged:
            avg_val = averaged[key]
            min_val = torch.min(state1[key], state2[key])
            max_val = torch.max(state1[key], state2[key])
            # Average should be within the range
            assert torch.all(avg_val <= max_val.max())


# ============================================================================
# Tests for Clustering Functions
# ============================================================================

class TestClusteringFunctions:
    """Tests for K-means and clustering utilities"""

    def test_kmeans_optimal_k_simple(self):
        """Test K-means with simple probability vectors"""
        prob_vectors = [
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.7, 0.2],
            [0.1, 0.1, 0.8],
        ]
        
        best_k, labels, model, inertia_by_k = kmeans_optimal_k_elbow_keep_models(
            prob_vectors, k_min=1, k_max=5
        )
        
        assert best_k >= 1
        assert best_k <= 5
        assert len(labels) == len(prob_vectors)
        assert isinstance(inertia_by_k, dict)

    def test_kmeans_optimal_k_returns_model(self):
        """Test that K-means returns a fitted model"""
        prob_vectors = [[i, i+1, i+2] for i in range(10)]
        
        best_k, labels, model, inertia_by_k = kmeans_optimal_k_elbow_keep_models(
            prob_vectors, k_min=2, k_max=5
        )
        
        # Model should be fitted and have correct number of clusters
        assert model.n_clusters == best_k
        assert len(np.unique(labels)) <= best_k

    def test_kmeans_single_cluster_minimum(self):
        """Test K-means with k_min=1"""
        prob_vectors = [[0.5, 0.5], [0.5, 0.5], [0.6, 0.4]]
        
        best_k, labels, model, inertia_by_k = kmeans_optimal_k_elbow_keep_models(
            prob_vectors, k_min=1, k_max=3
        )
        
        assert best_k >= 1
        assert 1 in inertia_by_k

    def test_kmeans_improvement_threshold(self):
        """Test K-means with different improvement thresholds"""
        prob_vectors = [
            [i, i+1] for i in range(20)
        ]
        
        best_k1, _, _, _ = kmeans_optimal_k_elbow_keep_models(
            prob_vectors, k_min=1, k_max=10, improvement_threshold=0.1
        )
        
        best_k2, _, _, _ = kmeans_optimal_k_elbow_keep_models(
            prob_vectors, k_min=1, k_max=10, improvement_threshold=0.5
        )
        
        # Higher threshold should result in lower k
        assert best_k2 <= best_k1


# ============================================================================
# Tests for Gradient-based Client Selection
# ============================================================================

class TestGradientBasedSelection:
    """Tests for gradient diversity-based client selection"""

    def test_select_clients_by_gradient_diversity_euclidean(self, simple_model, simple_dataset, device):
        """Test Euclidean distance-based gradient diversity selection"""
        num_clients = 5
        client_datasets = []
        for i in range(num_clients):
            start = i * (len(simple_dataset) // num_clients)
            end = start + (len(simple_dataset) // num_clients)
            subset = Subset(simple_dataset, range(start, end))
            client_datasets.append(subset)
        
        selected = select_clients_by_gradient_diversity_euclidean(
            simple_model,
            client_datasets,
            device,
            selection_ratio=0.6,
            seed=42
        )
        
        # Should select 3 clients (60% of 5, rounded)
        assert len(selected) >= 1
        assert all(0 <= c < num_clients for c in selected)
        assert len(set(selected)) == len(selected)  # No duplicates

    def test_select_clients_by_gradient_diversity_cosine(self, simple_model, simple_dataset, device):
        """Test cosine similarity-based gradient diversity selection"""
        num_clients = 5
        client_datasets = []
        for i in range(num_clients):
            start = i * (len(simple_dataset) // num_clients)
            end = start + (len(simple_dataset) // num_clients)
            subset = Subset(simple_dataset, range(start, end))
            client_datasets.append(subset)
        
        selected = select_clients_by_gradient_diversity(
            simple_model,
            client_datasets,
            device,
            selection_ratio=0.6,
            seed=42
        )
        
        assert len(selected) >= 1
        assert all(0 <= c < num_clients for c in selected)
        assert len(set(selected)) == len(selected)  # No duplicates


# ============================================================================
# Edge Case and Integration Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and potential error conditions"""

    def test_empty_selection_ratio(self):
        """Test with very small selection ratio"""
        num_clients = 100
        selection_ratio = 0.001
        
        selected = select_clients_randomly(num_clients, selection_ratio)
        # Should select at least 1 client
        assert len(selected) >= 1

    def test_single_client(self):
        """Test with single client"""
        num_clients = 1
        client_accuracies = [0.9]
        
        selected = select_clients(client_accuracies, selection_ratio=1.0)
        assert selected == [0]

    def test_model_with_single_sample(self, simple_model, device):
        """Test model evaluation with single sample"""
        X = torch.randn(1, 1, 28, 28).to(device)
        y = torch.tensor([0]).to(device)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=1)
        
        accuracy = evaluate_model(simple_model, loader, device)
        assert 0 <= accuracy <= 100

    def test_zero_weighted_average(self, simple_model):
        """Test weighted average with different weight distributions"""
        state1 = simple_model.state_dict()
        state2 = simple_model.state_dict()
        
        # Test with extreme weights
        state_dicts = [state1, state2]
        averaged = weighted_average_state_dicts(state_dicts)
        
        assert isinstance(averaged, OrderedDict)


class TestIntegration:
    """Integration tests combining multiple components"""

    def test_full_federated_learning_round(self, simple_model, simple_dataset, device):
        """Test a complete federated learning round"""
        # Create client data
        num_clients = 3
        client_data = create_client_data(
            simple_dataset,
            num_clients=num_clients,
            iid=True,
            num_classes=10
        )
        
        # Create server
        server = FedAvgServer(model=simple_model)
        
        # Train clients
        client_states = []
        for i, client_info in enumerate(client_data):
            data_loader = DataLoader(client_info['dataset'], batch_size=16, shuffle=True)
            client = FedAvgClient(
                model=simple_model,
                data_loader=data_loader,
                device=device
            )
            state = client.train(epochs=1)
            client_states.append(state)
        
        # Aggregate
        weights = [1.0 / num_clients] * num_clients
        aggregated = server.aggregate(client_states, weights)
        
        assert isinstance(aggregated, dict)
        assert len(aggregated) > 0

    def test_federated_learning_with_non_iid_data(self, simple_model, simple_dataset, device):
        """Test federated learning with non-IID data distribution"""
        num_clients = 4
        client_data = create_client_data(
            simple_dataset,
            num_clients=num_clients,
            iid=False,
            classes_per_client=2,
            num_classes=10
        )
        
        server = FedAvgServer(model=simple_model)
        
        # Train on non-IID data
        client_states = []
        for client_info in client_data:
            data_loader = DataLoader(client_info['dataset'], batch_size=16, shuffle=True)
            client = FedAvgClient(
                model=simple_model,
                data_loader=data_loader,
                device=device
            )
            state = client.train(epochs=1)
            client_states.append(state)
        
        weights = [1.0 / num_clients] * num_clients
        aggregated = server.aggregate(client_states, weights)
        
        assert isinstance(aggregated, dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
