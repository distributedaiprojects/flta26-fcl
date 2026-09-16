# Configuration for experiment 18
dataset = 'pokemon'  # Options: 'cifar10', 'fashion_mnist', 'pokemon'
num_clients = 3         # pokemon clients: only 3
iid_setting = False
classes_per_client = 2
use_client_selection = False
selection_method = 'hard'
selection_ratio = 1
client_as_server = True
server_client_ids = [0,1]
aggregation_client_id = 0
use_smo = True

# Dynamic redistribution configuration
dynamic_redistribution = True

# Manual client label configuration
client_label_config = {
    0: {0: [0, 1], 1: [2, 3], 2: [1, 4]},
    5: {0: [2, 3], 1: [2, 3], 2: [1, 4]},
    10: {0: [0, 1], 1: [2, 3], 2: [1, 4]},
    15: {0: [2, 3], 1: [2, 3], 2: [1, 4]},
    50: {0: [0, 1], 1: [2, 3], 2: [1, 4]},
    55: {0: [2, 3], 1: [2, 3], 2: [1, 4]},
    60: {0: [0, 1], 1: [2, 3], 2: [1, 4]},
    65: {0: [2, 3], 1: [2, 3], 2: [1, 4]},
}

# Sample selection configuration
use_sample_selection = False
sample_selection_ratio = 1
sample_selection_method = 'server_assisted'
server_guidance_method = 'hard'

# Server gradient methods
server_gradient_methods = {
    0: 'euclidean',
    1: 'euclidean',
    2: 'euclidean'
}

experiment_id = 75
num_rounds = 80
