# Configuration for experiment 18
dataset = 'cifar10'  # Options: 'cifar10', 'fashion_mnist', 'pokemon'
num_clients = 10         # pokemon clients: only 3
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
    0: {0: [0, 1],1: [2, 3],2: [4, 5],3: [6, 7],4: [8, 9],5: [0, 1],6: [2, 3],7: [4,5],8: [6,7],9: [8,9]},
    5:  {0: [2, 3],1: [2, 3],2: [4, 5],3: [6, 7],4: [8, 9],5: [0, 1],6: [2, 3],7: [4,5],8: [6,7],9: [8,9]},
    10:  {0: [0, 1],1: [2, 3],2: [4, 5],3: [6, 7],4: [8, 9],5: [0, 1],6: [2, 3],7: [4,5],8: [6,7],9: [8,9]},
    15:  {0:  [2, 3],1: [2, 3],2: [4, 5],3: [6, 7],4: [8, 9],5: [0, 1],6: [2, 3],7: [4,5],8: [6,7],9: [8,9]},
    50:  {0: [0, 1],1: [2, 3],2: [4, 5],3: [6, 7],4: [8, 9],5: [0, 1],6: [2, 3],7: [4,5],8: [6,7],9: [8,9]},
    55:  {0:  [2, 3],1: [2, 3],2: [4, 5],3: [6, 7],4: [8, 9],5: [0, 1],6: [2, 3],7: [4,5],8: [6,7],9: [8,9]},
    60:  {0: [0, 1],1: [2, 3],2: [4, 5],3: [6, 7],4: [8, 9],5: [0, 1],6: [2, 3],7: [4,5],8: [6,7],9: [8,9]},
    65:  {0:  [2, 3],1: [2, 3],2: [4, 5],3: [6, 7],4: [8, 9],5: [0, 1],6: [2, 3],7: [4,5],8: [6,7],9: [8,9]},
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
    2: 'euclidean',
    3: 'euclidean',
    4: 'euclidean',
    5: 'euclidean',
    6: 'euclidean',
    7: 'euclidean',
    8: 'euclidean',
    9: 'euclidean'
}

experiment_id = 78
num_rounds = 80
