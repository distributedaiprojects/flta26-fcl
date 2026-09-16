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
    0: {0: [0, 1],1: [0, 1],2: [0, 1],3: [0, 1],4: [0, 1],5: [2, 3],6: [2, 3],7: [2, 3],8: [2, 3],9: [2, 3]},
    5:  {0: [0, 1],1: [0, 1],2: [0, 1],3: [0, 1],4: [0, 1],5: [2, 3],6: [2, 3],7: [2, 3],8: [2, 3],9: [2, 3]},
    10: {0: [0, 1],1: [0, 1],2: [0, 1],3: [0, 1],4: [0, 1],5: [2, 3],6: [2, 3],7: [2, 3],8: [2, 3],9: [2, 3]},
    15:  {0: [0, 1],1: [0, 1],2: [0, 1],3: [0, 1],4: [0, 1],5: [2, 3],6: [2, 3],7: [2, 3],8: [2, 3],9: [2, 3]},
    50:  {0: [0, 1],1: [0, 1],2: [0, 1],3: [0, 1],4: [0, 1],5: [2, 3],6: [2, 3],7: [2, 3],8: [2, 3],9: [2, 3]},
    55:  {0: [0, 1],1: [0, 1],2: [0, 1],3: [0, 1],4: [0, 1],5: [2, 3],6: [2, 3],7: [2, 3],8: [2, 3],9: [2, 3]},
    60:  {0: [0, 1],1: [0, 1],2: [0, 1],3: [0, 1],4: [0, 1],5: [2, 3],6: [2, 3],7: [2, 3],8: [2, 3],9: [2, 3]},
    65:  {0: [0, 1],1: [0, 1],2: [0, 1],3: [0, 1],4: [0, 1],5: [2, 3],6: [2, 3],7: [2, 3],8: [2, 3],9: [2, 3]},
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

experiment_id = 124
num_rounds = 80
