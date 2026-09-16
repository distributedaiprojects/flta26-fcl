# Configuration for experiment 18
dataset = 'fashion_mnist'  # Options: 'cifar10', 'fashion_mnist', 'pokemon'
num_clients = 10         # pokemon clients: only 3
iid_setting = False
classes_per_client = 2
use_client_selection = False
selection_method = 'hard'
selection_ratio = 1
client_as_server = True
server_client_ids = [0]
aggregation_client_id = 0
use_smo = False
fedprox_mu = None

# Dynamic redistribution configuration
dynamic_redistribution = True

# Manual client label configuration
client_label_config =  {0: {0: [0,1], 1: [0,1], 2: [0,1], 3: [0,1], 4: [0,1], 5: [1,0], 6: [1,0], 7: [1,0], 8: [1,0], 9: [1,0]},
                        50: {0: [2,3], 1: [2,3], 2: [2,3], 3: [2,3], 4: [2,3], 5: [4,5], 6: [4,5], 7: [4,5], 8: [4,5], 9: [4,5]}, 
                        100: {0: [4,5], 1: [4,5], 2: [4,5], 3: [4,5], 4: [4,5], 5: [6,7], 6: [6,7], 7: [6,7], 8: [6,7], 9: [6,7]},
                        150: {0: [6,7], 1: [6,7], 2: [6,7], 3: [6,7], 4: [6,7], 5: [8,9], 6: [8,9], 7: [8,9], 8: [8,9], 9: [8,9]},
                        200: {0: [8,9], 1: [8,9], 2: [8,9], 3: [8,9], 4: [8,9], 5: [2,3], 6: [2,3], 7: [2,3], 8: [2,3], 9: [2,3]},
                        250: {0: [0,1], 1: [0,1], 2: [0,1], 3: [0,1], 4: [0,1], 5: [1,0], 6: [1,0], 7: [1,0], 8: [1,0], 9: [1,0]}
                        }
# Sample selection configuration
use_sample_selection = True
sample_selection_ratio = 0.7
sample_selection_method = 'hard'
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

experiment_id = 291
num_rounds = 250


