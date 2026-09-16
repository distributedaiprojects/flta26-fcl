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
use_smo = True

# Dynamic redistribution configuration
dynamic_redistribution = True

# Manual client label configuration
client_label_config =  {0: {0:  [0, 1], 1: [3, 6], 2: [4, 9], 3: [3, 9], 4: [5, 9], 5: [2, 4], 6: [6, 9], 7: [1, 3], 8: [1, 7], 9: [6, 8]}, 
                        10: {0: [0, 1], 1: [3, 6], 2: [4, 9], 3: [3, 9], 4: [5, 9], 5: [2, 4], 6: [6, 9], 7: [1, 3], 8: [1, 7], 9: [6, 8]}, 
                        20: {0: [0, 1], 1: [3, 6], 2: [4, 9], 3: [3, 9], 4: [5, 9], 5: [2, 4], 6: [6, 9], 7: [1, 3], 8: [1, 7], 9: [6, 8]}, 
                        30: {0: [0, 1], 1: [3, 6], 2: [4, 9], 3: [3, 9], 4: [5, 9], 5: [2, 4], 6: [6, 9], 7: [1, 3], 8: [1, 7], 9: [6, 8]}, 
                        40: {0: [2, 1], 1: [3, 6], 2: [4, 9], 3: [3, 9], 4: [5, 9], 5: [2, 4], 6: [6, 9], 7: [1, 3], 8: [1, 7], 9: [6, 8]}, 
                        50: {0: [0, 1], 1: [3, 6], 2: [4, 9], 3: [3, 9], 4: [5, 9], 5: [2, 4], 6: [6, 9], 7: [1, 3], 8: [1, 7], 9: [6, 8]}, 
                        60: {0: [0, 1], 1: [3, 6], 2: [4, 9], 3: [3, 9], 4: [5, 9], 5: [2, 4], 6: [6, 9], 7: [1, 3], 8: [1, 7], 9: [6, 8]}, 
                        70: {0: [0, 1], 1: [3, 6], 2: [4, 9], 3: [3, 9], 4: [5, 9], 5: [2, 4], 6: [6, 9], 7: [1, 3], 8: [1, 7], 9: [6, 8]}, 
                        80: {0: [0, 1], 1: [3, 6], 2: [4, 9], 3: [3, 9], 4: [5, 9], 5: [2, 4], 6: [6, 9], 7: [1, 3], 8: [1, 7], 9: [6, 8]}}

# Sample selection configuration
use_sample_selection = False
sample_selection_ratio = 1
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

experiment_id = 140
num_rounds = 80
