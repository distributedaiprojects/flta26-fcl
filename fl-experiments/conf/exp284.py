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
client_label_config =  {0: {0: [1,4], 1: [9,2], 2: [8,1], 3: [4,6], 4: [2,9], 5: [6,8], 6: [1,2], 7: [4,9], 8: [6,1], 9: [8,2]}, 
                        10: {0: [1,4], 1: [9,2], 2: [8,1], 3: [4,6], 4: [2,9], 5: [6,8], 6: [1,2], 7: [4,9], 8: [6,1], 9: [8,2]}, 
                        20: {0: [1,4], 1: [9,2], 2: [8,1], 3: [4,6], 4: [2,9], 5: [6,8], 6: [1,2], 7: [4,9], 8: [6,1], 9: [8,2]}, 
                        30: {0: [1,4], 1: [9,2], 2: [8,1], 3: [4,6], 4: [2,9], 5: [6,8], 6: [1,2], 7: [4,9], 8: [6,1], 9: [8,2]}, 
                        40: {0: [1,4], 1: [9,2], 2: [8,1], 3: [4,6], 4: [2,9], 5: [6,8], 6: [1,2], 7: [4,9], 8: [6,1], 9: [8,2]}, 
                        50: {0: [3,5], 1: [0,7], 2: [8,1], 3: [4,6], 4: [2,9], 5: [6,8], 6: [1,2], 7: [4,9], 8: [6,1], 9: [8,2]}, 
                        60: {0: [3,5], 1: [0,7], 2: [8,1], 3: [4,6], 4: [2,9], 5: [6,8], 6: [1,2], 7: [4,9], 8: [6,1], 9: [8,2]}, 
                        70: {0: [3,5], 1: [0,7], 2: [8,1], 3: [4,6], 4: [2,9], 5: [6,8], 6: [1,2], 7: [4,9], 8: [6,1], 9: [8,2]}, 
                        80: {0: [3,5], 1: [0,7], 2: [8,1], 3: [4,6], 4: [2,9], 5: [6,8], 6: [1,2], 7: [4,9], 8: [6,1], 9: [8,2]},
                        90: {0: [1,4], 1: [9,2], 2: [8,1], 3: [4,6], 4: [2,9], 5: [6,8], 6: [1,2], 7: [4,9], 8: [6,1], 9: [8,2]},
                        100: {0: [1,4], 1: [9,2], 2: [8,1], 3: [4,6], 4: [2,9], 5: [6,8], 6: [1,2], 7: [4,9], 8: [6,1], 9: [8,2]},
                        110: {0: [1,4], 1: [9,2], 2: [8,1], 3: [4,6], 4: [2,9], 5: [6,8], 6: [1,2], 7: [4,9], 8: [6,1], 9: [8,2]},
                        120: {0: [1,4], 1: [9,2], 2: [8,1], 3: [4,6], 4: [2,9], 5: [6,8], 6: [1,2], 7: [4,9], 8: [6,1], 9: [8,2]}}

# Sample selection configuration
use_sample_selection = False
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

experiment_id = 284
num_rounds = 120


