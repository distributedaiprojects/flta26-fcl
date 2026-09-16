# Configuration for experiment 18
dataset = 'cifar10'  # Options: 'cifar10', 'fashion_mnist', 'pokemon'
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
client_label_config =  {0: {0: [3,7], 1: [0,5], 2: [9,1], 3: [4,8], 4: [2,6], 5: [7,0], 6: [5,3], 7: [8,9], 8: [1,4], 9: [6,2]}, 
                        10: {0: [3,7], 1: [0,5], 2: [9,1], 3: [4,8], 4: [2,6], 5: [7,0], 6: [5,3], 7: [8,9], 8: [1,4], 9: [6,2]}, 
                        20: {0: [3,7], 1: [0,5], 2: [9,1], 3: [4,8], 4: [2,6], 5: [7,0], 6: [5,3], 7: [8,9], 8: [1,4], 9: [6,2]}, 
                        30: {0: [3,7], 1: [0,5], 2: [9,1], 3: [4,8], 4: [2,6], 5: [7,0], 6: [5,3], 7: [8,9], 8: [1,4], 9: [6,2]}, 
                        40: {0: [3,5], 1: [0,7], 2: [1,2], 3: [4,6], 4: [8,9], 5: [7,0], 6: [5,3], 7: [2,1], 8: [6,4], 9: [9,8]}, 
                        50: {0: [3,7], 1: [0,5], 2: [9,1], 3: [4,8], 4: [2,6], 5: [7,0], 6: [5,3], 7: [8,9], 8: [1,4], 9: [6,2]}, 
                        60: {0: [3,7], 1: [0,5], 2: [9,1], 3: [4,8], 4: [2,6], 5: [7,0], 6: [5,3], 7: [8,9], 8: [1,4], 9: [6,2]}, 
                        70: {0: [3,7], 1: [0,5], 2: [9,1], 3: [4,8], 4: [2,6], 5: [7,0], 6: [5,3], 7: [8,9], 8: [1,4], 9: [6,2]}, 
                        80: {0: [3,7], 1: [0,5], 2: [9,1], 3: [4,8], 4: [2,6], 5: [7,0], 6: [5,3], 7: [8,9], 8: [1,4], 9: [6,2]}}

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

experiment_id = 239
num_rounds = 80


