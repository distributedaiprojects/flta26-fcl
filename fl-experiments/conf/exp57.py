# Configuration for experiment 18
iid_setting = False
classes_per_client = 2
use_client_selection = True
selection_method = 'gradient_diversity'
selection_ratio = 0.8
client_as_server = True
server_client_ids = [0,1]
aggregation_client_id = 0
use_smo = False

# Dynamic redistribution configuration
dynamic_redistribution = True

# Manual client label configuration
client_label_config = {
    0: {0: [0, 1],1: [1, 2],2: [2, 3],3: [3, 4],4: [4, 5],5: [5, 6],6: [6, 7],7: [7,8],8: [8,9],9: [9,0]},
    5: {0: [3, 9],1: [1, 2],2: [2, 3],3: [3, 4],4: [4, 5],5: [5, 6],6: [6, 7],7: [7,8],8: [8,9],9: [9,0]},
    10: {0: [0, 1],1: [1, 2],2: [2, 3],3: [3, 4],4: [4, 5],5: [5, 6],6: [6, 7],7: [7,8],8: [8,9],9: [9,0]},
    15: {0: [3, 9],1: [1, 2],2: [2, 3],3: [3, 4],4: [4, 5],5: [5, 6],6: [6, 7],7: [7,8],8: [8,9],9: [9,0]},
    50: {0: [0, 1],1: [1, 2],2: [2, 3],3: [3, 4],4: [4, 5],5: [5, 6],6: [6, 7],7: [7,8],8: [8,9],9: [9,0]},
    55: {0: [3, 9],1: [1, 2],2: [2, 3],3: [3, 4],4: [4, 5],5: [5, 6],6: [6, 7],7: [7,8],8: [8,9],9: [9,0]},
    60: {0: [0, 1],1: [1, 2],2: [2, 3],3: [3, 4],4: [4, 5],5: [5, 6],6: [6, 7],7: [7,8],8: [8,9],9: [9,0]},
    65: {0: [3, 9],1: [1, 2],2: [2, 3],3: [3, 4],4: [4, 5],5: [5, 6],6: [6, 7],7: [7,8],8: [8,9],9: [9,0]},
}

# Sample selection configuration
use_sample_selection = True
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

experiment_id = 57
num_rounds = 80