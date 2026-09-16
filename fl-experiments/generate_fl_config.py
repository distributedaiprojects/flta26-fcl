from itertools import combinations
import random

def generate_config(
    n_clients=20,
    classes=range(10),
    rounds=range(0, 81, 10),
    drift_clients=0,
    abrupt_rounds=None,          # None | int | iterable of ints (e.g. [30, 60])
    seed=19,
    start_pairs=None,
    return_to_start=True,
    avoid_static_pairs_in_drift=True,
):
    """
    Returns:
      client_label_config[round][client] = [a,b]
      drift_paths: dict client -> list of pairs per round

    abrupt_rounds:
      - None            -> only smooth drift
      - 30              -> abrupt at round 30
      - [30, 60]        -> abrupt at rounds 30 and 60
    """

    rng = random.Random(seed)
    rounds = list(rounds)
    classes = list(classes)

    # normalize drift_clients
    if isinstance(drift_clients, int):
        drift_clients = [drift_clients]
    else:
        drift_clients = list(drift_clients)

    # normalize abrupt_rounds
    if abrupt_rounds is None:
        abrupt_rounds_set = set()
    elif isinstance(abrupt_rounds, int):
        abrupt_rounds_set = {abrupt_rounds}
    else:
        abrupt_rounds_set = set(abrupt_rounds)

    # map abrupt rounds -> indices in rounds list
    abrupt_indices = set()
    for ar in abrupt_rounds_set:
        if ar not in rounds:
            raise ValueError(f"abrupt round {ar} not in rounds={rounds}")
        abrupt_indices.add(rounds.index(ar))

    def norm(a, b):
        return [a, b] if a < b else [b, a]

    def smooth_step(prev):
        keep = rng.choice(prev)
        candidates = [c for c in classes if c != keep]
        new_label = rng.choice(candidates)
        return norm(keep, new_label)

    def abrupt_step(prev):
        prev_set = set(prev)
        candidates = [c for c in classes if c not in prev_set]
        a, b = rng.sample(candidates, 2)
        return norm(a, b)

    if start_pairs is None:
        start_pairs = {}

    drift_paths = {}
    drift_pairs_used = set()

    # build drift paths
    for c in drift_clients:
        if c in start_pairs:
            start = norm(*start_pairs[c])
        else:
            a, b = rng.sample(classes, 2)
            start = norm(a, b)

        path = [start]
        for t in range(1, len(rounds)):
            prev = path[-1]
            if t in abrupt_indices:
                nxt = abrupt_step(prev)
            else:
                nxt = smooth_step(prev)
            path.append(nxt)

        if return_to_start:
            path[-1] = start

        drift_paths[c] = path
        for p in path:
            drift_pairs_used.add(tuple(p))

    # static clients: unique pairs
    all_pairs = [list(p) for p in combinations(classes, 2)]
    if avoid_static_pairs_in_drift:
        static_candidates = [p for p in all_pairs if tuple(p) not in drift_pairs_used]
    else:
        static_candidates = all_pairs

    static_clients = [c for c in range(n_clients) if c not in drift_clients]

    if len(static_candidates) < len(static_clients):
        raise ValueError(
            "Not enough unique label pairs for static clients. "
            "Set avoid_static_pairs_in_drift=False or reduce drift_clients."
        )

    rng.shuffle(static_candidates)
    static_assignment = {c: static_candidates[i] for i, c in enumerate(static_clients)}

    # build final config
    client_label_config = {}
    for t, r in enumerate(rounds):
        client_label_config[r] = {}
        for c in range(n_clients):
            if c in drift_paths:
                client_label_config[r][c] = drift_paths[c][t]
            else:
                client_label_config[r][c] = static_assignment[c]

    return client_label_config, drift_paths

if __name__ == "__main__":
    rounds = list(range(0, 81, 10))

    # config0, paths0 = generate_config(
    #     n_clients=10,
    #     classes=range(10),
    #     rounds=rounds,
    #     drift_clients=None,
    #     abrupt_rounds=None,
    #     seed=19,
    #     start_pairs={0:(0,1)}
    # )

    config1, paths1 = generate_config(
        n_clients=20,
        classes=range(10),
        rounds=rounds,
        drift_clients=[0,6,7,9,10,19],
        abrupt_rounds=[30, 60],
        seed=19,
        start_pairs={0:(0,1)}
    )

    # print("NO ABRUPT")
    # print(config0)

    print("\nWITH ABRUPT at 30,60")
    print(config1)
