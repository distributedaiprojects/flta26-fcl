import subprocess
import sys

def run_federated_learning(exp):
    """Run federated learning with specified server client IDs"""
    cmd = [sys.executable, "federated_learning.py", str(exp)]
    subprocess.run(cmd, check=True)

def main():
    # Different server configurations to test
    experiments = [49,50,51,52,53] #28,39,31,42,43,44,45]#30,31,32,33]#[12,7,18,21,23,28] #,12,18,21,23 
    for k in range (20):
        for exp in experiments:
            print(f"\n=== Experiment {exp} ===")
            run_federated_learning(exp)
            print(f"Experiment {exp} completed")

if __name__ == "__main__":
    main()