"""
Grid search for optimal KD parameters
Tests different LL/GL ratios, temperatures, and KD weights
Runs with reduced rounds (30 instead of 100) for quick convergence analysis
"""
import os
import sys
import subprocess
import csv
from datetime import datetime

# Grid search parameters - reduced for quick testing
GRID_PARAMS = {
    'll_ratio': [0.0, 0.5, 1.0],              # 0=GL only, 0.5=balanced, 1=LL only
    'temperature': [1.5, 2.5, 3.5],           # KD temperature
    'kd_weight': [0.01, 0.05, 0.1],           # KD contribution weight
}

def create_grid_config(exp_id, ll_ratio, temperature, kd_weight):
    """Create experiment config for grid search"""
    # Config file must match what federated_learning.py expects: exp{arg}.py
    config_file = f'/Users/milenaangelova/git-repo/hints/fl-experiments/conf/expgrid{exp_id}.py'
    
    config_content = f"""# Auto-generated grid search config {exp_id}
# Test: LL={ll_ratio:.2f}, GL={1.0-ll_ratio:.2f}, T={temperature}, KD_weight={kd_weight}

# Read base config from exp212
import sys
import os
sys.path.insert(0, '/Users/milenaangelova/git-repo/hints/fl-experiments')

# Load all variables from exp212
exp212_path = '/Users/milenaangelova/git-repo/hints/fl-experiments/conf/exp216.py'
with open(exp212_path, 'r') as f:
    exec(f.read())

# Override settings for grid search
num_rounds = 30  # Reduced from 100 for quick testing

# Grid search parameters (will be read by federated_learning.py)
GRID_SEARCH_PARAMS = {{
    'll_ratio': {ll_ratio},
    'temperature': {temperature},
    'kd_weight': {kd_weight}
}}
"""
    
    # Write the config file
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"[DEBUG] Created config: {config_file}")
    
    # Verify file was created and contains GRID_SEARCH_PARAMS
    with open(config_file, 'r') as f:
        content = f.read()
        if 'GRID_SEARCH_PARAMS' in content:
            print(f"[DEBUG] ✓ GRID_SEARCH_PARAMS present in config file")
        else:
            print(f"[DEBUG] ✗ GRID_SEARCH_PARAMS NOT found in config file!")
    
    return config_file

def run_experiment(exp_id, ll_ratio, temperature, kd_weight):
    """Run one grid search experiment"""
    config_file = create_grid_config(exp_id, ll_ratio, temperature, kd_weight)
    
    print(f"\n{'='*70}")
    print(f"Exp {exp_id}: LL={ll_ratio:.1f}, GL={1.0-ll_ratio:.1f}, T={temperature:.1f}, KD_w={kd_weight:.2f}")
    print(f"{'='*70}")
    
    try:
        # Pass argument that matches config file: grid1 -> expgrid1.py
        result = subprocess.run(
            ['python', 'fl-experiments/federated_learning.py', f'grid{exp_id}'],
            cwd='/Users/milenaangelova/git-repo/hints',
            capture_output=True,
            text=True,
            timeout=1200  # 20 minutes per experiment
        )
        
        print(f"[DEBUG] Return code: {result.returncode}")
        
        if result.returncode == 0:
            # First, try to extract from CSV files in results directory
            final_acc = extract_accuracy_from_csv(exp_id)
            
            if final_acc is None:
                # Fallback: try to extract from stdout
                print("[DEBUG] Trying to extract from stdout...")
                lines = result.stdout.split('\n')
                for line in lines[-20:]:  # Check last 20 lines
                    if 'Final Average Precision' in line or 'Global Accuracy' in line:
                        print(f"[DEBUG] Found line: {line}")
                        parts = line.split()
                        for part in parts:
                            try:
                                val = float(part.rstrip('%').rstrip(':'))
                                if 70 <= val <= 80:  # Reasonable accuracy range
                                    final_acc = val
                                    print(f"[DEBUG] Extracted accuracy: {final_acc}")
                            except:
                                pass
            
            print(f"✓ Completed. Final Accuracy: {final_acc if final_acc else 'N/A'}")
            return final_acc
        else:
            print(f"✗ Failed with return code {result.returncode}")
            if result.stderr:
                print(f"[DEBUG] stderr: {result.stderr[-500:]}")
            if result.stdout:
                print(f"[DEBUG] stdout tail: {result.stdout[-500:]}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"✗ Timed out")
        return None
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_accuracy_from_csv(exp_id):
    """Extract final accuracy from CSV results"""
    try:
        results_dir = '/Users/milenaangelova/git-repo/hints/fl-experiments/results'
        if not os.path.exists(results_dir):
            print(f"[DEBUG] Results dir not found: {results_dir}")
            return None
        
        # Look for CSV files matching this experiment
        csv_files = [f for f in os.listdir(results_dir) if f'grid{exp_id}' in f and f.endswith('.csv')]
        print(f"[DEBUG] Found CSV files: {csv_files}")
        
        if not csv_files:
            return None
        
        # Get the most recent CSV for this experiment
        csv_file = os.path.join(results_dir, sorted(csv_files)[-1])
        print(f"[DEBUG] Reading from: {csv_file}")
        
        final_acc = None
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                # Get last round's accuracy
                final_row = rows[-1]
                final_acc = float(final_row['global_accuracy'])
                print(f"[DEBUG] Extracted accuracy from CSV: {final_acc}")
        
        return final_acc
    except Exception as e:
        print(f"[DEBUG] CSV extraction error: {e}")
        return None

def main():
    print("\n" + "="*70)
    print("KD PARAMETER GRID SEARCH (30 rounds each)")
    print("="*70)
    
    total_combos = len(GRID_PARAMS['ll_ratio']) * len(GRID_PARAMS['temperature']) * len(GRID_PARAMS['kd_weight'])
    print(f"Total combinations to test: {total_combos}")
    print(f"Estimated time: ~{total_combos * 15} minutes (15 min per experiment)")
    print(f"Results will be saved to: kd_grid_results_*.csv")
    print("="*70 + "\n")
    
    results = []
    exp_id = 0
    
    for ll_ratio in GRID_PARAMS['ll_ratio']:
        for temperature in GRID_PARAMS['temperature']:
            for kd_weight in GRID_PARAMS['kd_weight']:
                exp_id += 1
                
                final_acc = run_experiment(exp_id, ll_ratio, temperature, kd_weight)
                
                results.append({
                    'exp_id': exp_id,
                    'll_ratio': ll_ratio,
                    'gl_ratio': 1.0 - ll_ratio,
                    'temperature': temperature,
                    'kd_weight': kd_weight,
                    'ce_weight': 1.0 - kd_weight,
                    'final_accuracy': final_acc if final_acc else 'N/A'
                })
                
                # Save intermediate results after each experiment
                csv_file = f'/Users/milenaangelova/git-repo/hints/kd_grid_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}_partial.csv'
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['exp_id', 'll_ratio', 'gl_ratio', 'temperature', 'kd_weight', 'ce_weight', 'final_accuracy'])
                    writer.writeheader()
                    writer.writerows(results)
    
    # Sort by accuracy
    valid_results = [r for r in results if r['final_accuracy'] != 'N/A']
    valid_results.sort(key=lambda x: x['final_accuracy'] if isinstance(x['final_accuracy'], (int, float)) else 0, reverse=True)
    
    print("\n" + "="*70)
    if valid_results:
        print("TOP 5 RESULTS")
    else:
        print("NO VALID RESULTS FOUND")
    print("="*70)
    print(f"{'Rank':<6} {'LL%':<7} {'GL%':<7} {'T':<5} {'KD_w':<7} {'Accuracy':<10}")
    print("-"*70)
    
    for i, r in enumerate(valid_results[:5], 1):
        if r['final_accuracy'] != 'N/A':
            print(f"{i:<6} {r['ll_ratio']*100:<6.0f}% {r['gl_ratio']*100:<6.0f}% "
                  f"{r['temperature']:<4.1f} {r['kd_weight']:<7.2f} {r['final_accuracy']:<9.2f}%")
    
    # Save to CSV
    csv_file = f'/Users/milenaangelova/git-repo/hints/kd_grid_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['exp_id', 'll_ratio', 'gl_ratio', 'temperature', 'kd_weight', 'ce_weight', 'final_accuracy'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to: {csv_file}")
    
    if valid_results:
        best = valid_results[0]
        print(f"\n{'='*70}")
        print("🏆 BEST CONFIGURATION")
        print(f"{'='*70}")
        print(f"LL_ratio:   {best['ll_ratio']:.1f}")
        print(f"GL_ratio:   {best['gl_ratio']:.1f}")
        print(f"Temperature: {best['temperature']}")
        print(f"KD_weight:   {best['kd_weight']}")
        print(f"Accuracy:    {best['final_accuracy']:.2f}%")
    else:
        print(f"\n{'='*70}")
        print("⚠️  All experiments failed to extract accuracy")
        print(f"{'='*70}")
        print("Check the CSV files and output logs for debugging.")

if __name__ == '__main__':
    main()

