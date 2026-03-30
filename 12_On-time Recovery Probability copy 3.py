
import random
import hashlib
import bisect
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import os
import pickle





DEADLINE_START_MS = 50
DEADLINE_END_MS = 1000
MALICIOUS_PROBS_TO_TEST = [0.0, 0.1, 0.2, 0.3]
MALICIOUS_SHARD_LOSS_PROB = 1.0
HC_NODE_POOL_SIZE = 255
LATENCY_MEAN_MS = 50
BANDWIDTH_MEAN_KBPS = 4096
BANDWIDTH_STDEV_KBPS = 1024
K_PARAM = 10
M_PARAM = 5


NUM_EXPERIMENT_RUNS = 50  
RESULTS_FILE_PATH = 'on_time_recovery_results_1.pkl'  
CSV_RESULTS_PATH = 'on_time_recovery_results.csv'  
MAX_PLOT_DEADLINE = 400  




class Node:
    def __init__(self, latency, bandwidth):
        self.latency_ms = latency
        self.bandwidth_kbps = bandwidth
        self.is_malicious = False
    def has_lost_shard(self):
        return self.is_malicious and random.random() < MALICIOUS_SHARD_LOSS_PROB

def get_hash(key):
    return int.from_bytes(hashlib.sha256(str(key).encode()).digest(), 'big')

def calculate_single_retrieval_latency(node_pool, hash_ring, k, m, data_size_bytes, data_key):
    num_total_shards = k + m
    shard_holders = []
    for shard_index in range(num_total_shards):
        shard_key = f"{data_key}_shard_{shard_index}"
        shard_hash = get_hash(shard_key)
        idx = bisect.bisect_left(hash_ring, (shard_hash, -1))
        if idx == len(hash_ring): idx = 0
        node_id = hash_ring[idx][1]
        shard_holders.append(node_pool[node_id])

    shard_arrival_times = []
    shard_size_kb = (data_size_bytes / k) / 1024
    if shard_size_kb == 0: return 0

    for node in shard_holders:
        if node.has_lost_shard(): continue
        transfer_time_ms = (shard_size_kb / node.bandwidth_kbps) * 1000
        total_time = node.latency_ms + transfer_time_ms
        shard_arrival_times.append(total_time)
        
    if len(shard_arrival_times) < k: return float('inf')
    
    shard_arrival_times.sort()
    return shard_arrival_times[k - 1]




def run_deadline_experiment_with_real_data(dataframe):
    
    deadlines_ms = np.linspace(DEADLINE_START_MS, DEADLINE_END_MS, 100)
    results = {}

    base_node_properties = [
        (random.expovariate(1.0 / LATENCY_MEAN_MS), max(512, random.normalvariate(BANDWIDTH_MEAN_KBPS, BANDWIDTH_STDEV_KBPS)))
        for _ in range(HC_NODE_POOL_SIZE)
    ]
    
    trials_data = dataframe[['block_number', 'block_size']].to_dict('records')
    num_trials = len(trials_data)

    for prob in MALICIOUS_PROBS_TO_TEST:
        node_pool = {}
        malicious_node_indices = random.sample(range(HC_NODE_POOL_SIZE), int(HC_NODE_POOL_SIZE * prob))
        for i in range(HC_NODE_POOL_SIZE):
            latency, bandwidth = base_node_properties[i]
            node = Node(latency, bandwidth)
            if i in malicious_node_indices: node.is_malicious = True
            node_pool[i] = node

        hash_ring = []
        for node_id in node_pool:
            node_hash = get_hash(str(node_id)); bisect.insort(hash_ring, (node_hash, node_id))
        
        retrieval_latencies = [
            calculate_single_retrieval_latency(
                node_pool, hash_ring, K_PARAM, M_PARAM,
                trial['block_size'],
                trial['block_number']
            )
            for trial in trials_data
        ]
        
        success_probabilities = [
            sum(1 for lat in retrieval_latencies if lat <= deadline) / num_trials
            for deadline in deadlines_ms
        ]
            
        
        results[f"$P_f = {prob:.1f}$"] = success_probabilities
        
    return deadlines_ms, results




def save_results_to_csv(deadlines_ms, results_dict, csv_path):
    df = pd.DataFrame({'Deadline (ms)': deadlines_ms})
    for label, probs in results_dict.items():
        
        clean_label = f"P_f={label.split('=')[1].strip().replace('$', '').replace('.', '_')}"
        df[clean_label] = probs
    
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to CSV file: {csv_path}")




def load_results_from_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file {csv_path} not found!")
    
    df = pd.read_csv(csv_path)
    
    deadlines_ms = df['Deadline (ms)'].values
    
    results_dict = {}
    for col in df.columns:
        if col == 'Deadline (ms)':
            continue
        
        prob = col.split('=')[1].replace('_', '.')
        original_label = f"$P_f = {prob}$"
        results_dict[original_label] = df[col].values.tolist()
    
    return deadlines_ms, results_dict




def plot_final_results(x_deadlines, results_dict):
    
    
    mask = x_deadlines <= MAX_PLOT_DEADLINE
    filtered_deadlines = x_deadlines[mask]
    
    filtered_results = {}
    for label, probs in results_dict.items():
        
        filtered_probs = np.array(probs)[mask].tolist()
        filtered_results[label] = filtered_probs

    
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.serif'] = ['Times New Roman']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    
    styles = [
        {'marker': 'o', 'linestyle': '-'},
        {'marker': 's', 'linestyle': '--'},
        {'marker': '^', 'linestyle': ':'},
        {'marker': 'D', 'linestyle': '-.'}
    ]

    for i, (label, y_probs) in enumerate(filtered_results.items()):  
        style = styles[i % len(styles)]
        y_percent = [p * 100 for p in y_probs]
        ax.plot(filtered_deadlines, y_percent,  
                label=label,
                linewidth=4,  
                marker=style['marker'],
                linestyle=style['linestyle'],
                markersize=8,
                markevery=5) 
    
    
    
    ax.set_xlabel('Retrieval deadline (ms)', fontsize=34)
    ax.set_ylabel('On-time recovery probability', fontsize=34)
    
    
    ax.set_xlim(0, MAX_PLOT_DEADLINE)
    ax.tick_params(axis='both', which='major', labelsize=30)
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter('{:.0f}%'.format))
    
    
    
    ax.legend(fontsize=30)
    ax.set_ylim(0, 105)
    
    fig.tight_layout()
    plt.savefig("on_time_recovery_final_plot.png", dpi=300)
    print("\nFinal plot saved as on_time_recovery_final_plot.png")
    plt.show()




if __name__ == "__main__":
    
    if os.path.exists(CSV_RESULTS_PATH):
        print(f"Found existing CSV results file: '{CSV_RESULTS_PATH}'. Loading data and plotting.")
        try:
            deadlines_ms, final_averaged_results = load_results_from_csv(CSV_RESULTS_PATH)
            plot_final_results(deadlines_ms, final_averaged_results)
            exit()
        except Exception as e:
            print(f"Error loading or plotting from CSV file: {e}")
            print("Trying to load from pickle file instead...")
    
    
    if os.path.exists(RESULTS_FILE_PATH):
        print(f"Found existing results file: '{RESULTS_FILE_PATH}'. Loading data and plotting.")
        try:
            with open(RESULTS_FILE_PATH, 'rb') as f:
                saved_data = pickle.load(f)
            
            save_results_to_csv(saved_data['deadlines'], saved_data['results'], CSV_RESULTS_PATH)
            plot_final_results(saved_data['deadlines'], saved_data['results'])
        except Exception as e:
            print(f"Error loading or plotting from file: {e}")

    else:
        
        print(f"No results file found. Starting full simulation...")
        DATASET_PATH = 'bitcoin_blocks_3months.csv'
        
        try:
            print(f"Loading Bitcoin dataset from: {DATASET_PATH}")
            df_bitcoin_blocks = pd.read_csv(DATASET_PATH, usecols=['block_number', 'block_size'])
            df_sample = df_bitcoin_blocks.sample(n=2000, random_state=42) if len(df_bitcoin_blocks) > 2000 else df_bitcoin_blocks
            print(f"Dataset loaded successfully. Using {len(df_sample)} blocks as sample.")

            start_time = time.time()
            
            
            results_from_all_runs = []
            for i in range(NUM_EXPERIMENT_RUNS):
                print(f"\n--- Starting Experiment Run {i+1}/{NUM_EXPERIMENT_RUNS} ---")
                random.seed(42 + i)  
                _, single_run_result = run_deadline_experiment_with_real_data(df_sample)
                results_from_all_runs.append(single_run_result)

            
            print("\nAll experiment runs complete. Averaging results...")
            final_averaged_results = {}
            
            prob_labels = results_from_all_runs[0].keys()
            for prob_label in prob_labels:
                all_probs_for_label = [run_result[prob_label] for run_result in results_from_all_runs]
                avg_probs = np.mean(all_probs_for_label, axis=0)
                final_averaged_results[prob_label] = avg_probs.tolist()
            
            deadlines_ms = np.linspace(DEADLINE_START_MS, DEADLINE_END_MS, 100)
            
            
            data_to_save = {
                'deadlines': deadlines_ms,
                'results': final_averaged_results
            }
            with open(RESULTS_FILE_PATH, 'wb') as f:
                pickle.dump(data_to_save, f)
            print(f"Averaged results saved to '{RESULTS_FILE_PATH}'")

            
            save_results_to_csv(deadlines_ms, final_averaged_results, CSV_RESULTS_PATH)

            end_time = time.time()
            
            plot_final_results(deadlines_ms, final_averaged_results)
            
            print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")

        except FileNotFoundError:
            print(f"错误：找不到文件 '{DATASET_PATH}'")
        except Exception as e:
            print(f"程序运行出错: {e}")