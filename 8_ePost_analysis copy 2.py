
import random
import hashlib
import bisect
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import os
import pickle





TOTAL_NODES = 255
INITIAL_HOTCOLD_NODES_RATIO = 0.3


NUM_TRIALS = 10000 
NUM_EXPERIMENT_RUNS = 10 


CHALLENGE_BATCH_SIZES = list(range(1, 101, 2)) 
SHARD_LOSS_PROBS_TO_TEST = [0.1, 0.2, 0.3, 0.4, 0.5] 


NODE_MALICIOUS_PROBABILITY = 0.20
K_PARAM, M_PARAM = 10, 10



RESULTS_FILE_PATH = 'batch_audit_results_1.pkl'




class MaliciousNode(object):
    def __init__(self, node_id):
        self.id = node_id
        self.is_malicious = False
        self.lost_shards_maps = {} 

    def decide_malice(self, prob):
        self.is_malicious = (random.random() < prob)

    def has_lost_shard(self, data_key, shard_id, k, m, loss_prob):
        if not self.is_malicious: return False
        
        loss_prob_key = int(loss_prob * 100)
        if loss_prob_key not in self.lost_shards_maps:
            self.lost_shards_maps[loss_prob_key] = {}

        if data_key not in self.lost_shards_maps[loss_prob_key]:
            self.lost_shards_maps[loss_prob_key][data_key] = {i for i in range(k + m) if random.random() < loss_prob}
            
        return shard_id in self.lost_shards_maps[loss_prob_key][data_key]
    
class World:
    def __init__(self, total_nodes, hc_ratio, k, m):
        self.k, self.m = k, m
        self.nodes = {i: MaliciousNode(i) for i in range(total_nodes)}
        self.hc_nodes_id = random.sample(list(self.nodes.keys()), int(total_nodes * hc_ratio))
        self.hash_ring = []
        for node_id in self.hc_nodes_id:
            node_hash = int.from_bytes(hashlib.sha256(str(node_id).encode()).digest(), 'big')
            bisect.insort(self.hash_ring, (node_hash, node_id))
        self.node_to_shards_map = {node_id: [] for node_id in self.nodes}

    def _get_hash(self, key): return int.from_bytes(hashlib.sha256(str(key).encode()).digest(), 'big')

    def distribute_data(self, num_epochs):
        for epoch_id in range(num_epochs):
            epoch_key = f"epoch_{epoch_id}"
            for shard_index in range(self.k + self.m):
                shard_key = f"{epoch_key}_shard_{shard_index}"; shard_hash = self._get_hash(shard_key)
                idx = bisect.bisect_left(self.hash_ring, (shard_hash, -1))
                node_id = self.hash_ring[idx % len(self.hash_ring)][1]
                self.node_to_shards_map[node_id].append((epoch_key, shard_index))




def run_batch_audit_trial(num_challenges, challenge_pool, lost_shards_set):
    if not challenge_pool: return False
    
    challenge_pool_set = set(challenge_pool)
    for _ in range(num_challenges):
        challenged_shard = random.choice(challenge_pool)
        if challenged_shard in lost_shards_set:
            return True
    return False

def run_single_experiment_pass(dataframe):
    
    k, m = K_PARAM, M_PARAM
    world = World(TOTAL_NODES, INITIAL_HOTCOLD_NODES_RATIO, k, m)
    for node in world.nodes.values(): node.decide_malice(NODE_MALICIOUS_PROBABILITY)
    world.distribute_data(len(dataframe))
    
    malicious_hc_nodes = [nid for nid in world.hc_nodes_id if world.nodes[nid].is_malicious]
    if not malicious_hc_nodes: return None
    target_node_id = random.choice(malicious_hc_nodes)
    
    challenge_pool = [(target_node_id, ek, si) for ek, si in world.node_to_shards_map.get(target_node_id, [])]
    if not challenge_pool: return None

    single_run_results = {}
    for loss_prob in SHARD_LOSS_PROBS_TO_TEST:
        target_node = world.nodes[target_node_id]
        lost_shards_set = {
            (target_node_id, dk, si) for dk, si in world.node_to_shards_map.get(target_node_id, []) 
            if target_node.has_lost_shard(dk, si, k, m, loss_prob)
        }
        if not lost_shards_set: continue
        
        success_rates = []
        for batch_size in CHALLENGE_BATCH_SIZES:
            successful_trials = sum(1 for _ in range(NUM_TRIALS) if run_batch_audit_trial(batch_size, challenge_pool, lost_shards_set))
            success_rates.append(successful_trials / NUM_TRIALS)
        single_run_results[f"α = {loss_prob:.0%}"] = success_rates
        
    return single_run_results




def plot_final_results(x_values, results_dict):
    
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    y_percent_dict = {label: [p * 100 for p in y_values] for label, y_values in results_dict.items()}
    
    
    styles = [
        {'marker': 'o', 'linestyle': '-'},
        {'marker': 's', 'linestyle': '--'},
        {'marker': '^', 'linestyle': ':'},
        {'marker': 'D', 'linestyle': '-.'},
        {'marker': 'v', 'linestyle': '-'}
    ]
    
    for i, (label, y_percent) in enumerate(y_percent_dict.items()):
        style = styles[i % len(styles)]
        ax.plot(x_values, y_percent, 
                label=label, 
                linewidth=4,  
                marker=style['marker'],
                linestyle=style['linestyle'],
                markersize=8,
                markevery=5) 

    
    
    ax.set_xlabel('Number of challenges per audit batch', fontsize=30)
    ax.set_ylabel('Probability of successful detection', fontsize=30)
    
    ax.tick_params(axis='both', which='major', labelsize=30)
    
    
    
    ax.legend(title='k=10,m=5',fontsize=30, title_fontsize=30)
    ax.yaxis.set_major_formatter(plt.FuncFormatter('{:.0f}%'.format))
    ax.set_ylim(0, 105)
    
    
    fig.tight_layout(rect=[0, 0, 0.95, 0.95])
    plt.savefig("batch_audit_final_plot.png", dpi=300)
    print("\nFinal plot saved as batch_audit_final_plot.png")
    plt.show()




if __name__ == "__main__":
    
    
    if os.path.exists(RESULTS_FILE_PATH):
        print(f"Found existing results file: '{RESULTS_FILE_PATH}'. Loading data and plotting.")
        try:
            with open(RESULTS_FILE_PATH, 'rb') as f:
                saved_data = pickle.load(f)
            plot_final_results(saved_data['x_axis'], saved_data['results'])
        except Exception as e:
            print(f"Error loading or plotting from file: {e}")

    else:
        
        print(f"No results file found. Starting full simulation...")
        DATASET_PATH = 'bitcoin_blocks_3months.csv'
        
        try:
            df_blocks = pd.read_csv(DATASET_PATH, usecols=['block_number'])
            df_sample = df_blocks.sample(n=1000, random_state=42)
            print(f"Dataset loaded. Using {len(df_sample)} blocks as sample.")

            start_time = time.time()
            
            
            results_from_all_runs = []
            for i in range(NUM_EXPERIMENT_RUNS):
                print(f"\n--- Starting Experiment Run {i+1}/{NUM_EXPERIMENT_RUNS} ---")
                random.seed(42 + i)
                single_run_result = run_single_experiment_pass(df_sample)
                if single_run_result:
                    results_from_all_runs.append(single_run_result)

            
            print("\nAll experiment runs complete. Averaging results...")
            final_averaged_results = {}
            if results_from_all_runs:
                
                prob_labels = results_from_all_runs[0].keys()
                for prob_label in prob_labels:
                    all_results_for_label = [run_result[prob_label] for run_result in results_from_all_runs if prob_label in run_result]
                    if all_results_for_label:
                        avg_results = np.mean(all_results_for_label, axis=0)
                        final_averaged_results[prob_label] = avg_results.tolist()

            
            data_to_save = {
                'x_axis': CHALLENGE_BATCH_SIZES,
                'results': final_averaged_results
            }
            with open(RESULTS_FILE_PATH, 'wb') as f:
                pickle.dump(data_to_save, f)
            print(f"Averaged results saved to '{RESULTS_FILE_PATH}'")

            end_time = time.time()
            
            plot_final_results(CHALLENGE_BATCH_SIZES, final_averaged_results)
            
            print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")

        except FileNotFoundError:
            print(f"错误：找不到文件 '{DATASET_PATH}'")
        except Exception as e:
            print(f"程序运行出错: {e}")