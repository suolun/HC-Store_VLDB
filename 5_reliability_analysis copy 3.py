
import random
import hashlib
import bisect
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import os





TOTAL_NODES = 255
INITIAL_HOTCOLD_NODES_RATIO = 0.30 


KM_PAIRS_TO_TEST = [(10, 5), (10, 10)] 
NUM_QUERIES = 2000
UNAVAILABILITY_PROBS = np.linspace(0.0, 0.3, 17) 


NUM_EXPERIMENT_RUNS = 5 


RESULT_CSV_PATH = "reliability_analysis_results.csv"  
LOAD_FROM_CSV = True  




class NetworkNode:
    def __init__(self, node_id): self.id = node_id
class HotColdNode(NetworkNode):
    def __init__(self, node_id): super().__init__(node_id)
def get_binary_group_info(n):
    if n == 0: return 0, []
    binary_str = bin(n)[2:]; group_sizes = []
    for i, bit in enumerate(binary_str):
        if bit == '1': group_sizes.append(1 << (len(binary_str) - 1 - i))
    return len(group_sizes), group_sizes

class HybridChainSimulator:
    def __init__(self, total_nodes, initial_hc_nodes_count, k, m, hot_window_blocks):
        self.total_nodes = total_nodes
        self.initial_hc_nodes_count = initial_hc_nodes_count
        self.k = k
        self.m = m
        self.hot_window_size = hot_window_blocks
        self.nodes = {i: NetworkNode(i) for i in range(total_nodes)}
        self.hc_nodes = {}; self.hc_hash_ring = []; self.ec_node_groups = []
        self.models_data = {'ec': {}, 'hc': {}}
        for key in self.models_data: self.models_data[key] = {'hot': {}, 'cold': {}, 'creation_height': {}}
        self.current_block_height = 0
        self._initialize_pools()
        self.ec_data_shard_map = {}; self.hc_data_to_shards_map = {}
        
    def _get_hash(self, key): return int.from_bytes(hashlib.sha256(str(key).encode()).digest(), 'big')

    def _initialize_pools(self):
        node_ids = list(self.nodes.keys()); random.shuffle(node_ids)
        hc_node_ids = node_ids[:self.initial_hc_nodes_count]
        for node_id in hc_node_ids:
            self.hc_nodes[node_id] = HotColdNode(node_id)
            node_hash = self._get_hash(str(node_id)); bisect.insort(self.hc_hash_ring, (node_hash, node_id))
        _, group_sizes = get_binary_group_info(self.total_nodes)
        node_pool_for_ec = list(self.nodes.values()); random.shuffle(node_pool_for_ec)
        start_idx = 0
        for size in group_sizes:
            group = node_pool_for_ec[start_idx : start_idx + size]
            self.ec_node_groups.append(tuple(g for g in group)); start_idx += size

    def process_real_block(self, block_number, transactions_in_block):
        self.current_block_height = block_number
        for tx in transactions_in_block:
            addr = tx.get('from_address');
            if pd.isna(addr): continue
            for model in self.models_data.values():
                if addr not in model['hot'] and addr not in model['cold']:
                    model['hot'][addr] = f"state_data_for_{addr}"
                    model['creation_height'][addr] = self.current_block_height
        for model_key in ['ec', 'hc']: self._check_expiry_per_block(model_key)

    def _check_expiry_per_block(self, model_key):
        model = self.models_data[model_key]
        expired_block = self.current_block_height - self.hot_window_size
        addrs_to_archive = [addr for addr, c_block in model['creation_height'].items() if addr in model['hot'] and c_block <= expired_block]
        for addr in addrs_to_archive:
            data = model['hot'].pop(addr); model['cold'][addr] = data
            num_shards = self.k + self.m
            if model_key == 'ec':
                self.ec_data_shard_map[addr] = {}
                for group in self.ec_node_groups:
                    group_shard_map = {}
                    for shard_index in range(num_shards):
                        target_node = group[self._get_hash(f"{addr}_{shard_index}") % len(group)]
                        group_shard_map[shard_index] = target_node
                    self.ec_data_shard_map[addr][group] = group_shard_map
            elif model_key == 'hc':
                shards_map = {}
                for shard_index in range(num_shards):
                    shard_key = f"{addr}_shard_{shard_index}"; shard_hash = self._get_hash(shard_key)
                    idx = bisect.bisect_left(self.hc_hash_ring, (shard_hash, -1))
                    if idx == len(self.hc_hash_ring): idx = 0
                    node_id = self.hc_hash_ring[idx][1]
                    shards_map[shard_index] = self.nodes[node_id]
                self.hc_data_to_shards_map[addr] = shards_map




def run_query_success_test(simulator, cold_data_keys, availability_prob):
    ec_success_count, hc_success_count = 0, 0
    if not cold_data_keys: return 0.0, 0.0
    test_keys = random.sample(cold_data_keys, min(len(cold_data_keys), NUM_QUERIES))
    k = simulator.k 
    for data_key in test_keys:
        group_to_test = random.choice(simulator.ec_node_groups)
        shard_map_nodes_ec = simulator.ec_data_shard_map[data_key][group_to_test]
        online_nodes_ec = {node for node in set(shard_map_nodes_ec.values()) if random.random() < availability_prob}
        available_shards_ec = sum(1 for node in shard_map_nodes_ec.values() if node in online_nodes_ec)
        if available_shards_ec >= k: ec_success_count += 1
        shard_map_nodes_hc = simulator.hc_data_to_shards_map[data_key]
        online_nodes_hc = {node for node in set(shard_map_nodes_hc.values()) if random.random() < availability_prob}
        available_shards_hc = sum(1 for node in shard_map_nodes_hc.values() if node in online_nodes_hc)
        if available_shards_hc >= k: hc_success_count += 1
    return ec_success_count / len(test_keys), hc_success_count / len(test_keys)

def plot_exp5_multikm_final(unavailability_probs, final_results):
    
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.serif'] = 'Times New Roman'
    
    fig, ax = plt.subplots(figsize=(10, 8)) 
    x_axis_percent = [p * 100 for p in unavailability_probs]
    
    
    styles = [
        {'color': 'C0', 'marker': 'o', 'linestyle': '-'},
        {'color': 'C1', 'marker': 's', 'linestyle': '-'},
        {'color': 'C2', 'marker': '^', 'linestyle': '-'},
        {'color': 'C3', 'marker': 'D', 'linestyle': '-'}
    ]
    
    for i, ((k, m), results) in enumerate(final_results.items()):
        style = styles[i % len(styles)]
        
        
        ax.plot(x_axis_percent, results['ec'], 
                label=f'EC-Chain (k={k}, m={m})',
                color=style['color'],
                marker=style['marker'],
                linestyle=style['linestyle'],
                linewidth=4,  
                markersize=8)
                
        ax.plot(x_axis_percent, results['hc'], 
                label=f'HC-Store (k={k}, m={m})',
                color=style['color'],
                marker=style['marker'],
                linestyle='--', 
                linewidth=4,  
                dashes=[4, 2],
                markersize=8)
    
    
    
    ax.set_xlabel('Node unavailability probability(%)', fontsize=40, fontfamily='Times New Roman')
    ax.set_ylabel('Query success rate', fontsize=40, fontfamily='Times New Roman')
    
    
    ax.tick_params(axis='both', which='major', labelsize=34)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')
    
    
    
    
    legend = ax.legend(fontsize=30)
    for text in legend.get_texts():
        text.set_fontname('Times New Roman')
    
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter('{:.0%}'.format))
    
    fig.tight_layout() 
    plt.savefig("reliability_analysis_final_plot.png", dpi=300)
    print("\nFinal, large-font plot saved to reliability_analysis_final_plot.png")

def save_results_to_csv(unavailability_probs, final_results, csv_path):
    """将实验结果保存到CSV文件"""
    
    data = {'unavailability_prob': unavailability_probs}
    
    
    for (k, m), results in final_results.items():
        data[f'EC_k{k}_m{m}'] = results['ec']
        data[f'HC_k{k}_m{m}'] = results['hc']
    
    
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    print(f"\n实验结果已保存到: {csv_path}")

def load_results_from_csv(csv_path, km_pairs):
    """从CSV文件加载实验结果"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV文件 {csv_path} 不存在")
    
    df = pd.read_csv(csv_path)
    unavailability_probs = df['unavailability_prob'].values
    final_results = {}
    
    
    for k, m in km_pairs:
        ec_col = f'EC_k{k}_m{m}'
        hc_col = f'HC_k{k}_m{m}'
        
        if ec_col not in df.columns or hc_col not in df.columns:
            raise ValueError(f"CSV文件中缺少 (k={k}, m={m}) 的结果列")
        
        final_results[(k, m)] = {
            'ec': df[ec_col].tolist(),
            'hc': df[hc_col].tolist()
        }
    
    print(f"\n已从 {csv_path} 加载实验结果")
    return unavailability_probs, final_results




if __name__ == "__main__":
    DATASET_PATH = 'blockchain_transactions.csv'

    try:
        start_time = time.time()
        
        
        if LOAD_FROM_CSV:
            
            unavailability_probs, final_averaged_results = load_results_from_csv(RESULT_CSV_PATH, KM_PAIRS_TO_TEST)
            
            plot_exp5_multikm_final(unavailability_probs, final_averaged_results)
        else:
            
            print(f"Loading full dataset from: {DATASET_PATH}")
            df_full = pd.read_csv(DATASET_PATH, usecols=['block_number', 'from_address'])
            df_full.dropna(subset=['from_address'], inplace=True)
            
            block_span = df_full['block_number'].max() - df_full['block_number'].min()
            HOT_WINDOW_BLOCKS = int(block_span * 0.1) if block_span > 0 else 100
            print(f"Dynamic hot window size set to: {HOT_WINDOW_BLOCKS} blocks")

            results_from_all_runs = []

            for i in range(NUM_EXPERIMENT_RUNS):
                print(f"\n{'='*30}\n--- Starting Experiment Run {i+1}/{NUM_EXPERIMENT_RUNS} ---\n{'='*30}")
                
                current_run_results = {} 

                for k, m in KM_PAIRS_TO_TEST:
                    print(f"\n>>> Running for (k={k}, m={m}) <<<")
                    random.seed(42 + i) 
                    
                    print(f"Setting up simulation world...")
                    simulator = HybridChainSimulator(TOTAL_NODES, int(TOTAL_NODES * INITIAL_HOTCOLD_NODES_RATIO), k, m, HOT_WINDOW_BLOCKS)
                    
                    df_sample = df_full.sample(frac=0.5, random_state=42+i) if len(df_full) > 50000 else df_full
                    grouped_by_block = df_sample.groupby('block_number')
                    
                    for block_num, block_txs in grouped_by_block:
                        transactions_list = block_txs.to_dict('records')
                        simulator.process_real_block(block_num, transactions_list)
                    print("World setup complete.")

                    cold_data_keys = list(simulator.models_data['hc']['cold'].keys())
                    
                    ec_results, hc_results = [], []
                    for prob in UNAVAILABILITY_PROBS:
                        availability = 1.0 - prob
                        num_trials_per_prob = 5
                        ec_rates, hc_rates = [], []
                        for _ in range(num_trials_per_prob):
                            ec_rate, hc_rate = run_query_success_test(simulator, cold_data_keys, availability)
                            ec_rates.append(ec_rate)
                            hc_rates.append(hc_rate)
                        
                        ec_results.append(np.mean(ec_rates))
                        hc_results.append(np.mean(hc_rates))
                    
                    current_run_results[(k, m)] = {'ec': ec_results, 'hc': hc_results}
                
                results_from_all_runs.append(current_run_results)

            print("\n" + "="*60)
            print("All experiment runs complete. Averaging results...")
            final_averaged_results = {}
            for k, m in KM_PAIRS_TO_TEST:
                all_ec_results = [run_result[(k,m)]['ec'] for run_result in results_from_all_runs]
                all_hc_results = [run_result[(k,m)]['hc'] for run_result in results_from_all_runs]
                
                avg_ec = np.mean(all_ec_results, axis=0)
                avg_hc = np.mean(all_hc_results, axis=0)
                
                final_averaged_results[(k, m)] = {'ec': avg_ec.tolist(), 'hc': avg_hc.tolist()}
            
            print("Averaging complete.")
            
            
            save_results_to_csv(UNAVAILABILITY_PROBS, final_averaged_results, RESULT_CSV_PATH)
            
            
            plot_exp5_multikm_final(UNAVAILABILITY_PROBS, final_averaged_results)

        end_time = time.time()
        print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        if "blockchain_transactions.csv" in str(e):
            print("Please make sure the dataset file is in the same directory or provide the correct path.")
        else:
            print("Please run the experiment first to generate the results CSV file.")
    except ValueError as e:
        print(f"\nERROR: {e}")
        print("The CSV file may be corrupted or does not match the current KM_PAIRS_TO_TEST configuration.")
    except Exception as e:
        print(f"\nAn error occurred during the program execution: {e}")