
import random
import hashlib
import bisect
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle





TOTAL_NODES = 255
INITIAL_HOTCOLD_NODES_RATIO = 0.30


NUM_QUERIES_EXP6 = 2000
SHARD_LOSS_PROBS_TO_TEST = np.linspace(0.0, 1.0, 21)
MALICIOUS_NODE_PROPORTIONS = [0.20]
KM_PAIRS_TO_TEST = [(10, 5), (10, 10)]
NUM_RUNS = 30


RESULTS_FILE_PATH = 'security_analysis_results.pkl'




class NetworkNode:
    def __init__(self, node_id):
        self.id = node_id
        self.is_malicious = False
        self.lost_shards = {}

    def has_lost_shard(self, data_key, shard_id, k, m, loss_prob):
        if not self.is_malicious:
            return False
        if data_key not in self.lost_shards:
            lost_shard_set = set()
            for i in range(k + m):
                if random.random() < loss_prob:
                    lost_shard_set.add(i)
            self.lost_shards[data_key] = lost_shard_set
        return shard_id in self.lost_shards.get(data_key, set())

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
        self.nodes = [NetworkNode(i) for i in range(total_nodes)]
        self.hc_nodes = {}
        self.hc_hash_ring = []
        self.ec_node_groups = []
        self.models_data = {'ec': {}, 'hc': {}}
        for key in self.models_data:
            self.models_data[key] = {'hot': {}, 'cold': {}, 'creation_height': {}}
        self.current_block_height = 0
        self._initialize_pools()
        self.ec_data_shard_map = {}
        self.hc_data_to_shards_map = {}

    def _get_hash(self, key): return int.from_bytes(hashlib.sha256(str(key).encode()).digest(), 'big')

    def _initialize_pools(self):
        nodes_for_hc = random.sample(self.nodes, self.initial_hc_nodes_count)
        for node in nodes_for_hc:
            self.hc_nodes[node.id] = node
            node_hash = self._get_hash(f"node-{node.id}")
            bisect.insort(self.hc_hash_ring, (node_hash, node.id))

        _, group_sizes = get_binary_group_info(self.total_nodes)
        node_pool_for_ec = list(self.nodes); random.shuffle(node_pool_for_ec)
        start_idx = 0
        for size in group_sizes:
            group = node_pool_for_ec[start_idx : start_idx + size]
            self.ec_node_groups.append(tuple(g for g in group))
            start_idx += size

    def process_real_block(self, block_number, transactions_in_block):
        self.current_block_height = block_number
        for tx in transactions_in_block:
            addr = tx.get('from_address')
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
            model['hot'].pop(addr)
            model['cold'][addr] = f"state_data_for_{addr}"
            num_shards = self.k + self.m
            if model_key == 'ec':
                self.ec_data_shard_map[addr] = {}
                for group in self.ec_node_groups:
                    group_shard_map = {i: group[self._get_hash(f"{addr}_{i}") % len(group)] for i in range(num_shards)}
                    self.ec_data_shard_map[addr][group] = group_shard_map
            elif model_key == 'hc':
                shards_map = {}
                for i in range(num_shards):
                    shard_hash = self._get_hash(f"{addr}_shard_{i}")
                    idx = bisect.bisect_left(self.hc_hash_ring, (shard_hash, -1))
                    node_id = self.hc_hash_ring[idx % len(self.hc_hash_ring)][1]
                    shards_map[i] = self.nodes[node_id]
                self.hc_data_to_shards_map[addr] = shards_map

    def _simulate_retrieval_outcome(self, shard_map_nodes, data_key, loss_prob):
        lost_shards_count = sum(1 for sid, node in shard_map_nodes.items() if node.has_lost_shard(data_key, sid, self.k, self.m, loss_prob))
        return (self.k + self.m - lost_shards_count) >= self.k




def setup_world_with_real_data(dataframe, k, m, hot_window_blocks):
    simulator = HybridChainSimulator(TOTAL_NODES, int(TOTAL_NODES * INITIAL_HOTCOLD_NODES_RATIO), k, m, hot_window_blocks)
    grouped_by_block = dataframe.groupby('block_number')
    for block_num, block_txs in grouped_by_block:
        simulator.process_real_block(block_num, block_txs.to_dict('records'))
    return simulator

def run_query_success_test(simulator, cold_data_keys, loss_prob):
    ec_success_count, hc_success_count = 0, 0
    if not cold_data_keys: return 0.0, 0.0
    test_keys = random.sample(cold_data_keys, min(len(cold_data_keys), NUM_QUERIES_EXP6))
    if not test_keys: return 0.0, 0.0
    
    for data_key in test_keys:
        for node in simulator.nodes:
            if data_key in node.lost_shards:
                del node.lost_shards[data_key]

        if data_key in simulator.ec_data_shard_map:
            group = random.choice(simulator.ec_node_groups)
            if simulator._simulate_retrieval_outcome(simulator.ec_data_shard_map[data_key][group], data_key, loss_prob):
                ec_success_count += 1
        if data_key in simulator.hc_data_to_shards_map:
            if simulator._simulate_retrieval_outcome(simulator.hc_data_to_shards_map[data_key], data_key, loss_prob):
                hc_success_count += 1
    return ec_success_count / len(test_keys), hc_success_count / len(test_keys)




def plot_final_results(all_results, shard_loss_probs):
    
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.serif'] = ['Times New Roman']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    x_axis_percent = [p * 100 for p in shard_loss_probs]

    
    
    color_map = {(10, 5): '#1f77b4', (10, 10): '#ff7f0e'}
    
    linestyle_map = {'ec': '-', 'hc': '--'}
    
    marker_map = {
        ('ec', (10, 5)): 'o',  
        ('hc', (10, 5)): 'o',
        ('ec', (10, 10)): '^', 
        ('hc', (10, 10)): '^'
    }
    
    LABEL_FONTSIZE = 40
    TICK_LABEL_FONTSIZE = 34
    LEGEND_TITLE_FONTSIZE = 30
    LEGEND_ITEM_FONTSIZE = 30

    for mal_prop, km_results in all_results.items():
        for (k, m), model_results in km_results.items():
            color = color_map.get((k, m), 'black')
            
            for model_type in ['ec', 'hc']:
                
                label = f'{"EC" if model_type == "ec" else "HC"} (k={k}, m={m})'
                marker = marker_map.get((model_type, (k, m)))
                linestyle = linestyle_map[model_type]

                
                ax.plot(x_axis_percent, model_results[model_type], 
                        marker=marker, linestyle=linestyle, color=color,
                        label=label, linewidth=4, markersize=8) 

    
    handles, labels = ax.get_legend_handles_labels()
    sorted_legend = sorted(zip(labels, handles), key=lambda t: t[0])
    labels_sorted = [l for l, h in sorted_legend]
    handles_sorted = [h for l, h in sorted_legend]
    
    
    ax.legend(handles_sorted, labels_sorted,
              loc='lower left', ncol=1, 
              fontsize=LEGEND_ITEM_FONTSIZE, 
              title=f'{int(MALICIOUS_NODE_PROPORTIONS[0]*100)}% Malicious Nodes',
              title_fontsize=LEGEND_TITLE_FONTSIZE)
              
    ax.set_xlabel('Shard loss probability(α) (%)', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('Query success rate', fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis='x', labelsize=TICK_LABEL_FONTSIZE)
    ax.tick_params(axis='y', labelsize=TICK_LABEL_FONTSIZE)
    
    ax.set_ylim(0.8, 1.01)
    ax.yaxis.set_major_formatter(plt.FuncFormatter('{:.0%}'.format))
    
    
    plt.tight_layout()
    plt.savefig("security_analysis_bw_compatible.png", dpi=300)
    print("\nPlot saved as security_analysis_bw_compatible.png")
    plt.show()




if __name__ == "__main__":
    
    
    if os.path.exists(RESULTS_FILE_PATH):
        print(f"Found existing results file: '{RESULTS_FILE_PATH}'. Loading data and plotting.")
        try:
            with open(RESULTS_FILE_PATH, 'rb') as f:
                saved_data = pickle.load(f)
                all_final_results = saved_data['results']
                shard_loss_probs = saved_data['x_axis']
            
            
            plot_final_results(all_final_results, shard_loss_probs)
            
        except Exception as e:
            print(f"Error loading or plotting from file: {e}")

    else:
        
        print(f"No results file found. Starting full simulation...")
        DATASET_PATH = 'blockchain_transactions.csv'
        all_final_results = {}

        try:
            print(f"Loading dataset from: {DATASET_PATH}")
            df_full = pd.read_csv(DATASET_PATH, usecols=['block_number', 'from_address'])
            df_full.dropna(subset=['from_address'], inplace=True)

            block_span = df_full['block_number'].max() - df_full['block_number'].min()
            HOT_WINDOW_BLOCKS = int(block_span * 0.1) if block_span > 0 else 100
            print(f"Hot window size: {HOT_WINDOW_BLOCKS} blocks")

            for mal_prop in MALICIOUS_NODE_PROPORTIONS:
                all_final_results[mal_prop] = {}
                for k, m in KM_PAIRS_TO_TEST:
                    print(f"\n{'='*70}\n>>> Running Exp for: {int(mal_prop*100)}% Malicious Nodes, (k={k}, m={m}) over {NUM_RUNS} runs <<<\n{'='*70}")
                    
                    runs_ec_results, runs_hc_results = [], []
                    for run in range(NUM_RUNS):
                        print(f"\n--- Starting Run {run + 1}/{NUM_RUNS} ---")
                        random.seed(42 + run)
                        
                        simulator = setup_world_with_real_data(df_full.copy(), k, m, HOT_WINDOW_BLOCKS)
                        cold_data_keys = list(simulator.models_data['hc']['cold'].keys())
                        
                        all_node_ids = [node.id for node in simulator.nodes]
                        malicious_ids = random.sample(all_node_ids, int(TOTAL_NODES * mal_prop))
                        for node in simulator.nodes: node.is_malicious = False
                        for node_id in malicious_ids:
                            simulator.nodes[node_id].is_malicious = True

                        single_run_ec, single_run_hc = [], []
                        for i, loss_prob in enumerate(SHARD_LOSS_PROBS_TO_TEST):
                            if i % 4 == 0: print(f"  Testing loss probability: {loss_prob:.0%}")
                            ec_rate, hc_rate = run_query_success_test(simulator, cold_data_keys, loss_prob)
                            single_run_ec.append(ec_rate)
                            single_run_hc.append(hc_rate)

                        runs_ec_results.append(single_run_ec)
                        runs_hc_results.append(single_run_hc)
                        print(f"  Run {run + 1} completed.")

                    avg_ec = np.mean(runs_ec_results, axis=0)
                    avg_hc = np.mean(runs_hc_results, axis=0)
                    all_final_results[mal_prop][(k, m)] = {'ec': avg_ec, 'hc': avg_hc}
                    print(f"\n>>> Averaged results calculated for {int(mal_prop*100)}% Malicious, (k={k}, m={m}). <<<")

            
            print("\nSimulation complete. Saving results to file...")
            data_to_save = {
                'results': all_final_results,
                'x_axis': SHARD_LOSS_PROBS_TO_TEST
            }
            with open(RESULTS_FILE_PATH, 'wb') as f:
                pickle.dump(data_to_save, f)
            print(f"Results successfully saved to '{RESULTS_FILE_PATH}'")

            
            plot_final_results(all_final_results, shard_loss_probs)

        except FileNotFoundError:
            print(f"错误：找不到文件 '{DATASET_PATH}'。请确保该文件与脚本在同一目录下。")
        except Exception as e:
            print(f"程序运行出错: {e}")