
import random
import hashlib
import bisect
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle
import time
from matplotlib.ticker import FuncFormatter, StrMethodFormatter




TOTAL_NODES = 255
INITIAL_HOTCOLD_NODES_RATIO = 0.3
EPOCH_LENGTH = 32
HOT_WINDOW_EPOCHS = 8
K_PARAM = 10
M_PARAM = 5


NUM_EXPERIMENT_RUNS = 100 
RESULTS_FILE_PATH = 'per_node_storage_cost_vs_time_results_1.pkl' 


STATE_DATA_COST_KB = 256 * 1024



HOT_STATE_RATIO = 0.25 

COLD_STATE_K = 10

COLD_STATE_M = 5





class Node:
    def __init__(self, node_id): self.id = node_id
class HotColdNode(Node):
    def __init__(self, node_id): super().__init__(node_id)
def get_binary_group_info(n):
    if n == 0: return 0, []
    binary_str = bin(n)[2:]; group_sizes = []
    for i, bit in enumerate(binary_str):
        if bit == '1': group_sizes.append(1 << (len(binary_str) - 1 - i))
    return len(group_sizes), group_sizes
    
class UnifiedSimulator:
    def __init__(self, total_nodes, hc_ratio):
        self.total_nodes = total_nodes
        self.nodes = {i: Node(i) for i in range(total_nodes)}
        self.hc_nodes = {}
        self.ec_node_groups = []
        self.ec_chain_groups_count = 0
        self._initialize_pools(hc_ratio)
        self.hot_epochs_data = {}
        self.cold_epochs_data = {}
        self.current_epoch = 0

    def _initialize_pools(self, hc_ratio):
        num_hc_nodes = int(self.total_nodes * hc_ratio)
        node_ids = list(self.nodes.keys())
        random.shuffle(node_ids)
        hc_node_ids = node_ids[:num_hc_nodes]
        for node_id in hc_node_ids: self.hc_nodes[node_id] = HotColdNode(node_id)
        self.ec_chain_groups_count, group_sizes = get_binary_group_info(self.total_nodes)
        node_pool_for_ec = list(self.nodes.values())
        random.shuffle(node_pool_for_ec)
        start_idx = 0
        for size in group_sizes:
            group = node_pool_for_ec[start_idx : start_idx + size]
            self.ec_node_groups.append(tuple(g for g in group))
            start_idx += size

    def run_one_epoch(self, epoch_data_chunk):
        self.current_epoch += 1
        total_size_kb = epoch_data_chunk['block_size'].sum() / 1024
        self.hot_epochs_data[self.current_epoch] = total_size_kb
        epoch_to_cool = self.current_epoch - HOT_WINDOW_EPOCHS
        if epoch_to_cool in self.hot_epochs_data:
            size_to_move = self.hot_epochs_data.pop(epoch_to_cool)
            self.cold_epochs_data[epoch_to_cool] = size_to_move

    def get_storage_costs(self):
        costs = {}
        hot_data_kb = sum(self.hot_epochs_data.values())
        cold_data_kb = sum(self.cold_epochs_data.values())
        
        
        
        
        costs['fr_per_node'] = hot_data_kb + cold_data_kb + STATE_DATA_COST_KB
        
        
        costs['hc_hot_only_node'] = hot_data_kb + STATE_DATA_COST_KB
        
        
        encoded_cold_kb = cold_data_kb * (K_PARAM + M_PARAM) / K_PARAM
        num_hc_nodes = len(self.hc_nodes)
        if num_hc_nodes > 0:
            costs['hc_cold_staking_node'] = hot_data_kb + (encoded_cold_kb / num_hc_nodes) + STATE_DATA_COST_KB
        else:
            costs['hc_cold_staking_node'] = hot_data_kb + STATE_DATA_COST_KB
            
        
        
        
        total_ec_chain_cost = (hot_data_kb * self.total_nodes) + (encoded_cold_kb * self.ec_chain_groups_count)
        base_ec_avg_chain_cost = total_ec_chain_cost / self.total_nodes if self.total_nodes > 0 else 0
        
        
        hot_state_cost_kb = STATE_DATA_COST_KB * HOT_STATE_RATIO
        cold_state_cost_kb = STATE_DATA_COST_KB * (1 - HOT_STATE_RATIO)
        
        
        encoded_cold_state_kb = cold_state_cost_kb * (COLD_STATE_K + COLD_STATE_M) / COLD_STATE_K
        total_encoded_cold_state_cost = encoded_cold_state_kb * self.ec_chain_groups_count
        
        
        total_hot_state_cost = hot_state_cost_kb * self.total_nodes
        
        
        total_ec_state_cost = total_hot_state_cost + total_encoded_cold_state_cost
        avg_ec_state_cost = total_ec_state_cost / self.total_nodes if self.total_nodes > 0 else 0
        
        
        costs['ec_per_node_avg'] = base_ec_avg_chain_cost + avg_ec_state_cost
        
        return costs




def run_single_experiment_pass(dataframe):
    
    simulator = UnifiedSimulator(TOTAL_NODES, INITIAL_HOTCOLD_NODES_RATIO)
    
    history = {'epoch': [], 'fr_cost': [], 'ec_cost': [], 'hc_cold_cost': [], 'hc_hot_cost': []}
    min_block = dataframe['block_number'].min()
    dataframe['epoch'] = (dataframe['block_number'] - min_block) // EPOCH_LENGTH
    
    
    grouped_by_epoch = dataframe.groupby('epoch')
    
    
    
    
    
    
    
    

    for _, epoch_chunk in grouped_by_epoch:
        simulator.run_one_epoch(epoch_chunk)
        costs = simulator.get_storage_costs()
        
        
        history['epoch'].append(simulator.current_epoch)
        history['fr_cost'].append(costs['fr_per_node'])
        history['ec_cost'].append(costs['ec_per_node_avg'])
        history['hc_cold_cost'].append(costs['hc_cold_staking_node'])
        history['hc_hot_cost'].append(costs['hc_hot_only_node'])
        
    return history



def kb_to_gb(x, pos):
    gb = x / 1024 / 1024
    return f'{gb:,.1f}'

def kb_to_mb(x, pos):
    mb = x / 1024
    return f'{mb:,.0f}'

def plot_final_results(history):
    
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.serif'] = ['Times New Roman']  
    plt.rcParams['axes.unicode_minus'] = False  
    
    fig, ax1 = plt.subplots(figsize=(10, 8))
    
    LABEL_FONTSIZE = 40
    TICK_LABEL_FONTSIZE = 34
    LEGEND_FONTSIZE = 30
    TITLE_FONTSIZE = 26

    color_ax1 = 'C0'
    
    ax1.set_xlabel('Epoch', fontsize=LABEL_FONTSIZE, fontname='Times new roman')
    ax1.set_ylabel('FR node storage cost(GB)', fontsize=LABEL_FONTSIZE, color=color_ax1, fontname='Times New Roman')
    
    
    line1 = ax1.plot(history['epoch'], history['fr_cost'], marker='o', color=color_ax1, label='FR node', markersize=8, markevery=20, linewidth=4)
    
    ax1.set_ylim(bottom=0)
    ax1.tick_params(axis='y', labelcolor=color_ax1, labelsize=TICK_LABEL_FONTSIZE)
    ax1.tick_params(axis='x', labelsize=TICK_LABEL_FONTSIZE)
    
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontname('Times New Roman')
    ax1.yaxis.set_major_formatter(FuncFormatter(kb_to_gb))
    
    
    all_epochs = history['epoch']
    if all_epochs:
        
        xticks = np.arange(0, all_epochs[-1], step=100)
        ax1.set_xticks(xticks)
    ax1.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

    ax2 = ax1.twinx()
    
    ax2.set_ylabel('Storage cost for others(GB)', fontsize=LABEL_FONTSIZE, fontname='Times New Roman')
    
    
    line2 = ax2.plot(history['epoch'], history['ec_cost'], marker='x', color='C1', label='EC node', markersize=8, markevery=20, linewidth=4)
    line3 = ax2.plot(history['epoch'], history['hc_cold_cost'], marker='s', color='C2', label='HC cold node', markersize=8, markevery=20, linewidth=4)
    line4 = ax2.plot(history['epoch'], history['hc_hot_cost'], marker='d', color='C3', label='HC hot node', markersize=8, markevery=20, linewidth=4)
    
    ax2.tick_params(axis='y', labelsize=TICK_LABEL_FONTSIZE)
    
    for label in ax2.get_yticklabels():
        label.set_fontname('Times New Roman')
    
    ax2.yaxis.set_major_formatter(FuncFormatter(kb_to_gb))

    if history['ec_cost']:
        max_right_y_value = max(history['ec_cost'])
        ax2.set_ylim(bottom=0, top=max_right_y_value * 1.3)

    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    
    legend = ax1.legend(lines, labels, loc='upper left', fontsize=LEGEND_FONTSIZE)
    for text in legend.get_texts():
        text.set_fontname('Times New Roman')

    
    fig.tight_layout()
    plt.savefig("per_node_storage_cost_vs_time_final_plot.png", dpi=500)
    print("\nFinal plot saved as per_node_storage_cost_vs_time_final_plot.png")
    plt.show()




if __name__ == "__main__":
    
    if os.path.exists(RESULTS_FILE_PATH):
        print(f"Found existing results file: '{RESULTS_FILE_PATH}'. Loading data...")
        try:
            with open(RESULTS_FILE_PATH, 'rb') as f:
                saved_data = pickle.load(f)
            
            
            plot_final_results(saved_data)
            
        except Exception as e:
            print(f"Error loading or plotting from file: {e}")

    else:
        print(f"No results file found. Starting full simulation...")
        DATASET_PATH = 'bitcoin_blocks_3months.csv'

        try:
            print(f"Loading Bitcoin dataset from: {DATASET_PATH}")
            df_bitcoin_blocks = pd.read_csv(DATASET_PATH)
            df_bitcoin_blocks = df_bitcoin_blocks.sort_values(by='block_number').reset_index(drop=True)
            print("Dataset loaded successfully.")
            
            start_time = time.time()
            
            results_from_all_runs = []
            for i in range(NUM_EXPERIMENT_RUNS):
                print(f"\n--- Starting Experiment Run {i+1}/{NUM_EXPERIMENT_RUNS} ---")
                random.seed(42 + i)
                single_run_result = run_single_experiment_pass(df_bitcoin_blocks.copy())
                results_from_all_runs.append(single_run_result)

            print("\nAll experiment runs complete. Averaging results...")
            
            
            first_valid_history = next((r for r in results_from_all_runs if r.get('epoch')), {'epoch': []})
            final_averaged_results = {'epoch': first_valid_history['epoch']}
            
            for key in ['fr_cost', 'ec_cost', 'hc_cold_cost', 'hc_hot_cost']:
                all_costs_for_key = [run_result[key] for run_result in results_from_all_runs if run_result.get(key)]
                if all_costs_for_key:
                    avg_costs = np.mean(all_costs_for_key, axis=0)
                    final_averaged_results[key] = avg_costs.tolist()
                else:
                    final_averaged_results[key] = []
            
            with open(RESULTS_FILE_PATH, 'wb') as f:
                pickle.dump(final_averaged_results, f)
            print(f"Averaged results saved to '{RESULTS_FILE_PATH}'")
            
            end_time = time.time()
            
            
            plot_final_results(final_averaged_results)

            print(f"\nTotal execution time for simulation: {end_time - start_time:.2f} seconds.")

        except FileNotFoundError:
            print(f"错误：找不到文件 '{DATASET_PATH}'")
        except Exception as e:
            print(f"程序运行出错: {e}")