
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




INITIAL_HOTCOLD_NODES_RATIO = 0.3
EPOCH_LENGTH = 32
HOT_WINDOW_EPOCHS = 8
K_PARAM = 10
M_PARAM = 5


FIXED_HC_NODE_COUNT = 76
NUM_EPOCHS_FOR_HISTORY = 800


STATE_DATA_COST_KB = 256 * 1024

NUM_EXPERIMENT_RUNS = 100 
RESULTS_FILE_PATH = 'per_node_storage_cost_results.pkl' 
CSV_RESULTS_FILE_PATH = 'per_node_storage_cost_results.csv' 



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
    def __init__(self, total_nodes, hc_ratio, fixed_hc_count=None):
        self.total_nodes = total_nodes
        self.nodes = {i: Node(i) for i in range(total_nodes)}
        self.hc_nodes = {}
        self.ec_node_groups = []
        self.ec_chain_groups_count = 0
        if fixed_hc_count is not None:
            num_hc_nodes = min(fixed_hc_count, total_nodes)
        else:
            num_hc_nodes = int(self.total_nodes * hc_ratio)
        self._initialize_pools(num_hc_nodes)
        self.hot_epochs_data = {}
        self.cold_epochs_data = {}
        self.current_epoch = 0

    def _initialize_pools(self, num_hc_nodes):
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
    
    node_counts_to_test = list(range(FIXED_HC_NODE_COUNT, 256, 1))
    single_run_costs = {'fr_cost': [], 'ec_cost': [], 'hc_cold_cost': [], 'hc_hot_cost': []}
    min_block = dataframe['block_number'].min()
    dataframe['epoch'] = (dataframe['block_number'] - min_block) // EPOCH_LENGTH
    grouped_by_epoch = list(dataframe.groupby('epoch'))
    standard_history_epochs = grouped_by_epoch[:NUM_EPOCHS_FOR_HISTORY]
    
    for num_nodes in node_counts_to_test:
        simulator = UnifiedSimulator(total_nodes=num_nodes, hc_ratio=INITIAL_HOTCOLD_NODES_RATIO, fixed_hc_count=FIXED_HC_NODE_COUNT)
        for _, epoch_chunk in standard_history_epochs:
            simulator.run_one_epoch(epoch_chunk)
        costs = simulator.get_storage_costs()
        single_run_costs['fr_cost'].append(costs['fr_per_node'])
        single_run_costs['ec_cost'].append(costs['ec_per_node_avg'])
        single_run_costs['hc_cold_cost'].append(costs['hc_cold_staking_node'])
        single_run_costs['hc_hot_cost'].append(costs['hc_hot_only_node'])
        
    return node_counts_to_test, single_run_costs




def save_results_to_csv(node_counts, costs, csv_path):
    """
    将实验结果保存到CSV文件
    :param node_counts: 节点数量列表
    :param costs: 各类成本数据字典
    :param csv_path: CSV文件保存路径
    """
    
    df_results = pd.DataFrame({
        'node_count': node_counts,
        'fr_cost': costs['fr_cost'],
        'ec_cost': costs['ec_cost'],
        'hc_cold_cost': costs['hc_cold_cost'],
        'hc_hot_cost': costs['hc_hot_cost']
    })
    
    df_results.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\nResults saved to CSV file: '{csv_path}'")




def load_results_from_csv(csv_path):
    """
    从CSV文件加载实验结果
    :param csv_path: CSV文件路径
    :return: node_counts列表, costs字典
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file '{csv_path}' not found!")
    
    df_results = pd.read_csv(csv_path, encoding='utf-8')
    node_counts = df_results['node_count'].tolist()
    costs = {
        'fr_cost': df_results['fr_cost'].tolist(),
        'ec_cost': df_results['ec_cost'].tolist(),
        'hc_cold_cost': df_results['hc_cold_cost'].tolist(),
        'hc_hot_cost': df_results['hc_hot_cost'].tolist()
    }
    return node_counts, costs




def kb_to_gb(x, pos):
    return f'{x / 1024 / 1024:,.1f}'

def plot_final_results(node_counts, costs_history):
    
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.serif'] = ['Times New Roman']
    
    fig, ax1 = plt.subplots(figsize=(10, 8))

    
    LABEL_FONTSIZE = 40
    TICK_LABEL_FONTSIZE = 34
    LEGEND_FONTSIZE = 30
    TITLE_FONTSIZE = 22

    
    color_ax1 = 'C0'
    ax1.set_xlabel('Total number of nodes in network', fontsize=LABEL_FONTSIZE)
    ax1.set_ylabel('FR node storage(GB)', fontsize=LABEL_FONTSIZE, color=color_ax1)
    
    line1 = ax1.plot(node_counts, costs_history['fr_cost'], marker='o', markevery=10, linestyle='-', 
                     label='FR Node Cost', color=color_ax1, markersize=8, linewidth=4)
    ax1.tick_params(axis='y', labelcolor=color_ax1, labelsize=TICK_LABEL_FONTSIZE)
    ax1.tick_params(axis='x', labelsize=TICK_LABEL_FONTSIZE)
    ax1.yaxis.set_major_formatter(FuncFormatter(kb_to_gb))

    if costs_history['fr_cost']:
        max_left_y_value = max(costs_history['fr_cost'])
        ax1.set_ylim(top=max_left_y_value * 1.05)

    
    ax2 = ax1.twinx()
    
    ax2.set_ylabel('Storage cost for others(GB)', fontsize=LABEL_FONTSIZE)
    
    line2 = ax2.plot(node_counts, costs_history['ec_cost'], marker='x', markevery=10, linestyle='-.', 
                     label='EC Node', color='C1', markersize=8, linewidth=4)
    line3 = ax2.plot(node_counts, costs_history['hc_cold_cost'], marker='s', markevery=10, linestyle='--', 
                     label='HC Cold Node', color='C2', markersize=8, linewidth=4)
    line4 = ax2.plot(node_counts, costs_history['hc_hot_cost'], marker='d', markevery=10, linestyle=':', 
                     label='HC Hot Node', color='C3', markersize=8, linewidth=4)
    ax2.tick_params(axis='y', labelsize=TICK_LABEL_FONTSIZE)
    
    ax2.yaxis.set_major_formatter(FuncFormatter(kb_to_gb))

    if costs_history['ec_cost']:
        max_right_y_value = max(costs_history['ec_cost'])
        ax2.set_ylim(top=max_right_y_value * 1.3)

    
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    
    ax1.legend(lines, labels, loc='upper right', fontsize=LEGEND_FONTSIZE)

    
    
    
    
    fig.tight_layout()
    plt.savefig("per_node_storage_cost_final_plot.png", dpi=300)
    print("\nFinal plot saved as per_node_storage_cost_final_plot.png")
    plt.show()




if __name__ == "__main__":
    
    
    if os.path.exists(CSV_RESULTS_FILE_PATH):
        print(f"Found existing CSV results file: '{CSV_RESULTS_FILE_PATH}'. Loading data and plotting.")
        try:
            node_counts, costs = load_results_from_csv(CSV_RESULTS_FILE_PATH)
            plot_final_results(node_counts, costs)
        except Exception as e:
            print(f"Error loading or plotting from CSV file: {e}")
    
    elif os.path.exists(RESULTS_FILE_PATH):
        print(f"Found existing results file: '{RESULTS_FILE_PATH}'. Loading data and plotting.")
        try:
            with open(RESULTS_FILE_PATH, 'rb') as f:
                saved_data = pickle.load(f)
            
            save_results_to_csv(saved_data['node_counts'], saved_data['costs'], CSV_RESULTS_FILE_PATH)
            plot_final_results(saved_data['node_counts'], saved_data['costs'])
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
            node_counts_axis = []
            for i in range(NUM_EXPERIMENT_RUNS):
                print(f"\n--- Starting Experiment Run {i+1}/{NUM_EXPERIMENT_RUNS} ---")
                random.seed(42 + i)
                
                node_counts_axis, single_run_costs = run_single_experiment_pass(df_bitcoin_blocks.copy())
                results_from_all_runs.append(single_run_costs)

            
            print("\nAll experiment runs complete. Averaging results...")
            final_averaged_costs = {}
            for key in results_from_all_runs[0].keys():
                
                all_costs_for_key = [run_result[key] for run_result in results_from_all_runs]
                
                avg_costs = np.mean(all_costs_for_key, axis=0)
                final_averaged_costs[key] = avg_costs.tolist()

            
            
            data_to_save = {
                'node_counts': node_counts_axis,
                'costs': final_averaged_costs
            }
            with open(RESULTS_FILE_PATH, 'wb') as f:
                pickle.dump(data_to_save, f)
            print(f"Averaged results saved to '{RESULTS_FILE_PATH}'")
            
            
            save_results_to_csv(node_counts_axis, final_averaged_costs, CSV_RESULTS_FILE_PATH)

            end_time = time.time()
            
            plot_final_results(node_counts_axis, final_averaged_costs)
            
            print(f"\nTotal execution time for simulation: {end_time - start_time:.2f} seconds.")

        except FileNotFoundError:
            print(f"错误：找不到文件 '{DATASET_PATH}'")
        except Exception as e:
            print(f"程序运行出错: {e}")