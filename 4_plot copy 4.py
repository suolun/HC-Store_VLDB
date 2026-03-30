
import random
import hashlib
import bisect
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle
import time
from matplotlib.ticker import FuncFormatter




INITIAL_HOTCOLD_NODES_RATIO = 0.3
EPOCH_LENGTH = 32
HOT_WINDOW_EPOCHS = 8
K_PARAM = 10
M_PARAM = 5


STATE_DATA_COST_KB = 256 * 1024

FIXED_HC_NODE_COUNT = 76
NUM_EPOCHS_FOR_HISTORY = 800


NUM_EXPERIMENT_RUNS = 100 
RESULTS_FILE_PATH = 'storage_cost_results.pkl' 
CSV_RESULTS_FILE_PATH = 'storage_cost_results.csv' 



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
        encoded_cold_kb = cold_data_kb * (K_PARAM + M_PARAM) / K_PARAM
        
        
        costs['fr_total_cost'] = (hot_data_kb + cold_data_kb) * self.total_nodes + STATE_DATA_COST_KB
        costs['hc_total_cost'] = (hot_data_kb * self.total_nodes) + encoded_cold_kb  + STATE_DATA_COST_KB
        
        

        
        total_ec_chain_cost = (hot_data_kb * self.total_nodes) + (encoded_cold_kb * self.ec_chain_groups_count)
        
        
        hot_state_cost_kb = STATE_DATA_COST_KB * HOT_STATE_RATIO
        cold_state_cost_kb = STATE_DATA_COST_KB * (1 - HOT_STATE_RATIO)
        
        
        encoded_cold_state_kb = cold_state_cost_kb * (COLD_STATE_K + COLD_STATE_M) / COLD_STATE_K
        total_encoded_cold_state_cost = encoded_cold_state_kb * self.ec_chain_groups_count
        
        
        total_hot_state_cost = hot_state_cost_kb * self.total_nodes
        
        
        total_ec_state_cost = total_hot_state_cost + total_encoded_cold_state_cost

        
        costs['ec_total_cost'] = total_ec_chain_cost + total_ec_state_cost

        return costs




def run_single_experiment_pass(dataframe):
    node_counts_to_test = list(range(FIXED_HC_NODE_COUNT, 257, 1))
    single_run_costs = {'fr_total': [], 'ec_total': [], 'hc_total': []}
    min_block = dataframe['block_number'].min()
    dataframe['epoch'] = (dataframe['block_number'] - min_block) // EPOCH_LENGTH
    grouped_by_epoch = list(dataframe.groupby('epoch'))
    standard_history_epochs = grouped_by_epoch[:NUM_EPOCHS_FOR_HISTORY]

    for num_nodes in node_counts_to_test:
        simulator = UnifiedSimulator(total_nodes=num_nodes, hc_ratio=INITIAL_HOTCOLD_NODES_RATIO, fixed_hc_count=FIXED_HC_NODE_COUNT)
        for _, epoch_chunk in standard_history_epochs:
            simulator.run_one_epoch(epoch_chunk)
        costs = simulator.get_storage_costs()
        single_run_costs['fr_total'].append(costs['fr_total_cost'])
        single_run_costs['ec_total'].append(costs['ec_total_cost'])
        single_run_costs['hc_total'].append(costs['hc_total_cost'])

    return node_counts_to_test, single_run_costs




def save_results_to_csv(node_counts, costs, csv_path=CSV_RESULTS_FILE_PATH):
    """将实验结果保存为CSV文件"""
    df = pd.DataFrame({
        'node_count': node_counts,
        'fr_total_cost': costs['fr_total'],
        'ec_total_cost': costs['ec_total'],
        'hc_total_cost': costs['hc_total']
    })
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\n实验结果已保存到CSV文件: {csv_path}")




def load_results_from_csv(csv_path=CSV_RESULTS_FILE_PATH):
    """从CSV文件加载实验结果"""
    if not os.path.exists(csv_path):
        return None, None
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
        node_counts = df['node_count'].tolist()
        costs = {
            'fr_total': df['fr_total_cost'].tolist(),
            'ec_total': df['ec_total_cost'].tolist(),
            'hc_total': df['hc_total_cost'].tolist()
        }
        print(f"成功从CSV文件加载结果: {csv_path}")
        return node_counts, costs
    except Exception as e:
        print(f"加载CSV文件失败: {e}")
        return None, None




def kb_to_gb(x, pos):
    gb = x / 1024 / 1024
    return f'{gb:,.1f}'

def plot_final_results(node_counts, costs_history):
    fig, ax1 = plt.subplots(figsize=(14, 8))

    
    LABEL_FONTSIZE = 24
    TICK_LABEL_FONTSIZE = 22
    LEGEND_FONTSIZE = 22

    
    color_ax1 = 'C0'
    ax1.set_xlabel('Total Number of Nodes in Network', fontsize=LABEL_FONTSIZE)
    ax1.set_ylabel('FR Total Storage Cost (GB)', fontsize=LABEL_FONTSIZE, color=color_ax1)
    line1 = ax1.plot(node_counts, costs_history['fr_total'], marker='o', markevery=10, linestyle='-', label='FR Total Cost', color=color_ax1, markersize=8)
    ax1.tick_params(axis='y', labelcolor=color_ax1, labelsize=TICK_LABEL_FONTSIZE)
    ax1.tick_params(axis='x', labelsize=TICK_LABEL_FONTSIZE)
    ax1.yaxis.set_major_formatter(FuncFormatter(kb_to_gb))

    if costs_history['fr_total']:
        max_left_y_value = max(costs_history['fr_total'])
        ax1.set_ylim(bottom=0, top=max_left_y_value * 1.05)

    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Total Cost for Other Schemes (GB)', fontsize=LABEL_FONTSIZE)
    line2 = ax2.plot(node_counts, costs_history['ec_total'], marker='x', markevery=10, linestyle='-.', label='EC Total Cost', color='C1', markersize=8)
    line3 = ax2.plot(node_counts, costs_history['hc_total'], marker='s', markevery=10, linestyle='--', label='HC Total Cost', color='C2', markersize=8)
    ax2.tick_params(axis='y', labelsize=TICK_LABEL_FONTSIZE)
    ax2.yaxis.set_major_formatter(FuncFormatter(kb_to_gb))

    if costs_history['ec_total'] and costs_history['hc_total']:
        max_right_y_value = max(max(costs_history['ec_total']), max(costs_history['hc_total']))
        ax2.set_ylim(top=max_right_y_value * 2.0)

    
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=LEGEND_FONTSIZE)
    
    
    ax1.grid(True, which="both", ls="--")
    fig.tight_layout()
    plt.savefig("storage_cost_final_plot.png", dpi=300)
    print("\nFinal plot saved as storage_cost_final_plot.png")
    plt.show()




if __name__ == "__main__":
    
    node_counts, final_costs = load_results_from_csv()
    
    
    if node_counts is None or final_costs is None:
        if os.path.exists(RESULTS_FILE_PATH):
            print(f"\n未找到CSV文件，尝试加载Pkl文件: '{RESULTS_FILE_PATH}'")
            try:
                with open(RESULTS_FILE_PATH, 'rb') as f:
                    saved_data = pickle.load(f)
                node_counts = saved_data['node_counts']
                final_costs = saved_data['costs']
                
                save_results_to_csv(node_counts, final_costs)
            except Exception as e:
                print(f"加载Pkl文件失败: {e}")
                node_counts, final_costs = None, None
        else:
            node_counts, final_costs = None, None
    
    
    if node_counts is None or final_costs is None:
        print(f"\n未找到可用的结果文件，开始完整模拟...")
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
            final_costs = {}
            for key in results_from_all_runs[0].keys():
                
                all_costs_for_key = [run_result[key] for run_result in results_from_all_runs]
                
                avg_costs = np.mean(all_costs_for_key, axis=0)
                final_costs[key] = avg_costs.tolist()

            
            data_to_save = {
                'node_counts': node_counts_axis,
                'costs': final_costs
            }
            with open(RESULTS_FILE_PATH, 'wb') as f:
                pickle.dump(data_to_save, f)
            print(f"Averaged results saved to '{RESULTS_FILE_PATH}'")
            
            
            save_results_to_csv(node_counts_axis, final_costs)
            
            node_counts = node_counts_axis
            
            end_time = time.time()
            print(f"\nTotal execution time for simulation: {end_time - start_time:.2f} seconds.")

        except FileNotFoundError:
            print(f"错误：找不到文件 '{DATASET_PATH}'")
            exit(1)
        except Exception as e:
            print(f"程序运行出错: {e}")
            exit(1)
    
    
    if node_counts and final_costs:
        plot_final_results(node_counts, final_costs)
    else:
        print("没有可用的结果数据来绘图")