
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


STATE_DATA_COST_KB = 256 * 1024

NUM_EXPERIMENT_RUNS = 100  
RESULTS_FILE_PATH = 'storage_cost_vs_time_results_1.pkl'  

CSV_RESULTS_FILE_PATH = 'storage_cost_vs_time_results_1.csv'



HOT_STATE_RATIO = 0.25

COLD_STATE_K = 10

COLD_STATE_M = 5




class Node:
    def __init__(self, node_id):
        self.id = node_id

class HotColdNode(Node):
    def __init__(self, node_id):
        super().__init__(node_id)

def get_binary_group_info(n):
    if n == 0:
        return 0, []
    binary_str = bin(n)[2:]
    group_sizes = []
    for i, bit in enumerate(binary_str):
        if bit == '1':
            group_sizes.append(1 << (len(binary_str) - 1 - i))
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
        for node_id in hc_node_ids:
            self.hc_nodes[node_id] = HotColdNode(node_id)
        self.ec_chain_groups_count, group_sizes = get_binary_group_info(self.total_nodes)
        node_pool_for_ec = list(self.nodes.values())
        random.shuffle(node_pool_for_ec)
        start_idx = 0
        for size in group_sizes:
            group = node_pool_for_ec[start_idx: start_idx + size]
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
        costs['hc_total_cost'] = ((hot_data_kb + STATE_DATA_COST_KB) * self.total_nodes) + encoded_cold_kb
        
        
        
        total_ec_chain_cost = (hot_data_kb * self.total_nodes) + (encoded_cold_kb * self.ec_chain_groups_count)
        
        
        hot_state_cost_kb = STATE_DATA_COST_KB * HOT_STATE_RATIO
        cold_state_cost_kb = STATE_DATA_COST_KB * (1 - HOT_STATE_RATIO)
        
        
        encoded_cold_state_kb = cold_state_cost_kb * (COLD_STATE_K + COLD_STATE_M) / COLD_STATE_K
        total_encoded_cold_state_cost = encoded_cold_state_kb * self.ec_chain_groups_count
        
        
        total_hot_state_cost = hot_state_cost_kb * self.total_nodes
        
        
        total_ec_state_cost = total_hot_state_cost + total_encoded_cold_state_cost

        
        costs['ec_total_cost'] = total_ec_chain_cost + total_ec_state_cost
        
        return costs




def run_single_experiment_pass(df_bitcoin_blocks):
    simulator = UnifiedSimulator(TOTAL_NODES, INITIAL_HOTCOLD_NODES_RATIO)
    epochs = []
    fr_total = []
    ec_total = []
    hc_total = []
    
    
    for i in range(0, len(df_bitcoin_blocks), EPOCH_LENGTH):
        epoch_chunk = df_bitcoin_blocks.iloc[i:i+EPOCH_LENGTH]
        simulator.run_one_epoch(epoch_chunk)
        costs = simulator.get_storage_costs()
        
        epochs.append(simulator.current_epoch)
        fr_total.append(costs['fr_total_cost'])
        ec_total.append(costs['ec_total_cost'])
        hc_total.append(costs['hc_total_cost'])
    
    return {
        'epoch': epochs,
        'fr_total': fr_total,
        'ec_total': ec_total,
        'hc_total': hc_total
    }




def save_results_to_csv(results, csv_path):
    """
    将实验结果保存为CSV文件
    :param results: 包含epoch、fr_total、ec_total、hc_total的字典
    :param csv_path: CSV文件保存路径
    """
    try:
        
        df_results = pd.DataFrame({
            'epoch': results['epoch'],
            'fr_total_KB': results['fr_total'],
            'ec_total_KB': results['ec_total'],
            'hc_total_KB': results['hc_total']
        })
        
        df_results['fr_total_GB'] = df_results['fr_total_KB'] / 1024 / 1024
        df_results['ec_total_GB'] = df_results['ec_total_KB'] / 1024 / 1024
        df_results['hc_total_GB'] = df_results['hc_total_KB'] / 1024 / 1024
        
        
        df_results.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"\n结果已成功保存到CSV文件: {csv_path}")
        return df_results
    except Exception as e:
        print(f"保存CSV文件时出错: {e}")
        return None




def load_results_from_csv(csv_path):
    """
    从CSV文件加载实验结果
    :param csv_path: CSV文件路径
    :return: 适合绘图的结果字典
    """
    try:
        if not os.path.exists(csv_path):
            print(f"CSV文件不存在: {csv_path}")
            return None
        
        df_results = pd.read_csv(csv_path, encoding='utf-8')
        
        results = {
            'epoch': df_results['epoch'].tolist(),
            'fr_total': df_results['fr_total_KB'].tolist(),
            'ec_total': df_results['ec_total_KB'].tolist(),
            'hc_total': df_results['hc_total_KB'].tolist()
        }
        print(f"已从CSV文件加载结果: {csv_path}")
        return results
    except Exception as e:
        print(f"加载CSV文件时出错: {e}")
        return None




def kb_to_gb(x, pos):
    gb = x / 1024 / 1024
    return f'{gb:,.2f}'

def plot_final_results(history):
    
    plt.rcParams['font.family'] = 'Times New Roman'  
    plt.rcParams['font.size'] = 12  
    plt.rcParams['axes.unicode_minus'] = False  

    fig, ax1 = plt.subplots(figsize=(10, 8))
    
    LABEL_FONTSIZE = 40
    TICK_LABEL_FONTSIZE = 34
    LEGEND_FONTSIZE = 30  
    TITLE_FONTSIZE = 22

    color_ax1 = 'C0'
    
    ax1.set_xlabel('Epoch', fontsize=LABEL_FONTSIZE, fontfamily='Times New Roman')
    ax1.set_ylabel('FR total storage cost(GB)', fontsize=LABEL_FONTSIZE, 
                   color=color_ax1, fontfamily='Times New Roman')
    
    
    line1 = ax1.plot(history['epoch'], history['fr_total'], marker='o', markersize=5, 
                     markevery=20, label='FR', color=color_ax1, linewidth=4)
    ax1.tick_params(axis='y', labelcolor=color_ax1, labelsize=TICK_LABEL_FONTSIZE)
    ax1.tick_params(axis='x', labelsize=TICK_LABEL_FONTSIZE)
    ax1.yaxis.set_major_formatter(FuncFormatter(kb_to_gb))
    
    if history['fr_total']:
        max_left_y_value = max(history['fr_total'])
        ax1.set_ylim(bottom=0, top=max_left_y_value * 1.0)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Total cost for others(GB)', fontsize=LABEL_FONTSIZE, 
                   fontfamily='Times New Roman')
    
    
    line2 = ax2.plot(history['epoch'], history['ec_total'], marker='x', markersize=6, 
                     markevery=20, label='EC', color='C1', linewidth=4)
    line3 = ax2.plot(history['epoch'], history['hc_total'], marker='s', markersize=5, 
                     markevery=20, label='HC', color='C2', linewidth=4)
    
    ax2.tick_params(axis='y', labelsize=TICK_LABEL_FONTSIZE)
    ax2.yaxis.set_major_formatter(FuncFormatter(kb_to_gb))

    if history['ec_total']:
        max_right_y_value = max(history['ec_total'])
        ax2.set_ylim(bottom=0, top=max_right_y_value * 1.4)

    
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    
    ax1.legend(
        lines, labels, 
        loc='upper left',
        prop={  
            'family': 'Times New Roman',
            'size': LEGEND_FONTSIZE
        },
        frameon=True,  
        edgecolor='black'  
    )

    
    fig.tight_layout()
    plt.savefig("storage_cost_vs_time_final_plot.png", dpi=300, bbox_inches='tight')  
    print("\nFinal plot saved as storage_cost_vs_time_final_plot.png")
    plt.show()




if __name__ == "__main__":
    
    if os.path.exists(CSV_RESULTS_FILE_PATH):
        print(f"Found existing CSV results file: '{CSV_RESULTS_FILE_PATH}'. Loading data...")
        csv_results = load_results_from_csv(CSV_RESULTS_FILE_PATH)
        if csv_results:
            plot_final_results(csv_results)
            exit()
    
    
    if os.path.exists(RESULTS_FILE_PATH):
        print(f"Found existing results file: '{RESULTS_FILE_PATH}'. Loading data...")
        try:
            with open(RESULTS_FILE_PATH, 'rb') as f:
                saved_data = pickle.load(f)
            
            
            save_results_to_csv(saved_data, CSV_RESULTS_FILE_PATH)
            
            
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

            for key in ['fr_total', 'ec_total', 'hc_total']:
                all_costs_for_key = [run_result[key] for run_result in results_from_all_runs if run_result.get(key)]
                if all_costs_for_key:
                    avg_costs = np.mean(all_costs_for_key, axis=0)
                    final_averaged_results[key] = avg_costs.tolist()
                else:
                    final_averaged_results[key] = []
            
            
            with open(RESULTS_FILE_PATH, 'wb') as f:
                pickle.dump(final_averaged_results, f)
            print(f"Averaged results saved to '{RESULTS_FILE_PATH}'")
            
            
            save_results_to_csv(final_averaged_results, CSV_RESULTS_FILE_PATH)

            end_time = time.time()
            
            
            plot_final_results(final_averaged_results)
            
            print(f"\nTotal execution time for simulation: {end_time - start_time:.2f} seconds.")

        except FileNotFoundError:
            print(f"错误：找不到文件 '{DATASET_PATH}'")
        except Exception as e:
            print(f"程序运行出错: {e}")