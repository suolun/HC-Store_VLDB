
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle
import time





TOTAL_NODES = 255
EPOCH_LENGTH = 32  
TARGET_EPOCHS = 400  


FAULT_PROB_PER_EPOCH = 0.10      
INITIAL_STAKE = 100.0            
PENALTY_STAKE_RATIO = 0.10       
T_MAX_MISTAKES = 5               


INITIAL_TREASURY = 5000.0
INFLATION_INITIAL_RATE = 0.01
INFLATION_DECAY_BASE = 0.98


Y_QUERIES_PER_EPOCH = 50000 
P2_NODE_QUERY_PRICE = 0.30       
A_BASE_SERVICE_RATE = 0.01       


RESULT_SAVE_PATH = 'simulation_results.csv'




class Node:
    def __init__(self, node_id):
        self.id = node_id
        self.stake = INITIAL_STAKE
        self.mistake_count = 0
        self.is_faulty = False

class FaultTolerantEconomicSimulator:
    def __init__(self, total_nodes):
        self.treasury = INITIAL_TREASURY
        self.nodes = {i: Node(i) for i in range(total_nodes)}
        self.current_epoch = 0
        
        self.history = {'epoch': [], 'breakeven_F': [], 'treasury': []}

    def inflation_rate_func(self, epoch):
        return INFLATION_INITIAL_RATE * (INFLATION_DECAY_BASE ** epoch)

    def run_one_epoch(self, block_height):
        penalty_from_stakes = 0.0
        newly_faulty_nodes = []
        
        for node in self.nodes.values():
            if not node.is_faulty and random.random() < FAULT_PROB_PER_EPOCH:
                node.is_faulty = True
                node.mistake_count += 1
                newly_faulty_nodes.append(node)

        for node in newly_faulty_nodes:
            count = node.mistake_count
            penalty = 0.0
            
            if 1 < count < T_MAX_MISTAKES:
                penalty = node.stake * PENALTY_STAKE_RATIO
            elif count >= T_MAX_MISTAKES:
                penalty = node.stake
            
            if penalty > 0:
                node.stake -= penalty
                penalty_from_stakes += penalty
        
        self.treasury += penalty_from_stakes

        H = block_height
        T_start = self.treasury
        
        
        current_inflation_rate = self.inflation_rate_func(self.current_epoch)
        inflation_I = T_start * current_inflation_rate

        cold_node_rewards_Fc = P2_NODE_QUERY_PRICE * Y_QUERIES_PER_EPOCH
        base_service_rewards_B = H * A_BASE_SERVICE_RATE
        total_rewards_R = cold_node_rewards_Fc + base_service_rewards_B
        
        breakeven_F = total_rewards_R - inflation_I
        
        
        self.history['epoch'].append(self.current_epoch)
        self.history['breakeven_F'].append(breakeven_F)
        self.history['treasury'].append(self.treasury)

        
        self.current_epoch += 1




def extend_block_data_to_target_epochs(df_blocks, target_epochs, epoch_length):
    required_blocks = target_epochs * epoch_length
    current_blocks = len(df_blocks)
    
    if current_blocks >= required_blocks:
        
        return df_blocks.head(required_blocks).reset_index(drop=True)
    else:
        
        print(f"警告：原始区块数据仅{current_blocks}个，需扩展到{required_blocks}个以满足{target_epochs}个epoch")
        extend_times = required_blocks // current_blocks
        remainder = required_blocks % current_blocks
        
        
        extended_blocks = pd.concat([df_blocks] * extend_times, ignore_index=True)
        
        if remainder > 0:
            extended_blocks = pd.concat([extended_blocks, df_blocks.head(remainder)], ignore_index=True)
        
        
        extended_blocks['block_number'] = range(1, len(extended_blocks)+1)
        return extended_blocks.reset_index(drop=True)




def save_simulation_results(history, save_path=RESULT_SAVE_PATH):
    df_results = pd.DataFrame(history)
    
    df_results.to_csv(save_path, index=False)
    print(f"\n模拟结果已保存到: {save_path}")

def load_simulation_results(load_path=RESULT_SAVE_PATH):
    if not os.path.exists(load_path):
        print(f"错误：找不到文件 '{load_path}'")
        return None
    
    df_results = pd.read_csv(load_path)
    
    history = {
        'epoch': df_results['epoch'].tolist(),
        'breakeven_F': df_results['breakeven_F'].tolist(),
        'treasury': df_results['treasury'].tolist()
    }
    print(f"\n已从 {load_path} 加载模拟结果")
    return history




def run_combined_experiment(dataframe):
    print("--- Running Combined Experiment with Faults and Penalties ---")
    
    simulator = FaultTolerantEconomicSimulator(TOTAL_NODES)
    
    min_block = dataframe['block_number'].min()
    dataframe['epoch'] = (dataframe['block_number'] - min_block) // EPOCH_LENGTH
    
    
    df_filtered = dataframe[dataframe['epoch'] < TARGET_EPOCHS].copy()
    grouped_by_epoch = df_filtered.groupby('epoch')

    print(f"Processing {len(df_filtered)} blocks across {len(grouped_by_epoch)} epochs (target: {TARGET_EPOCHS}).")
    
    
    
    for epoch_num, epoch_chunk in grouped_by_epoch:  
        start_block_height = epoch_chunk['block_number'].min()
        simulator.run_one_epoch(start_block_height)
    
    return simulator.history




def plot_breakeven_fees(history):
    fig, ax = plt.subplots(figsize=(14, 8))
    
    breakeven_F_clipped = np.maximum(0, history['breakeven_F'])
    
    
    ax.plot(history['epoch'], breakeven_F_clipped, marker='.', linestyle='-', markersize=4, label='Breakeven Total User Fees (F)')
    
    
    ax.set_xlabel('Epoch', fontsize=24)
    ax.set_ylabel('Breakeven Total Fees per Epoch (F) (USD)', fontsize=24)
    ax.set_xlim(0, TARGET_EPOCHS)  
    ax.tick_params(labelsize=20)
    ax.grid(True, which="both", ls="--")
    ax.legend(loc='upper left', fontsize=22)
    
    from matplotlib.ticker import StrMethodFormatter
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    
    plt.show()




if __name__ == "__main__":
    
    choice = input("请选择操作：\n1 - 运行新模拟并保存结果\n2 - 加载已有结果并绘图\n请输入数字 (1/2): ")
    
    if choice == '1':
        DATASET_PATH = 'bitcoin_blocks_3months.csv' 

        try:
            print(f"Loading dataset from: {DATASET_PATH}")
            df_full = pd.read_csv(DATASET_PATH, usecols=['block_number'])
            df_full.dropna(inplace=True)
            df_blocks = df_full.drop_duplicates(subset=['block_number']).sort_values(by='block_number').reset_index(drop=True)
            
            
            df_blocks = extend_block_data_to_target_epochs(df_blocks, TARGET_EPOCHS, EPOCH_LENGTH)
            
            print("Dataset loaded and extended successfully.")
            
            random.seed(42)
            final_history = run_combined_experiment(df_blocks)
            
            
            save_simulation_results(final_history)
            
            
            plot_breakeven_fees(final_history)

        except FileNotFoundError:
            print(f"\nERROR: Could not find file '{DATASET_PATH}'")
            print("Please ensure the CSV file is in the correct path.")
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            
            
            
    
    elif choice == '2':
        
        loaded_history = load_simulation_results()
        if loaded_history:
            plot_breakeven_fees(loaded_history)
    else:
        print("无效的选择，请输入 1 或 2")