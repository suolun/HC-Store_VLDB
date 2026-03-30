
import random
import hashlib
import bisect
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os  


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  





TOTAL_BACKGROUND_NODES = 255 
HC_RATIO_BACKGROUND = 0.30   
NUM_EPOCHS_TO_SIMULATE = 100   


ACCOUNT_CREATION_FEE = 1 
EPOCH_BASE_REWARD = 1500
STAKE_PER_NODE = 100



REWARD_PENALTY = STAKE_PER_NODE * 1 

STAKE_PENALTY_RATIO = 0.10

T_MAX_MISTAKES = 5


K_PARAM = 10
M_PARAM = 5




class User:
    def __init__(self, user_id):
        self.id = user_id
        self.accounts = []

    def get_net_profit(self):
        
        total_rewards_and_penalties = sum(acc.rewards_balance for acc in self.accounts)
        total_fees = len(self.accounts) * ACCOUNT_CREATION_FEE
        return total_rewards_and_penalties - total_fees

class NetworkNode:
    def __init__(self, node_id, owner_account_id):
        self.id = node_id
        self.owner_account_id = owner_account_id
        self.is_lazy = False
        
        
        self.stake = STAKE_PER_NODE  
        self.mistake_count = 0       
        self.is_jailed = False       

    def prove(self):
        
        if self.is_jailed:
            return False
        return not self.is_lazy

class Account:
    def __init__(self, account_id, owner_user_id):
        self.id = account_id
        self.owner_user_id = owner_user_id
        self.nodes_owned = []
        self.rewards_balance = 0 

    def deploy_nodes(self, simulator, num_nodes):
        new_nodes = simulator.add_nodes_for_account(self.id, num_nodes)
        self.nodes_owned.extend(n.id for n in new_nodes)
        return new_nodes

class SybilExperimentSimulator:
    def __init__(self, total_background_nodes):
        self.nodes = {}
        self.accounts = {}
        self.hash_ring = []
        self.cold_data_shards_map = {}
        self.next_node_id = 0
        self.next_account_id = 0
        bg_user = User("background_user")
        num_bg_hc_nodes = int(total_background_nodes * HC_RATIO_BACKGROUND)
        bg_acc = self.add_account(bg_user, "bg_acc")
        bg_acc.deploy_nodes(self, num_bg_hc_nodes)

    def _get_hash(self, key):
        return int.from_bytes(hashlib.sha256(key.encode()).digest(), 'big')

    def add_account(self, owner_user, name_prefix):
        acc_id = f"{name_prefix}_{self.next_account_id}"
        account = Account(acc_id, owner_user.id)
        self.accounts[acc_id] = account
        owner_user.accounts.append(account)
        self.next_account_id += 1
        return account

    def add_nodes_for_account(self, owner_acc_id, num_nodes):
        new_nodes = []
        for _ in range(num_nodes):
            node_id = self.next_node_id
            node = NetworkNode(node_id, owner_acc_id)
            self.nodes[node_id] = node
            node_hash = self._get_hash(f"{owner_acc_id}-{node_id}")
            bisect.insort(self.hash_ring, (node_hash, node_id))
            self.next_node_id += 1
            new_nodes.append(node)
        return new_nodes

    def distribute_cold_data(self, num_data_items):
        
        working_nodes = [node for node in self.nodes.values() if not node.is_lazy and not node.is_jailed]
        if not working_nodes:
            self.cold_data_shards_map = {}
            return
        total_shards = num_data_items * (K_PARAM + M_PARAM)
        num_working_nodes = len(working_nodes)
        shards_per_node = total_shards // num_working_nodes
        remainder_shards = total_shards % num_working_nodes
        self.cold_data_shards_map = {}
        random.shuffle(working_nodes)
        for i, node in enumerate(working_nodes):
            num_shards_for_this_node = shards_per_node
            if i < remainder_shards:
                num_shards_for_this_node += 1
            self.cold_data_shards_map[node.id] = num_shards_for_this_node

    def run_epoch(self):
        
        total_shards_stored_by_workers = sum(self.cold_data_shards_map.get(n_id, 0) for n_id, n in self.nodes.items() if not n.is_lazy and not n.is_jailed)
        if total_shards_stored_by_workers == 0: return

        for node_id, node in self.nodes.items():
            
            if node.is_jailed:
                continue
                
            owner_account = self.accounts[node.owner_account_id]
            
            if node.prove():
                
                shards_count = self.cold_data_shards_map.get(node_id, 0)
                reward = EPOCH_BASE_REWARD * (shards_count / total_shards_stored_by_workers)
                owner_account.rewards_balance += reward
            else:
                
                node.mistake_count += 1
                count = node.mistake_count
                
                
                owner_account.rewards_balance -= REWARD_PENALTY
                
                
                stake_penalty = 0.0
                if 1 < count < T_MAX_MISTAKES:
                    
                    stake_penalty = STAKE_PER_NODE * STAKE_PENALTY_RATIO
                elif count >= T_MAX_MISTAKES:
                    
                    stake_penalty = node.stake
                
                
                if stake_penalty > 0:
                    
                    actual_stake_loss = min(node.stake, stake_penalty)
                    
                    
                    node.stake -= actual_stake_loss
                    
                    
                    owner_account.rewards_balance -= actual_stake_loss
                    
                    
                    if node.stake <= 0:
                        node.stake = 0
                        node.is_jailed = True




def run_sybil_experiment(real_data_block_count):
    sybil_node_counts = [10, 20, 50, 100]  
    results = {'Honest': [], 'Sybil (Lazy)': [], 'Sybil (Working)': []}

    for num_nodes in sybil_node_counts:
        print(f"\n--- Testing with User owning {num_nodes} nodes ---")

        
        sim_honest = SybilExperimentSimulator(TOTAL_BACKGROUND_NODES)
        user_honest = User("user_honest")
        acc_honest = sim_honest.add_account(user_honest, "honest_acc")
        acc_honest.deploy_nodes(sim_honest, num_nodes)
        sim_honest.distribute_cold_data(real_data_block_count)
        for _ in range(NUM_EPOCHS_TO_SIMULATE): sim_honest.run_epoch()
        results['Honest'].append(user_honest.get_net_profit())
        print(f"  Honest Strategy Profit: {user_honest.get_net_profit():.2f}")

        
        sim_lazy = SybilExperimentSimulator(TOTAL_BACKGROUND_NODES)
        user_lazy = User("user_lazy")
        nodes_created = []
        for i in range(num_nodes):
            acc = sim_lazy.add_account(user_lazy, f"lazy_sybil_acc_{i}")
            nodes = acc.deploy_nodes(sim_lazy, 1)
            nodes_created.extend(nodes)
        for i in range(0, len(nodes_created)): nodes_created[i].is_lazy = True
        sim_lazy.distribute_cold_data(real_data_block_count)
        for _ in range(NUM_EPOCHS_TO_SIMULATE): sim_lazy.run_epoch()
        results['Sybil (Lazy)'].append(user_lazy.get_net_profit())
        print(f"  Lazy Sybil Strategy Profit: {user_lazy.get_net_profit():.2f}")

        
        sim_working = SybilExperimentSimulator(TOTAL_BACKGROUND_NODES)
        user_working = User("user_working")
        for i in range(num_nodes):
            acc = sim_working.add_account(user_working, f"working_sybil_acc_{i}")
            acc.deploy_nodes(sim_working, 1)
        sim_working.distribute_cold_data(real_data_block_count)
        for _ in range(NUM_EPOCHS_TO_SIMULATE): sim_working.run_epoch()
        results['Sybil (Working)'].append(user_working.get_net_profit())
        print(f"  Working Sybil Strategy Profit: {user_working.get_net_profit():.2f}")
        
    return sybil_node_counts, results




def save_results_to_csv(x_labels, results, save_path="sybil_experiment_results.csv"):
    """
    将实验结果保存到CSV文件
    :param x_labels: 横坐标（节点数量列表）
    :param results: 实验结果字典
    :param save_path: 保存路径
    """
    
    df = pd.DataFrame({
        "Node_Count": x_labels,
        "Honest_Profit": results["Honest"],
        "Lazy_Sybil_Profit": results["Sybil (Lazy)"],
        "Working_Sybil_Profit": results["Sybil (Working)"]
    })
    
    df.to_csv(save_path, index=False, encoding="utf-8")
    print(f"\n实验结果已保存到: {os.path.abspath(save_path)}")




def load_results_from_csv(load_path="sybil_experiment_results.csv"):
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"文件不存在: {os.path.abspath(load_path)}")
    
    df = pd.read_csv(load_path, encoding="utf-8")
    
    x_labels = df["Node_Count"].tolist()
    results = {
        "Honest": df["Honest_Profit"].tolist(),
        "Sybil (Lazy)": df["Lazy_Sybil_Profit"].tolist(),
        "Sybil (Working)": df["Working_Sybil_Profit"].tolist()
    }
    print(f"\n已从 {os.path.abspath(load_path)} 加载实验结果")
    return x_labels, results




def plot_exp7_sybil_rewards(x_labels, results, save_png_path="sybil_experiment_plot.png", dpi=300):
    x = np.arange(len(x_labels)) * 4.0  
    width = 0.8  
    fig, ax = plt.subplots(figsize=(16, 14))  
    
    
    rects1 = ax.bar(x - width, results['Honest'], width, 
                    label='Honest', 
                    color='blue')
    rects2 = ax.bar(x, results['Sybil (Lazy)'], width, 
                    label='Withholding Sybil', 
                    color='red')
    rects3 = ax.bar(x + width, results['Sybil (Working)'], width, 
                    label='Compliant Sybil', 
                    color='orange')
    
    
    all_values = results['Honest'] + results['Sybil (Lazy)'] + results['Sybil (Working)']
    y_max = max(all_values)
    y_min = min(all_values)
    
    ax.set_ylim(y_min * 1.15, y_max * 1.15)  
    
    
    ax.set_ylabel('Total revenue per cold node', fontsize=46)
    ax.set_xlabel('Number of virtual nodes from staking', fontsize=46)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=36)
    ax.tick_params(axis='y', which='major', labelsize=36)
    
    
    ax.legend(loc='lower left', fontsize=36, frameon=True)
    
    
    ax.axhline(0, color='black', lw=0.8)
    
    
    def add_outer_labels(rects, fontsize=28, padding=8):
        for rect in rects:
            height = rect.get_height()
            
            if height > 0:
                y_pos = height + padding
                va_align = 'bottom'
            
            else:
                y_pos = height - padding
                va_align = 'top'
            
            
            ax.text(rect.get_x() + rect.get_width()/2., y_pos,
                    f'{int(height)}',  
                    ha='center', va=va_align, fontsize=fontsize, color='black',) 
    
    
    add_outer_labels(rects1, 34, 1111)
    add_outer_labels(rects2, 34, 1111)
    add_outer_labels(rects3, 34, 1111)
    
    
    plt.subplots_adjust(top=0.92, bottom=0.12, left=0.08, right=0.98)
    fig.tight_layout()
    
    
    plt.savefig(save_png_path, dpi=dpi, bbox_inches='tight')
    print(f"\n图表已保存为PNG文件: {os.path.abspath(save_png_path)}")
    
    
    plt.show()




if __name__ == "__main__":
    DATASET_PATH = 'bitcoin_blocks_3months.csv'
    RESULT_SAVE_PATH = "sybil_experiment_results.csv"  
    PLOT_SAVE_PATH = "7_sybil_experiment_plot.png"  

    try:
        
        run_experiment = True  

        if run_experiment:
            
            print(f"Loading Bitcoin dataset from: {DATASET_PATH}")
            if os.path.exists(DATASET_PATH):
                df_bitcoin_blocks = pd.read_csv(DATASET_PATH)
                REAL_DATA_BLOCK_COUNT = len(df_bitcoin_blocks)
            else:
                print(f"警告：未找到 {DATASET_PATH}，使用模拟数据（1000个区块）")
                REAL_DATA_BLOCK_COUNT = 1000
            print(f"Using real data scale: {REAL_DATA_BLOCK_COUNT} total blocks.")
            
            random.seed(42)
            x_labels, final_results = run_sybil_experiment(REAL_DATA_BLOCK_COUNT)
            
            
            save_results_to_csv(x_labels, final_results, RESULT_SAVE_PATH)
        else:
            
            x_labels, final_results = load_results_from_csv(RESULT_SAVE_PATH)
        
        
        plot_exp7_sybil_rewards(x_labels, final_results, save_png_path=PLOT_SAVE_PATH, dpi=300)

    except FileNotFoundError as e:
        print(f"\n" + "="*60)
        print(f"错误：{e}")
        print("请检查文件路径是否正确！")
        print("="*60)
    except Exception as e:
        print(f"\n运行时发生错误: {e}")
        import traceback
        traceback.print_exc()  