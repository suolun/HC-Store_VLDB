
import random
import matplotlib.pyplot as plt
import numpy as np
import csv
import os  





NUM_SIMULATION_RUNS = 10000      
EPOCH_SIZE_KB = 1024 * 4           
CSV_SAVE_PATH = "repair_bandwidth_results.csv"  



HC_PARAMS_1 = {'k': 10, 'm': 5, 'label': 'HC (k=10, m=5)'}

HC_PARAMS_2 = {'k': 10, 'm': 10, 'label': 'HC (k=10, m=10)'}

FR_PARAMS = {'label': 'Full Replication'}






def simulate_average_hc_repair(k, m, epoch_size, num_runs):
    print(f"--- Simulating average HC repair for (k={k}, m={m}) ---")
    total_repair_bandwidth = 0
    shard_size = epoch_size / k

    for _ in range(num_runs):
        
        lost_shards_count = random.randint(1, m)
        
        
        current_run_bandwidth = (k * shard_size) + (lost_shards_count * shard_size)
        total_repair_bandwidth += current_run_bandwidth
        
    
    average_bandwidth = total_repair_bandwidth / num_runs
    print(f"  Average Repair Bandwidth over {num_runs} runs: {average_bandwidth:.2f} KB")
    return 2*average_bandwidth

def calculate_fr_repair(epoch_size, k):
    print(f"--- Calculating Full Replication (FR) repair for 1 lost node ---")
    
    
    repair_bandwidth = epoch_size
    
    print(f"  Repair Bandwidth for 1 epoch of data: {repair_bandwidth:.2f} KB")
    return 2*repair_bandwidth






def save_results_to_csv(results, file_path):
    
    headers = ['scheme_label', 'repair_bandwidth_kb', 'repair_bandwidth_mb']
    
    
    dir_path = os.path.dirname(file_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    
    
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for res in results:
            
            bandwidth_mb = res['bandwidth'] / 1024
            writer.writerow({
                'scheme_label': res['label'],
                'repair_bandwidth_kb': round(res['bandwidth'], 2),
                'repair_bandwidth_mb': round(bandwidth_mb, 2)
            })
    print(f"\n结果已保存到CSV文件: {os.path.abspath(file_path)}")  

def load_results_from_csv(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV文件不存在！绝对路径：{os.path.abspath(file_path)}")
    
    results = []
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            results.append({
                'label': row['scheme_label'],
                'bandwidth': float(row['repair_bandwidth_kb'])
            })
    print(f"\n已从CSV文件加载结果: {os.path.abspath(file_path)}")
    return results





def plot_repair_bandwidth_bar_chart(results):
    labels = [res['label'] for res in results]
    bandwidths_kb = [res['bandwidth'] for res in results]
    
    
    bandwidths_mb = [bw / 1024 for bw in bandwidths_kb]

    fig, ax = plt.subplots(figsize=(14, 8))

    
    bars = ax.bar(labels, bandwidths_mb, color=['#1f77b4', '#ff7f0e', '#2ca02c'])

    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', 
                va='bottom', ha='center', fontsize=16)

    
    ax.set_ylabel('Average Repair Bandwidth Overhead (MB)', fontsize=24)
    
    
    
    ax.set_xticks([])
    ax.set_xlabel('')
    
    
    ax.set_ylim(0, max(bandwidths_mb) * 1.2)
    ax.tick_params(labelsize=20)
    
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.legend(bars, labels, fontsize=20, loc='upper left')

    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    
    RUN_SIMULATION = True
    
    if RUN_SIMULATION:
        
        avg_bw_hc1 = simulate_average_hc_repair(
            k=HC_PARAMS_1['k'], 
            m=HC_PARAMS_1['m'], 
            epoch_size=EPOCH_SIZE_KB,
            num_runs=NUM_SIMULATION_RUNS
        )
        
        avg_bw_hc2 = simulate_average_hc_repair(
            k=HC_PARAMS_2['k'], 
            m=HC_PARAMS_2['m'], 
            epoch_size=EPOCH_SIZE_KB,
            num_runs=NUM_SIMULATION_RUNS
        )
        
        
        bw_fr = calculate_fr_repair(
            epoch_size=EPOCH_SIZE_KB,
            k=HC_PARAMS_1['k'] 
        )

        
        final_results = [
            {'label': HC_PARAMS_1['label'], 'bandwidth': avg_bw_hc1},
            {'label': HC_PARAMS_2['label'], 'bandwidth': avg_bw_hc2},
            {'label': FR_PARAMS['label'], 'bandwidth': bw_fr}
        ]
        
        
        save_results_to_csv(final_results, CSV_SAVE_PATH)
    else:
        
        try:
            final_results = load_results_from_csv(CSV_SAVE_PATH)
        except FileNotFoundError as e:
            print(f"错误：{e}")
            print("请先将RUN_SIMULATION设为True运行一次，生成CSV文件后再加载！")
            exit(1)

    
    plot_repair_bandwidth_bar_chart(final_results)