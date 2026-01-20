import torch
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans
from multiprocessing import Pool
import os
import time
import leidenalg as la 
import igraph as ig
from collections import defaultdict, Counter, deque
import random
import multiprocessing as mp
from functools import partial
import shutil
import warnings
from sklearn.exceptions import ConvergenceWarning


# Hyperparameters
HIDDEN_DIM = 16
EPOCHS = 20
NUM_PROCESSES = min(mp.cpu_count() - 1, 6)
# NUM_PROCESSES = 1
LEIDEN_ITERATIONS = 20
TEMP_DIR = "temp_embeddings"
RANDOM_SEED = 42

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def load_data(edge_file_path, comm_file_path):
    """Load SNAP Community datasets"""
    with open(comm_file_path) as f:
        communties = [[int(i) for i in x.split()] for x in f]
    
    with open(edge_file_path) as f:
        edges = [[int(i) for i in e.split()] for e in f]
    
    edges = [[u, v] if u < v else [v, u] for u, v in edges if u != v]
    
    raw_nodes = {node for e in edges for node in e}
    mapping = {u: i for i, u in enumerate(sorted(raw_nodes))}
    
    edges = [[mapping[u], mapping[v]] for u, v in edges]
    communties = [[mapping[node] for node in com] for com in communties]
    
    num_node, num_edges, num_comm = len(raw_nodes), len(edges), len(communties)
    print(f"[{os.path.basename(edge_file_path).upper()}] #Nodes {num_node}, #Edges {num_edges}, #Communities {num_comm}")
    
    new_nodes = list(range(len(raw_nodes)))
    
    return num_node, num_edges, num_comm, new_nodes, edges, communties

def convert_comm_dict_to_comms_list(comm_dict):
    """Convert node->community dict to list of community lists"""
    comm_to_nodes = defaultdict(list)
    for node, comm_id in comm_dict.items():
        comm_to_nodes[comm_id].append(node)
    
    return [nodes for nodes in comm_to_nodes.values()]


def evaluate_with_correct_format(true_comms, comm_dict):
    """Evaluate with correct format"""
    print("Converting data structure...")
    
    pred_comms = convert_comm_dict_to_comms_list(comm_dict)
    
    print(f"True communities: {len(true_comms)}, Predicted communities: {len(pred_comms)}")
    
    true_sizes = [len(comm) for comm in true_comms]
    pred_sizes = [len(comm) for comm in pred_comms]
    
    print(f"True community sizes: min={min(true_sizes)}, max={max(true_sizes)}, avg={np.mean(true_sizes):.1f}")
    print(f"Predicted community sizes: min={min(pred_sizes)}, max={max(pred_sizes)}, avg={np.mean(pred_sizes):.1f}")
    
    try:
        from metrics import eval_scores_fast_optimized_fixed
        avg_precision, avg_recall, avg_f1, avg_jaccard = eval_scores_fast_optimized_fixed(pred_comms, true_comms, tmp_print=True)
        
        print(f"  Average Precision: {avg_precision:.4f}")
        print(f"  Average Recall: {avg_recall:.4f}")
        print(f"  Average F1 Score: {avg_f1:.4f}")
        print(f"  Average Jaccard: {avg_jaccard:.4f}")
        
        return avg_precision, avg_recall, avg_f1, avg_jaccard
    except ImportError:
        print("Warning: metrics module not found, using fallback evaluation")
        return 0.0, 0.0, 0.0, 0.0


def build_global_graph_from_original_optimized(edge_df, nodes):
    """修复后的优化版本"""
    print("  Building global graph from original edges (optimized)...")
    start_time = time.time()
    
    # 1. 快速处理边（使用numpy）
    print("    Processing edges...")
    
    # 转换为numpy数组
    if isinstance(edge_df, pd.DataFrame):
        edges_array = edge_df[['u', 'v']].values
    else:
        edges_array = edge_df
    
    # 排序边（无向图）
    edges_sorted = np.sort(edges_array, axis=1)
    
    # 使用字典统计边权重
    print("    Counting edge weights...")
    edge_counts = {}
    batch_size = 1000000
    
    for i in range(0, len(edges_sorted), batch_size):
        batch = edges_sorted[i:i+batch_size]
        for u, v in batch:
            key = (u, v)
            if key in edge_counts:
                edge_counts[key] += 1.0
            else:
                edge_counts[key] = 1.0
    
    print(f"    Unique edges: {len(edge_counts):,}")
    
    # 2. 构建边列表
    print("    Building edge list...")
    edge_list = [(u, v, w) for (u, v), w in edge_counts.items()]
    
    # 3. 构建图
    print("    Creating graph...")
    G = nx.Graph()
    
    # 添加节点
    if nodes is not None:
        G.add_nodes_from(nodes)
        print(f"    Added {len(nodes):,} nodes from input")
    else:
        # 从边中提取所有节点
        node_set = set()
        for (u, v), _ in edge_counts.items():
            node_set.add(u)
            node_set.add(v)
        G.add_nodes_from(node_set)
        print(f"    Added {len(node_set):,} nodes from edges")
    
    # 添加边（关键修复：使用add_weighted_edges_from）
    G.add_weighted_edges_from(edge_list)
    
    # 验证图构建
    print(f"    Graph validation:")
    print(f"      - Nodes: {G.number_of_nodes():,}")
    print(f"      - Edges: {G.number_of_edges():,}")
    print(f"      - Is connected: {nx.is_connected(G) if G.number_of_nodes() > 0 else 'N/A'}")
    
    # 检查一些随机节点的边
    if G.number_of_nodes() > 0:
        sample_nodes = list(G.nodes())[:min(5, G.number_of_nodes())]
        for node in sample_nodes:
            degree = G.degree(node)
            print(f"      - Node {node}: degree = {degree}")
    
    print(f"    Time elapsed: {time.time() - start_time:.2f}s")
    
    return G

def execute_Leiden(edge_file_path, comm_file_path, network_type):
    """Main unsupervised HIDC pipeline"""
    """Luyun"""
    start_total_time = time.time()
    print("="*50)
    print(f"Main process started (unsupervised version): {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)

    os.makedirs(TEMP_DIR, exist_ok=True)
    
    print("\n[1/6 Loading data]")
    num_node, num_edges, num_comm, all_nodes, edges, communties = load_data(edge_file_path, comm_file_path)
    edge_df = pd.DataFrame(edges, columns=['u', 'v'])

    global_G = build_global_graph_from_original_optimized(edge_df, all_nodes)

    block_ig=ig.Graph.from_networkx(global_G)

    node_list = sorted(global_G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}

    total_time0 = (time.time() - start_total_time) / 60
    print(f"\nTotal time: {total_time0:.2f} minutes")

    run_leiden_and_evaluate(block_ig, all_nodes, node_to_idx, communties)

    total_time = (time.time() - start_total_time) / 60
    print(f"\nleiden Total time: {total_time:.2f} minutes")

    start_total_time1 = time.time()
    run_louvain_and_evaluate(global_G, all_nodes, node_to_idx, communties, resolution=1.0)

    total_time1 = (time.time() - start_total_time1) / 60
    print(f"\nTotal time: {total_time0+total_time1:.2f} minutes")





    


def run_leiden_and_evaluate(block_ig, all_nodes, node_to_idx, true_comms):
    """
    运行Leiden算法并评估的完整函数
    """
    print("Running Leiden algorithm...")
    
    try:
        # 运行Leiden算法
        partition = la.find_partition(
            block_ig,
            la.ModularityVertexPartition,
            weights='weight',
            n_iterations=20,
            seed=42
        )
        
        # 获取成员关系（igraph顶点的社区标签）
        leiden_membership = np.array(partition.membership)
        
        print(f"Leiden found {len(set(leiden_membership))} communities")
        
        # 验证数据结构
        print(f"Leiden membership shape: {leiden_membership.shape}")
        print(f"Number of igraph vertices: {block_ig.vcount()}")
        print(f"Number of all_nodes: {len(all_nodes)}")
        print(f"Number of nodes in node_to_idx: {len(node_to_idx)}")
        
        # 创建一个从节点到预测社区的映射
        node_comm_mapping = {}
        for original_node_id, comm_id in zip(all_nodes, leiden_membership):
            node_comm_mapping[original_node_id] = comm_id
        
        # 转换为社区列表格式
        comms_by_id = {}
        for node, comm_id in node_comm_mapping.items():
            if comm_id not in comms_by_id:
                comms_by_id[comm_id] = []
            comms_by_id[comm_id].append(node)
        
        pred_comms = list(comms_by_id.values())
        
        print(f"Converted to {len(pred_comms)} communities")
        
        # 直接评估
        print("\nEvaluating results...")
        try:
            from metrics import eval_scores_fast_optimized_fixed
            avg_precision, avg_recall, avg_f1, avg_jaccard = eval_scores_fast_optimized_fixed(
                pred_comms, true_comms, tmp_print=True
            )
            
            print(f"  Average Precision: {avg_precision:.4f}")
            print(f"  Average Recall: {avg_recall:.4f}")
            print(f"  Average F1 Score: {avg_f1:.4f}")
            print(f"  Average Jaccard: {avg_jaccard:.4f}")
            
            return pred_comms, (avg_precision, avg_recall, avg_f1, avg_jaccard)
            
        except Exception as e:
            print(f"Evaluation error: {e}")
            return pred_comms, (0.0, 0.0, 0.0, 0.0)
            
    except Exception as e:
        print(f"Leiden algorithm error: {e}")
        return [], (0.0, 0.0, 0.0, 0.0)
    
import community as community_louvain  # python-louvain 库
import networkx as nx
import numpy as np

def run_louvain_and_evaluate(block_nx_graph, all_nodes, node_to_idx, true_comms, resolution=1.0):
    """
    运行Louvain算法并评估的完整函数
    """
    print("Running Louvain algorithm...")
    start_time = time.time()
    
    try:
        # 运行Louvain算法
        partition = community_louvain.best_partition(
            block_nx_graph,
            weight='weight',
            resolution=resolution,
            random_state=42
        )
        
        louvain_time = time.time() - start_time
        print(f"Louvain completed in {louvain_time:.2f}s")
        
        # 转换为社区字典格式
        comms_by_id = {}
        for node, comm_id in partition.items():
            if comm_id not in comms_by_id:
                comms_by_id[comm_id] = []
            comms_by_id[comm_id].append(node)
        
        pred_comms = list(comms_by_id.values())
        
        print(f"Louvain found {len(pred_comms)} communities")
        
        # 验证数据结构
        print(f"Number of networkx nodes: {block_nx_graph.number_of_nodes()}")
        print(f"Number of all_nodes: {len(all_nodes)}")
        print(f"Number of nodes in partition: {len(partition)}")
        
        # 统计社区大小
        community_sizes = [len(comm) for comm in pred_comms]
        if community_sizes:
            print(f"Community sizes: min={min(community_sizes)}, "
                  f"max={max(community_sizes)}, avg={np.mean(community_sizes):.1f}")
        
        # 计算模块度
        modularity = community_louvain.modularity(partition, block_nx_graph, weight='weight')
        print(f"Modularity: {modularity:.4f}")
        
        # 直接评估
        print("\nEvaluating results...")
        try:
            from metrics import eval_scores_fast_optimized_fixed
            avg_precision, avg_recall, avg_f1, avg_jaccard = eval_scores_fast_optimized_fixed(
                pred_comms, true_comms, tmp_print=True
            )
            
            print(f"  Average Precision: {avg_precision:.4f}")
            print(f"  Average Recall: {avg_recall:.4f}")
            print(f"  Average F1 Score: {avg_f1:.4f}")
            print(f"  Average Jaccard: {avg_jaccard:.4f}")
            
            # 返回额外信息
            metrics_info = {
                'precision': avg_precision,
                'recall': avg_recall,
                'f1': avg_f1,
                'jaccard': avg_jaccard,
                'modularity': modularity,
                'num_communities': len(pred_comms),
                'time': louvain_time
            }
            
            return pred_comms, metrics_info
            
        except Exception as e:
            print(f"Evaluation error: {e}")
            # 返回默认值
            metrics_info = {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'jaccard': 0.0,
                'modularity': modularity,
                'num_communities': len(pred_comms),
                'time': louvain_time
            }
            return pred_comms, metrics_info
            
    except Exception as e:
        print(f"Louvain algorithm error: {e}")
        return [], {}

def membership_list_to_communities(membership, nodes):
    """
    将每个节点的社区编号列表转换为社区列表。
    membership: 列表，第i个元素表示节点nodes[i]的社区编号。
    nodes: 节点列表，与membership顺序一致。
    返回: 社区列表，每个社区是一个节点列表。
    """
    communities_dict = {}
    for node, comm_id in zip(nodes, membership):
        communities_dict.setdefault(comm_id, []).append(node)
    return list(communities_dict.values())



DATASET_CONFIGS = {
    'facebook': {
        'edge_path': 'dataset/facebook-1.90.ungraph.txt',
        'community_path': 'dataset/facebook-1.90.cmty.txt',
        'description': 'Facebook social network',
        'network_type': 'social'
    },
    'amazon1': {
        'edge_path': 'dataset/amazon-1.90.ungraph.txt',
        'community_path': 'dataset/amazon-1.90.cmty.txt',
        'description': 'Amazon co-purchasing network',
        'network_type': 'co-purchase'
    },
    'lj1': {
        'edge_path': 'dataset/lj-1.90.ungraph.txt',
        'community_path': 'dataset/lj-1.90.cmty.txt',
        'description': 'LiveJournal social network',
        'network_type': 'social'
    },
    'dblp1': {
        'edge_path': 'dataset/dblp-1.90.ungraph.txt',
        'community_path': 'dataset/dblp-1.90.cmty.txt',
        'description': 'DBLP collaboration network',
        'network_type': 'collaboration'
    },
    'dblp2': {
        'edge_path': 'dataset/dblp.ungraph.txt',
        'community_path': 'dataset/dblp_communities.txt',
        'description': 'DBLP collaboration network',
        'network_type': 'collaboration'
    },
    'amazon2': {
        'edge_path': 'dataset/com-amazon.ungraph.txt',
        'community_path': 'dataset/com-amazon.all.dedup.cmty.txt',
        'description': 'Amazon co-purchasing network',
        'network_type': 'co-purchase'
    },
    'lj2': {
        'edge_path': 'dataset/lj.ungraph.txt',
        'community_path': 'dataset/lj.cmty.txt',
        'description': 'LiveJournal social network',
        'network_type': 'social'
    },

}

if __name__ == "__main__":
    dataset_name = "lj2"
    configds = DATASET_CONFIGS.get(dataset_name)
    
    EDGE_FILE_PATH = configds["edge_path"]
    COMMUNITY_FILE_PATH = configds["community_path"]
    network_type = configds["network_type"]
    print(f"CPU cores: {mp.cpu_count()}")
    
    if not os.path.exists(EDGE_FILE_PATH):
        raise FileNotFoundError(f"Edge file not found: {EDGE_FILE_PATH}")
    
    if not os.path.exists(COMMUNITY_FILE_PATH):
        print(f"Warning: Community file not found, running unsupervised community detection")
    
    execute_Leiden(EDGE_FILE_PATH, COMMUNITY_FILE_PATH, network_type)