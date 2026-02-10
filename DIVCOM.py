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

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# Hyperparameters
HIDDEN_DIM = 16
EPOCHS = 20
NUM_PROCESSES = min(mp.cpu_count() - 1, 6)
# NUM_PROCESSES = 1
LEIDEN_ITERATIONS = 20
TEMP_DIR = "temp_embeddings"
RANDOM_SEED = 42

# Fix random seeds
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def enforce_array_type(arr, dtype=np.float32, shape=None):
    """Ensure array has correct type and shape"""
    try:
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr, dtype=dtype)
        if arr.dtype != dtype:
            arr = arr.astype(dtype)
        if shape is not None and arr.shape != shape:
            arr = np.zeros(shape, dtype=dtype)
        if np.issubdtype(dtype, np.floating):
            arr[np.isnan(arr) | np.isinf(arr)] = 0.0
        return arr
    except:
        return np.zeros(shape, dtype=dtype) if shape else np.array([], dtype=dtype)
    


def split_data_by_connectivity(edge_df, all_nodes, node_degree_dict, block_num):
    """Split data based on graph connectivity"""
    block_size = len(all_nodes) // block_num
    
    visited = set()
    adjacency = defaultdict(list)
    for u, v in edge_df[['u', 'v']].values:
        adjacency[u].append(v)
        adjacency[v].append(u)
    
    #print(f"Adjacency nodes: {len(set(adjacency.keys()))}")
    #print(f"Common nodes: {len(adjacency.keys() & all_nodes)}")
    
    sorted_nodes = sorted(all_nodes, key=lambda x: node_degree_dict.get(x, 0), reverse=True)
    #print(f"Sorted nodes: {len(sorted_nodes)}")
    
    blocks = []
    current_block = []
    
    for node in sorted_nodes:
        if node in visited:
            continue
        
        queue = deque([node])
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            
            visited.add(current)
            current_block.append(current)
            if len(current_block) >= block_size:
                blocks.append(current_block)
                #print(f"Current block size: {len(current_block)}")
                current_block = []
            
            neighbors = adjacency.get(current, [])
            neighbors_sorted = sorted(neighbors, 
                                    key=lambda x: node_degree_dict.get(x, 0), 
                                    reverse=True)
            for neighbor in neighbors_sorted:
                queue.append(neighbor)
    
    if current_block:
        #print(f"Final block: {len(current_block)}")
        if len(current_block) <= block_size and len(blocks) > 0:
            if len(blocks[-1]) + len(current_block) <= block_size * 1.5:
                blocks[-1].extend(current_block)
            else:
                blocks.append(current_block)
        else:
            blocks.append(current_block)
    
    unvisited_nodes = [node for node in all_nodes if node not in visited]
    if unvisited_nodes:
        for i in range(0, len(unvisited_nodes), block_size):
            small_block = unvisited_nodes[i:i+block_size]
            if small_block:
                blocks.append(small_block)
    
    #print(f"Generated {len(blocks)} blocks")
    
    new_all_nodes = []
    for i, block in enumerate(blocks):
        #print(f"{i}. Block size: {len(block)}")
        new_all_nodes.extend(block)
    
    new_all_nodes = list(set(new_all_nodes))
    
    return blocks, new_all_nodes


def spp_fastest(edge_df, all_nodes, block_num):
    """
    最快SPP版本 - 只保留核心逻辑
    """
    # 1. 邻接表
    adj = defaultdict(set)
    for u, v in edge_df[['u', 'v']].values:
        adj[u].add(v)
        adj[v].add(u)
    
    # 2. 计算度数（替代聚类系数）
    degrees = {node: len(adj.get(node, set())) for node in all_nodes}
    
    # 3. 选择高度数节点作为种子
    block_size = len(all_nodes) // block_num
    seeds = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:block_num]
    seeds = [node for node, _ in seeds]
    
    # 4. 从每个种子BFS扩展
    visited = set()
    blocks = []
    node_sequence = []
    
    for seed in seeds:
        if seed in visited:
            continue
            
        community = []
        stack = [seed]
        
        while stack and len(community) < block_size:
            node = stack.pop()
            if node in visited:
                continue
                
            visited.add(node)
            community.append(node)
            node_sequence.append(node)
            
            # 按度数添加邻居
            neighbors = sorted(adj.get(node, set()), 
                             key=lambda x: degrees.get(x, 0), 
                             reverse=True)
            for nb in neighbors:
                if nb not in visited and nb not in stack:
                    stack.append(nb)
        
        if community:
            blocks.append(community)
    
    # 5. 剩余节点直接分配
    remaining = [n for n in all_nodes if n not in visited]
    if remaining:
        for node in remaining:
            # 找连接最多的块
            best_idx = 0
            best_conn = 0
            for i, block in enumerate(blocks):
                conn = len(adj[node] & set(block))
                if conn > best_conn:
                    best_conn = conn
                    best_idx = i
            
            if best_conn > 0:
                blocks[best_idx].append(node)
            else:
                # 随机分配
                blocks[0].append(node)
            
            node_sequence.append(node)
    
    print(f"SPP完成: {len(blocks)} blocks, {len(node_sequence)} nodes")
    return blocks, node_sequence

def split_data_by_node(all_nodes, node_degree_dict, K):
    """根据节点ID进行顺序分块
    
    Args:
        all_nodes: 所有节点列表
        node_degree_dict: 节点度字典 {node: degree}
        K: 目标分块数
    
    Returns:
        blocks: 分块后的节点列表
        new_all_nodes: 去重后的所有节点
    """
    import time
    start_time = time.time()
    
    # 1. 按节点ID排序（升序或降序均可，这里使用升序）
    block_size = len(all_nodes) // K
    sorted_nodes = sorted(all_nodes)  # 按ID升序排列
    
    total_nodes = len(sorted_nodes)
    
    # 2. 按顺序分配节点到块
    blocks = []
    current_block = []
    current_size = 0
    
    for node in sorted_nodes:
        current_block.append(node)
        current_size += 1
        
        # 如果块达到目标大小，创建新块
        if current_size >= block_size:
            blocks.append(current_block.copy())
            current_block = []
            current_size = 0
    
    # 3. 处理最后一个块
    if current_block:
        # 检查最后一个块是否太小
        if len(current_block) < block_size and len(blocks) > 0:
            # 合并到最后一个大块中
            if len(blocks[-1]) + len(current_block) <= block_size * 1.5:
                blocks[-1].extend(current_block)
            else:
                blocks.append(current_block)
        else:
            blocks.append(current_block)
    
    # 4. 验证和去重
    final_blocks = []
    for block in blocks:
        unique_block = list(set(block))
        if unique_block:  # 只保留非空块
            final_blocks.append(unique_block)
    
    # 5. 生成去重后的所有节点列表
    new_all_nodes = []
    for block in final_blocks:
        new_all_nodes.extend(block)
    new_all_nodes = list(set(new_all_nodes))
    
    # 6. 输出统计信息
    elapsed_time = time.time() - start_time
    #print(f"数据分块完成：共{len(final_blocks)}个块，总节点{total_nodes}，耗时{elapsed_time:.4f}秒")
    
    # 打印每个块的统计信息
    for i, block in enumerate(final_blocks):
        block_nodes = len(block)
        # 计算块的ID范围
        if block_nodes > 0:
            min_id = min(block)
            max_id = max(block)
            avg_degree = sum(node_degree_dict.get(node, 0) for node in block) / max(block_nodes, 1)
            #print(f"  块{i}: {block_nodes}个节点，ID范围[{min_id}, {max_id}]，平均度{avg_degree:.2f}")
        else:
            print(f"  块{i}: 0个节点")
    
    return final_blocks, new_all_nodes


def calc_block_edge_weight_no_queue(edge_df, block_nodes, block_id, cn_base_alpha, 
                                   node_embed_dict=None, embed_weight_alpha=0.3):
    """Calculate edge weights for a block with optional embedding-based adjustment"""
    try:
        start_time = time.time()
        block_node_set = set(block_nodes)
        
        block_mask = edge_df['u'].isin(block_node_set) & edge_df['v'].isin(block_node_set)
        block_edge = edge_df[block_mask].copy()
        block_edge[['u_sorted', 'v_sorted']] = np.sort(block_edge[['u', 'v']].values, axis=1)
        
        
        edge_count = len(block_edge)
        
        if edge_count == 0:
            #print(f"Warning: Block {block_id} has no valid edges")
            return (block_id, pd.DataFrame(columns=['u', 'v', 'weight']))
        
        block_edge[['u_sorted', 'v_sorted']] = np.sort(block_edge[['u', 'v']].values, axis=1)
        edge_counts = block_edge.groupby(['u_sorted', 'v_sorted']).size().reset_index(name='count')
        
        neighbor_dict = defaultdict(list)
        for _, row in edge_counts.iterrows():
            u, v = row['u_sorted'], row['v_sorted']
            neighbor_dict[u].append(v)
            neighbor_dict[v].append(u)
        for u in neighbor_dict:
            neighbor_dict[u].sort()
        
        def count_common(u, v):
            neighbors_u = neighbor_dict.get(u, [])
            neighbors_v = neighbor_dict.get(v, [])
            i = j = common = 0
            len_u, len_v = len(neighbors_u), len(neighbors_v)
            while i < len_u and j < len_v:
                if neighbors_u[i] == neighbors_v[j]:
                    common += 1
                    i += 1
                    j += 1
                elif neighbors_u[i] < neighbors_v[j]:
                    i += 1
                else:
                    j += 1
            return common
        
        edge_counts['common'] = edge_counts.apply(
            lambda row: count_common(row['u_sorted'], row['v_sorted']), axis=1
        )
        
        # 基础权重：边计数 + 共同邻居调整
        edge_counts['weight'] = edge_counts['count'] + cn_base_alpha * edge_counts['common']
        
      
        
        result_edge = block_edge[['u', 'v']].drop_duplicates()
        result_edge = result_edge.merge(
            edge_counts[['u_sorted', 'v_sorted', 'weight']],
            left_on=['u', 'v'],
            right_on=['u_sorted', 'v_sorted'],
            how='left'
        ).fillna(1)[['u', 'v', 'weight']]
        
        # 确保权重为正
        result_edge['weight'] = result_edge['weight'].clip(lower=0.001)
        
        #print(f"Process {os.getpid()}: Block {block_id} completed, {len(result_edge)} edges, "
         #     f"avg_weight={result_edge['weight'].mean():.3f}, time: {time.time()-start_time:.2f}s")
        return (block_id, result_edge)
    
    except Exception as e:
        #print(f"Process {os.getpid()}: Block {block_id} edge weight calculation failed! Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return (block_id, pd.DataFrame(columns=['u', 'v', 'weight']))
    



def compute_embeddings(edges, nodes):
    # 这里使用我们之前设计的嵌入函数，返回一个字典，节点->嵌入向量
    # 假设我们已经实现了 minimal_community_aware_embedding 函数
    _, embeddings = minimal_community_aware_embedding(edges, nodes, block_id=None)
    return embeddings

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))




import numpy as np
import pandas as pd
import time

def simplest_structural_embedding(edges, nodes, block_id):
    """Simple structural embedding based on graph structure"""
    G = nx.Graph()
    for idx, row in edges.iterrows():
        u = row['u']
        v = row['v']
        G.add_edge(u, v)
    
    # 确保所有节点都在图中（可能有孤立节点）
    for node in nodes:
        if node not in G:
            G.add_node(node)
    
    # 计算特征
    embeddings = {}
    for node in nodes:
        # 度
        deg = G.degree(node)
        # 局部聚类系数
        clustering = nx.clustering(G, node)
        # 邻居平均度
        neighbors = list(G.neighbors(node))
        if len(neighbors) > 0:
            avg_neighbor_deg = sum(G.degree(n) for n in neighbors) / len(neighbors)
        else:
            avg_neighbor_deg = 0
        
        # 构建3维向量
        vec = np.array([deg, clustering, avg_neighbor_deg], dtype=np.float32)
        # 归一化
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        embeddings[node] = vec
    
    return (block_id, embeddings)
    

def minimal_community_aware_embedding(edges, nodes, block_id):
    """
    优化的社区感知嵌入，确保有变化
    """
    #print(f"计算嵌入，节点数: {len(nodes)}, 边数: {len(edges)}")
    
    # 方法1: 基于度的特征 + 社区边界特征
    degree = {}
    neighbor_counts = {}
    
    # 计算度和邻居
    for _, row in edges.iterrows():
        u, v = row['u'], row['v']
        
        # 更新度
        degree[u] = degree.get(u, 0) + 1
        degree[v] = degree.get(v, 0) + 1
        
        # 更新邻居集合
        if u not in neighbor_counts:
            neighbor_counts[u] = set()
        if v not in neighbor_counts:
            neighbor_counts[v] = set()
        
        neighbor_counts[u].add(v)
        neighbor_counts[v].add(u)
    
    n_nodes = len(nodes)
    embeddings = {}
    
    # 计算一些全局统计信息
    if degree:
        max_degree = max(degree.values())
        avg_degree = sum(degree.values()) / n_nodes if n_nodes > 0 else 0
    else:
        max_degree = 1
        avg_degree = 0
    
    # 为每个节点生成嵌入
    for i, node in enumerate(nodes):
        deg = degree.get(node, 0)
        
        # 计算邻居特征
        if node in neighbor_counts:
            neighbors = neighbor_counts[node]
            
            # 邻居的度统计
            neighbor_degrees = [degree.get(n, 0) for n in neighbors]
            avg_neighbor_deg = np.mean(neighbor_degrees) if neighbor_degrees else 0
            
            # 社区边界指标
            # 边界节点: 连接不同社区，其邻居之间的连接较少
            neighbor_connections = 0
            for n1 in neighbors:
                for n2 in neighbors:
                    if n1 != n2 and n2 in neighbor_counts.get(n1, set()):
                        neighbor_connections += 1
            
            possible_connections = len(neighbors) * (len(neighbors) - 1) if len(neighbors) > 1 else 1
            clustering = neighbor_connections / possible_connections if possible_connections > 0 else 0
            
            # 边界强度: 节点作为社区边界的可能性
            boundary_strength = deg * (1 - clustering) if clustering < 1 else 0
        else:
            avg_neighbor_deg = 0
            clustering = 0
            boundary_strength = 0
        
        # 创建8维嵌入向量
        vec = np.zeros(8, dtype=np.float32)
        
        # 特征1-2: 度特征
        vec[0] = deg  # 原始度
        vec[1] = deg / max_degree if max_degree > 0 else 0  # 归一化度
        
        # 特征3: 相对于平均度的位置
        vec[2] = 1 if deg > avg_degree else -1 if deg < avg_degree else 0
        
        # 特征4: 社区边界指标
        vec[3] = boundary_strength / max_degree if max_degree > 0 else 0
        
        # 特征5: 局部聚类系数
        vec[4] = clustering
        
        # 特征6: 邻居平均度比率
        vec[5] = avg_neighbor_deg / max_degree if max_degree > 0 else 0
        
        # 特征7-8: 基于节点ID的确定性随机特征
        node_hash = hash(str(node)) % 10000
        vec[6] = np.sin(node_hash / 1000.0)
        vec[7] = np.cos(node_hash / 777.0)
        
        # 确保有变化：添加时间相关的微小变化
        time_seed = int(time.time() * 1000) % 1000
        vec = vec * (1.0 + 0.001 * (time_seed % 10))
        
        # 归一化
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        
        embeddings[node] = vec
        
        # 每处理1000个节点打印一次进度
        if (i + 1) % 1000 == 0:
            print(f"已处理 {i+1}/{n_nodes} 个节点")
    
    #print(f"嵌入计算完成，生成了 {len(embeddings)} 个嵌入向量")
    return (block_id, embeddings)



def generate_block_community(args):
    """Generate communities for a block using Leiden algorithm"""
    try:
        weighted_edge_block,  block_nodes, block_id = args
        start_time = time.time()
        
        block_G = nx.Graph()
        if not weighted_edge_block.empty:
            valid_edges = weighted_edge_block[['u', 'v', 'weight']].values
            if len(valid_edges) > 0:
                block_G.add_weighted_edges_from(valid_edges)
        for node in block_nodes:
            if node not in block_G:
                block_G.add_node(node)
        
        if not block_G.nodes():
            return (block_id, {})
        
        num_nodes = len(block_G.nodes())
        num_edges = len(block_G.edges())
        # max_possible_edges = num_nodes * (num_nodes - 1) / 2 if num_nodes > 1 else 1
        # density = num_edges / max_possible_edges
        
        def run_leiden():
            # try:
            node_list = sorted(block_G.nodes())
            # node_list = list(block_G.nodes())
            node_to_idx = {node: idx for idx, node in enumerate(node_list)}
            # idx_to_node = {idx: node for idx, node in enumerate(node_list)}
            
            block_ig = ig.Graph(directed=False)
            block_ig.add_vertices(len(node_list))
            
            edges, edge_weights = [], []
            for u, v, data in block_G.edges(data=True):
                edges.append((node_to_idx[u], node_to_idx[v]))
                edge_weights.append(float(data.get('weight', 1.0)))
            if edges:
                block_ig.add_edges(edges)
                block_ig.es['weight'] = edge_weights
            

            
            partition = la.find_partition(
                block_ig,
                la.ModularityVertexPartition,
                weights='weight',
                n_iterations=20,
                seed=42
            )
            
            leiden_comm = np.array(partition.membership)
            return [leiden_comm[node_to_idx[node]] if node in node_to_idx else 0 
                    for node in block_nodes]
            
            # except Exception as e:
            #     print(f"Block {block_id} Leiden failed")
            #     # return run_kmeans()
        

        block_fine_comm = run_leiden()
        
        global_comm_prefix = block_id * 1000000
        block_comm_dict = {node: global_comm_prefix + comm 
                         for node, comm in zip(block_nodes, block_fine_comm)}
        
        #print(f"Process {os.getpid()}: Block {block_id} generated {len(set(block_fine_comm))} communities, time: {time.time()-start_time:.2f}s")
        return (block_id, block_comm_dict)
    
    except Exception as e:
        #print(f"Process {os.getpid()}: Block {block_id} community generation failed! Error: {str(e)}")
        block_comm_dict = {node: block_id * 1000000 + i for i, node in enumerate(block_nodes)}
        return (block_id, block_comm_dict)
    

def auto_kmeans_elbow(block_z, max_k=15):
    """Determine optimal k using elbow method"""
    inertias = []
    k_range = range(1, min(max_k, len(block_z)//2) + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=3)
        kmeans.fit(block_z)
        inertias.append(kmeans.inertia_)
    
    improvements = [inertias[i-1] - inertias[i] for i in range(1, len(inertias))]
    if improvements:
        elbow_idx = np.argmax(improvements) + 1
        optimal_k = k_range[elbow_idx]
    else:
        optimal_k = 2
    
    return optimal_k


def build_global_graph_fast(weighted_edge_df, all_nodes):
    """Build global graph from weighted edges"""
    #print("  Building global graph...")
    start_time = time.time()
    
    global_G = nx.Graph()
    global_G.add_nodes_from(all_nodes)
    all_nodes_set = set(all_nodes)
    
    for row in weighted_edge_df.itertuples():
        u, v, weight = row.u, row.v, row.weight
        if u in all_nodes_set and v in all_nodes_set:
            global_G.add_edge(u, v, weight=float(weight))
    
    #print(f"  Global graph: {global_G.number_of_nodes()} nodes, {global_G.number_of_edges()} edges")
    #print(f"  Build time: {time.time()-start_time:.2f}s")
    return global_G

def process_block(args, edge_df, cn_base_alpha):
    """Process block for edge weight calculation"""
    block_id, block_nodes = args
    return calc_block_edge_weight_no_queue(edge_df, block_nodes, block_id, cn_base_alpha)

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
    #print(f"[{os.path.basename(edge_file_path).upper()}] #Nodes {num_node}, #Edges {num_edges}, #Communities {num_comm}")
    
    new_nodes = list(range(len(raw_nodes)))
    
    return num_node, num_edges, num_comm, new_nodes, edges, communties

# def convert_comm_dict_to_comms_list(comm_dict):
#     """Convert node->community dict to list of community lists"""
#     comm_to_nodes = defaultdict(list)
#     for node, comm_id in comm_dict.items():
#         comm_to_nodes[comm_id].append(node)
    
#     return [nodes for nodes in comm_to_nodes.values()]

def convert_comm_dict_to_comms_list(comm_dict):
    """
    将社区字典转换为社区列表格式
    
    参数:
    comm_dict: 字典，键为节点，值为社区ID或社区ID列表
    
    返回:
    list: 社区列表，每个社区是节点列表
    """
    from collections import defaultdict
    
    # 使用字典存储社区到节点的映射
    comm_to_nodes = defaultdict(list)
    
    for node, comm_value in comm_dict.items():
        if isinstance(comm_value, list):
            # 重叠社区情况：节点属于多个社区
            for comm_id in comm_value:
                # 确保社区ID是可哈希的类型
                if isinstance(comm_id, (int, str, tuple)):
                    comm_to_nodes[comm_id].append(node)
                else:
                    # 如果是列表等不可哈希类型，转换为元组或字符串
                    try:
                        hash_key = tuple(comm_id) if isinstance(comm_id, list) else str(comm_id)
                        comm_to_nodes[hash_key].append(node)
                    except:
                        # 最后的备选方案
                        comm_to_nodes[str(comm_id)].append(node)
        else:
            # 非重叠社区情况：节点属于单个社区
            comm_id = comm_value
            if isinstance(comm_id, (int, str, tuple)):
                comm_to_nodes[comm_id].append(node)
            else:
                # 处理不可哈希类型
                try:
                    hash_key = tuple(comm_id) if isinstance(comm_id, list) else str(comm_id)
                    comm_to_nodes[hash_key].append(node)
                except:
                    comm_to_nodes[str(comm_id)].append(node)
    
    # 转换为社区列表
    communities = list(comm_to_nodes.values())
    
    return communities

def evaluate_with_correct_format(true_comms, comm_dict):
    """Evaluate with correct format"""
    #print("Converting data structure...")
    
    # pred_comms = convert_comm_dict_to_comms_list(comm_dict)

    pred_comms = convert_comm_dict_to_comms_list(comm_dict)
    
    #print(f"True communities: {len(true_comms)}, Predicted communities: {len(pred_comms)}")
    
    true_sizes = [len(comm) for comm in true_comms]
    pred_sizes = [len(comm) for comm in pred_comms]
    
    #print(f"True community sizes: min={min(true_sizes)}, max={max(true_sizes)}, avg={np.mean(true_sizes):.1f}")
    #print(f"Predicted community sizes: min={min(pred_sizes)}, max={max(pred_sizes)}, avg={np.mean(pred_sizes):.1f}")
    
    try:
        from metrics import eval_scores_fast_optimized_fixed
        avg_precision, avg_recall, avg_f1, avg_jaccard = eval_scores_fast_optimized_fixed(pred_comms, true_comms, tmp_print=True)
        
        print(f"  Average Precision: {avg_precision:.4f}")
        print(f"  Average Recall: {avg_recall:.4f}")
        print(f"  Average F1 Score: {avg_f1:.4f}")
        print(f"  Average Jaccard: {avg_jaccard:.4f}")
        
        return avg_precision, avg_recall, avg_f1, avg_jaccard
    except ImportError:
        #print("Warning: metrics module not found, using fallback evaluation")
        return 0.0, 0.0, 0.0, 0.0



def build_global_graph_from_original_optimized(edge_df, nodes):
    """修复后的优化版本"""
    #print("  Building global graph from original edges (optimized)...")
    start_time = time.time()
    
    # 1. 快速处理边（使用numpy）
    #print("    Processing edges...")
    
    # 转换为numpy数组
    if isinstance(edge_df, pd.DataFrame):
        edges_array = edge_df[['u', 'v']].values
    else:
        edges_array = edge_df
    
    # 排序边（无向图）
    edges_sorted = np.sort(edges_array, axis=1)
    
    # 使用字典统计边权重
    #print("    Counting edge weights...")
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
    
    #print(f"    Unique edges: {len(edge_counts):,}")
    
    # 2. 构建边列表
    #print("    Building edge list...")
    edge_list = [(u, v, w) for (u, v), w in edge_counts.items()]
    
    # 3. 构建图
    #print("    Creating graph...")
    G = nx.Graph()
    
    # 添加节点
    if nodes is not None:
        G.add_nodes_from(nodes)
        #print(f"    Added {len(nodes):,} nodes from input")
    else:
        # 从边中提取所有节点
        node_set = set()
        for (u, v), _ in edge_counts.items():
            node_set.add(u)
            node_set.add(v)
        G.add_nodes_from(node_set)
        #print(f"    Added {len(node_set):,} nodes from edges")
    
    # 添加边（关键修复：使用add_weighted_edges_from）
    G.add_weighted_edges_from(edge_list)
    
    # 验证图构建
    #print(f"    Graph validation:")
    #print(f"      - Nodes: {G.number_of_nodes():,}")
    #print(f"      - Edges: {G.number_of_edges():,}")
    #print(f"      - Is connected: {nx.is_connected(G) if G.number_of_nodes() > 0 else 'N/A'}")
    
    # 检查一些随机节点的边
    if G.number_of_nodes() > 0:
        sample_nodes = list(G.nodes())[:min(5, G.number_of_nodes())]
        for node in sample_nodes:
            degree = G.degree(node)
            #print(f"      - Node {node}: degree = {degree}")
    
    #print(f"    Time elapsed: {time.time() - start_time:.2f}s")
    
    return G

def merge_small_communities_fast(partition_dict, graph, min_size=3):
    """Fast merging of small communities"""
    from collections import defaultdict
    
    communities = partition_dict.copy()
    
    small_comms = {}
    large_comms = {}
    node_to_comm = {}
    
    for cid, nodes in communities.items():
        node_list = list(nodes)
        if len(node_list) < min_size:
            small_comms[cid] = node_list
        else:
            large_comms[cid] = node_list
        for node in node_list:
            node_to_comm[node] = cid
    
    if not small_comms:
        return communities
    
    #print(f"Small communities to merge: {len(small_comms)}")
    
    comm_connections = defaultdict(lambda: defaultdict(int))
    for u, v in graph.edges():
        comm_u = node_to_comm[u]
        comm_v = node_to_comm[v]
        
        if comm_u != comm_v:
            comm_connections[comm_u][comm_v] += 1
            comm_connections[comm_v][comm_u] += 1
    
    merged_result = {cid: set(nodes) for cid, nodes in large_comms.items()}
    comm_size = {cid: len(nodes) for cid, nodes in merged_result.items()}
    
    for small_cid, small_nodes in small_comms.items():
        connections = comm_connections.get(small_cid, {})
        
        candidate_large = {}
        for neighbor_comm, weight in connections.items():
            if neighbor_comm in merged_result:
                candidate_large[neighbor_comm] = weight
        
        if candidate_large:
            best_comm = max(
                candidate_large.items(),
                key=lambda x: (x[1], -comm_size.get(x[0], 0))
            )[0]
        else:
            if comm_size:
                best_comm = min(comm_size.items(), key=lambda x: x[1])[0]
            else:
                best_comm = max(merged_result.keys(), default=-1) + 1
                merged_result[best_comm] = set()
                comm_size[best_comm] = 0
        
        if best_comm not in merged_result:
            merged_result[best_comm] = set()
            comm_size[best_comm] = 0
        
        merged_result[best_comm].update(small_nodes)
        comm_size[best_comm] += len(small_nodes)
    
    final_result = {}
    for new_id, (old_id, nodes) in enumerate(merged_result.items()):
        if nodes:
            final_result[new_id] = list(nodes)
    
    sizes = [len(nodes) for nodes in final_result.values()]
    remaining_small = sum(1 for size in sizes if size < min_size)
    
    #print(f"Merged communities: {len(final_result)}")
    #print(f"Remaining small communities: {remaining_small}")
    if sizes:
        print(f"Community sizes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.2f}")
    
    original_nodes = sum(len(nodes) for nodes in communities.values())
    result_nodes = sum(len(nodes) for nodes in final_result.values())
    assert original_nodes == result_nodes, f"Node count mismatch: {original_nodes} != {result_nodes}"
    
    return final_result

# def optimize_community_structure(node_to_community_dict, global_G, min_size=3):
#     """Optimize community structure"""
#     print("Optimizing community structure...")
    
#     community_to_nodes = defaultdict(list)
#     for node, comm_id in node_to_community_dict.items():
#         community_to_nodes[comm_id].append(node)
    
#     #print(f"Converted communities: {len(community_to_nodes)}")
    
#     comm_sizes = [len(nodes) for nodes in community_to_nodes.values()]
#     #print(f"Community size stats: min={min(comm_sizes)}, max={max(comm_sizes)}, avg={np.mean(comm_sizes):.2f}")
    
#     merged_community_to_nodes = merge_small_communities_fast(community_to_nodes, global_G, min_size)
    
#     #print(f"Merged communities: {len(merged_community_to_nodes)}")
    
#     final_node_to_community = {}
#     for comm_id, nodes in merged_community_to_nodes.items():
#         for node in nodes:
#             final_node_to_community[node] = comm_id
    
#     return final_node_to_community


# def optimize_community_structure(comm_dict, G, min_comm_size, overlap_threshold=1):
#     """Optimize community structure by merging small communities and cleaning up
    
#     Args:
#         comm_dict: Community assignment dictionary
#         G: NetworkX graph
#         min_comm_size: Minimum community size
#         overlap_threshold: Overlap threshold used (for proper handling)
#     """
#     from collections import defaultdict
    
#     # Normalize community dictionary to always use list representation internally
#     normalized_dict = {}
#     for node, comm in comm_dict.items():
#         if isinstance(comm, list):
#             normalized_dict[node] = comm[:overlap_threshold] if overlap_threshold > 1 else [comm[0] if comm else -1]
#         else:
#             normalized_dict[node] = [comm]
    
#     # Create community to nodes mapping
#     comm_to_nodes = defaultdict(list)
#     for node, comm_list in normalized_dict.items():
#         for comm_id in comm_list:
#             if comm_id != -1:  # Skip invalid communities
#                 comm_to_nodes[comm_id].append(node)
    
#     # Get community sizes
#     comm_sizes = {comm_id: len(nodes) for comm_id, nodes in comm_to_nodes.items()}
    
#     # Find small communities
#     small_comms = [comm_id for comm_id, size in comm_sizes.items() if size < min_comm_size]
    
#     if len(small_comms) < 2:
#         # Return in original format
#         if overlap_threshold == 1:
#             return {node: comm_list[0] if comm_list else -1 
#                     for node, comm_list in normalized_dict.items()}
#         else:
#             return normalized_dict
    
#     # Merge small communities
#     small_comms.sort(key=lambda x: comm_sizes[x])
    
#     # Create a mapping for community merging
#     merge_map = {}
#     current_comm = small_comms[0]
    
#     for i in range(1, len(small_comms)):
#         merge_map[small_comms[i]] = current_comm
    
#     # Apply merging to normalized dictionary
#     merged_dict = {}
#     for node, comm_list in normalized_dict.items():
#         new_comm_list = []
#         for comm_id in comm_list:
#             if comm_id in merge_map:
#                 new_comm_id = merge_map[comm_id]
#                 if new_comm_id not in new_comm_list:
#                     new_comm_list.append(new_comm_id)
#             else:
#                 if comm_id not in new_comm_list:
#                     new_comm_list.append(comm_id)
        
#         # Ensure we don't exceed overlap threshold
#         if overlap_threshold > 1:
#             new_comm_list = new_comm_list[:overlap_threshold]
#         merged_dict[node] = new_comm_list
    
#     # Return in appropriate format
#     if overlap_threshold == 1:
#         return {node: comm_list[0] if comm_list else -1 
#                 for node, comm_list in merged_dict.items()}
#     else:
#         return merged_dict
# def optimize_community_structure(comm_dict, G, min_comm_size, overlap_threshold=1):
#     """
#     Optimize community structure by merging small communities into well-connected larger ones.
    
#     Args:
#         comm_dict (dict): Community assignment. Values can be int or list of ints.
#         G (networkx.Graph): The graph (used to assess connectivity between communities).
#         min_comm_size (int): Communities smaller than this will be merged.
#         overlap_threshold (int): Max number of communities per node (1 = non-overlapping).
    
#     Returns:
#         dict: Optimized community assignment in the same format as input.
#     """
#     from collections import defaultdict

#     # --- Step 1: Normalize input to list-of-communities format ---
#     assign = {}
#     for node, comms in comm_dict.items():
#         if isinstance(comms, list):
#             clean = [c for c in comms if c != -1]
#             if overlap_threshold == 1:
#                 assign[node] = [clean[0]] if clean else [-1]
#             else:
#                 assign[node] = clean[:overlap_threshold]
#         else:
#             assign[node] = [comms] if comms != -1 else [-1]

#     # --- Step 2: Build community -> nodes mapping and compute sizes ---
#     comm_to_nodes = defaultdict(list)
#     for node, comm_list in assign.items():
#         for cid in comm_list:
#             if cid != -1:
#                 comm_to_nodes[cid].append(node)

#     comm_sizes = {cid: len(nodes) for cid, nodes in comm_to_nodes.items()}
#     small_comms = {cid for cid, size in comm_sizes.items() if size < min_comm_size}
    
#     if not small_comms:
#         # No small communities → return as-is
#         return {n: cs[0] if overlap_threshold == 1 and cs else (-1 if overlap_threshold == 1 else [])
#                 for n, cs in assign.items()}

#     large_comms = set(comm_sizes.keys()) - small_comms

#     # --- Step 3: Determine merge target for each small community ---
#     merge_map = {}
#     for small_cid in small_comms:
#         # Count connections from this small community to other communities
#         neighbor_comm_count = defaultdict(int)
#         for node in comm_to_nodes[small_cid]:
#             for nb in G.neighbors(node):
#                 for nb_cid in assign.get(nb, []):
#                     if nb_cid != -1 and nb_cid != small_cid:
#                         neighbor_comm_count[nb_cid] += 1

#         # Prefer merging into a large community with strongest connection
#         best_target = small_cid  # default: keep itself (fallback)
#         best_score = -1

#         # Check large communities first
#         for candidate in large_comms:
#             if neighbor_comm_count[candidate] > best_score:
#                 best_score = neighbor_comm_count[candidate]
#                 best_target = candidate

#         # If no good large target, consider other small communities
#         if best_score <= 0:
#             for candidate in small_comms:
#                 if candidate != small_cid and neighbor_comm_count[candidate] > best_score:
#                     best_score = neighbor_comm_count[candidate]
#                     best_target = candidate

#         # If still no connection, merge into the largest small community (including itself)
#         if best_score <= 0:
#             best_target = max(small_comms, key=lambda c: comm_sizes[c])

#         merge_map[small_cid] = best_target

#     # --- Step 4: Apply merging to all nodes ---
#     new_assign = {}
#     for node, comm_list in assign.items():
#         new_list = []
#         for cid in comm_list:
#             target = merge_map.get(cid, cid)  # if not small, keep original
#             if target != -1 and target not in new_list:
#                 new_list.append(target)
        
#         # Enforce overlap limit
#         if overlap_threshold > 1:
#             new_list = new_list[:overlap_threshold]
#         elif overlap_threshold == 1:
#             new_list = [new_list[0]] if new_list else [-1]
        
#         new_assign[node] = new_list

#     # --- Step 5: Format output to match input style ---
#     if overlap_threshold == 1:
#         return {node: comm_list[0] if comm_list and comm_list[0] != -1 else -1
#                 for node, comm_list in new_assign.items()}
#     else:
#         return {node: [c for c in comm_list if c != -1] 
#                 for node, comm_list in new_assign.items()}

# def optimize_community_structure(comm_dict, G, min_comm_size, overlap_threshold=1):
#     """
#     High-performance version for large graphs (300k+ nodes).
#     Optimized to avoid nested loops and redundant lookups.
#     """
#     from collections import defaultdict

#     # --- Step 1: Normalize assignment ---
#     assign = {}
#     for node, comms in comm_dict.items():
#         if isinstance(comms, list):
#             clean = [c for c in comms if c != -1]
#             if overlap_threshold == 1:
#                 assign[node] = [clean[0]] if clean else [-1]
#             else:
#                 assign[node] = clean[:overlap_threshold]
#         else:
#             assign[node] = [comms] if comms != -1 else [-1]

#     # --- Step 2: Build community -> nodes and sizes ---
#     comm_to_nodes = defaultdict(list)
#     for node, comm_list in assign.items():
#         for cid in comm_list:
#             if cid != -1:
#                 comm_to_nodes[cid].append(node)

#     comm_sizes = {cid: len(nodes) for cid, nodes in comm_to_nodes.items()}
#     small_comms = {cid for cid, size in comm_sizes.items() if size < min_comm_size}
    
#     if not small_comms:
#         if overlap_threshold == 1:
#             return {n: cs[0] if cs and cs[0] != -1 else -1 for n, cs in assign.items()}
#         else:
#             return {n: [c for c in cs if c != -1] for n, cs in assign.items()}

#     large_comms = set(comm_sizes.keys()) - small_comms

#     # Precompute largest small community (for fallback)
#     largest_small = max(small_comms, key=lambda c: comm_sizes[c])

#     # Pre-cache node -> communities for fast access
#     node_to_comms = assign  # alias for clarity

#     # --- Step 3: For each small community, count external connections ---
#     merge_map = {}
#     for small_cid in small_comms:
#         neighbor_comm_count = defaultdict(int)
#         nodes_in_small = comm_to_nodes[small_cid]

#         # Traverse all neighbors of all nodes in this small community
#         for node in nodes_in_small:
#             for nb in G.neighbors(node):
#                 nb_comms = node_to_comms.get(nb)
#                 if nb_comms is None:
#                     continue
#                 for nb_cid in nb_comms:
#                     if nb_cid != -1 and nb_cid != small_cid:
#                         neighbor_comm_count[nb_cid] += 1

#         best_target = small_cid
#         best_score = -1

#         # Check large communities first
#         for candidate in large_comms:
#             score = neighbor_comm_count[candidate]
#             if score > best_score:
#                 best_score = score
#                 best_target = candidate

#         # If no good large target, check other small communities
#         if best_score <= 0:
#             for candidate in small_comms:
#                 if candidate == small_cid:
#                     continue
#                 score = neighbor_comm_count[candidate]
#                 if score > best_score:
#                     best_score = score
#                     best_target = candidate

#         # Final fallback: merge into the largest small community
#         if best_score <= 0:
#             best_target = largest_small

#         merge_map[small_cid] = best_target

#     # --- Step 4: Apply merging ---
#     new_assign = {}
#     for node, comm_list in assign.items():
#         new_list = []
#         seen = set()
#         for cid in comm_list:
#             if cid == -1:
#                 continue
#             target = merge_map.get(cid, cid)
#             if target != -1 and target not in seen:
#                 seen.add(target)
#                 new_list.append(target)
        
#         if overlap_threshold == 1:
#             new_assign[node] = [new_list[0]] if new_list else [-1]
#         else:
#             new_assign[node] = new_list[:overlap_threshold]

#     # --- Step 5: Format output ---
#     if overlap_threshold == 1:
#         return {node: comm_list[0] if comm_list and comm_list[0] != -1 else -1
#                 for node, comm_list in new_assign.items()}
#     else:
#         return {node: [c for c in comm_list if c != -1] 
#                 for node, comm_list in new_assign.items()}

def global_optimization_with_overlap_fixed(G, comm_dict, all_new_nodes):
    """Global optimization with overlap nodes - 修复版"""
    from collections import defaultdict
    
    # 1. 规范化输入数据，确保社区ID是整数
    improved_comm_dict = {}
    for node, comm in comm_dict.items():
        if isinstance(comm, list):
            # 如果是列表，取第一个有效值
            if comm and comm[0] != -1:
                improved_comm_dict[node] = comm[0]
            else:
                improved_comm_dict[node] = -1
        else:
            # 如果是整数，直接使用
            improved_comm_dict[node] = comm
    
    # 2. 统计社区大小（过滤-1）
    comm_sizes = defaultdict(int)
    for node, comm_id in improved_comm_dict.items():
        if comm_id != -1:
            comm_sizes[comm_id] += 1
    
    # 3. 优化循环
    for new_node in all_new_nodes:
        if new_node not in G:
            continue
            
        current_comm = improved_comm_dict.get(new_node, -1)
        neighbors = list(G.neighbors(new_node))
        
        if not neighbors:
            continue
            
        # 4. 统计邻居社区（确保邻居社区是整数）
        neighbor_comms = defaultdict(int)
        for neighbor in neighbors:
            neighbor_comm = improved_comm_dict.get(neighbor, -1)
            if neighbor_comm != -1:  # 只统计有效社区
                neighbor_comms[neighbor_comm] += 1
        
        if not neighbor_comms:
            continue
        
        # 5. 找到最佳社区
        best_comm = max(
            neighbor_comms.items(),
            key=lambda x: (x[1], -comm_sizes.get(x[0], 0))
        )[0]
        
        # 6. 计算连接数
        current_conn = neighbor_comms.get(current_comm, 0)
        best_conn = neighbor_comms[best_comm]
        
        # 7. 决定是否移动
        should_move = (
            best_comm != current_comm and 
            best_conn > current_conn
        )
        
        if should_move:
            # 更新社区大小
            if current_comm != -1:
                comm_sizes[current_comm] = max(0, comm_sizes.get(current_comm, 0) - 1)
            comm_sizes[best_comm] = comm_sizes.get(best_comm, 0) + 1
            
            improved_comm_dict[new_node] = best_comm
    
    return improved_comm_dict

def unified_leiden_optimization(G, comm_dict, all_new_nodes,overlap_threshold=1):
    """统一版本，确保与下面函数结果一致"""
    from collections import defaultdict
    
    # 步骤1：完全复制下面函数的初始化逻辑
    improved_comm_dict = {}
    for node, comm in comm_dict.items():
        if isinstance(comm, list):
            improved_comm_dict[node] = comm[:overlap_threshold]  # 只取第一个
        else:
            improved_comm_dict[node] = [comm]
    
    # 步骤2：完全复制下面函数的统计逻辑
    comm_sizes = defaultdict(int)
    for node, comm_list in improved_comm_dict.items():
        for comm_id in comm_list:
            comm_sizes[comm_id] += 1
    
    # 步骤3：完全复制下面函数的优化逻辑
    for new_node in all_new_nodes:
        if new_node not in G:
            continue
            
        current_comms = improved_comm_dict[new_node]
        neighbors = list(G.neighbors(new_node))
        
        if not neighbors:
            continue
            
        neighbor_comms = defaultdict(int)
        for neighbor in neighbors:
            neighbor_comm_list = improved_comm_dict[neighbor]
            for neighbor_comm in neighbor_comm_list:
                neighbor_comms[neighbor_comm] += 1
        
        if not neighbor_comms:
            continue
            
        # 与下面函数完全相同的选择逻辑
        best_comm = max(
            neighbor_comms.items(),
            key=lambda x: (x[1], -comm_sizes.get(x[0], 0))
        )[0]
        
        # 与下面函数完全相同的移动逻辑
        if current_comms:  # 应该总是True
            current_comm = current_comms[0]
            current_conn = neighbor_comms.get(current_comm, 0)
            best_conn = neighbor_comms[best_comm]
            
            if best_comm != current_comm and best_conn > current_conn:
                comm_sizes[current_comm] = max(0, comm_sizes[current_comm] - 1)
                comm_sizes[best_comm] += 1
                improved_comm_dict[new_node] = [best_comm]
    
    # 步骤4：转换为整数格式
    result_dict = {}
    for node, comm_list in improved_comm_dict.items():
        result_dict[node] = comm_list[0] if comm_list else 0
    
    return result_dict






# def global_optimization_with_overlap(G, comm_dict, new_nodes, allow_overlap=False):
#     """
#     超高效版本：通过预计算和批量处理大幅提升性能
#     """
#     from collections import defaultdict
    
#     # --- 快速规范化社区分配 ---
#     assign = {}
#     for node, comms in comm_dict.items():
#         if isinstance(comms, list):
#             clean = [c for c in comms if c != -1]
#             assign[node] = clean if clean else []
#         else:
#             assign[node] = [comms] if comms != -1 else []
    
#     # --- 预计算：每个节点的社区分配（快速访问）---
#     # 创建 {节点: 社区集合} 用于快速查找
#     node_communities = {}
#     # 同时计算社区大小
#     comm_size = defaultdict(int)
    
#     for node, comms in assign.items():
#         comm_set = set(comms)
#         node_communities[node] = comm_set
#         for c in comm_set:
#             comm_size[c] += 1
    
#     # --- 批量预计算邻居社区统计 ---
#     # 只处理实际在图中存在的新节点
#     valid_new_nodes = [n for n in new_nodes if n in G]
#     if not valid_new_nodes:
#         return _format_output(assign, allow_overlap)
    
#     # 获取所有邻居（一次批量操作）
#     node_neighbors = {n: list(G.neighbors(n)) for n in valid_new_nodes}
    
#     # 批量计算每个新节点的邻居社区统计
#     node_stats = {}  # 节点 -> {community: count}
#     node_neighbor_counts = {}  # 节点 -> 有效邻居数
    
#     for node, neighbors in node_neighbors.items():
#         comm_counts = defaultdict(int)
#         valid_count = 0
        
#         for nb in neighbors:
#             nb_comms = node_communities.get(nb)
#             if nb_comms:
#                 valid_count += 1
#                 for c in nb_comms:
#                     comm_counts[c] += 1
        
#         if valid_count > 0:
#             node_stats[node] = comm_counts
#             node_neighbor_counts[node] = valid_count
    
#     if not node_stats:
#         return _format_output(assign, allow_overlap)
    
#     # --- 批量处理所有节点 ---
#     DOMINANCE_THRESHOLD = 0.55
    
#     # 为非重叠模式预计算
#     if not allow_overlap:
#         for node, comm_counts in node_stats.items():
#             if not comm_counts:
#                 continue
                
#             current_comms = node_communities.get(node, set())
#             current_comm = next(iter(current_comms)) if current_comms else None
            
#             # 找到最佳社区
#             total_hits = sum(comm_counts.values())
#             best_comm = max(comm_counts.items(), key=lambda x: x[1])[0]
#             best_support = comm_counts[best_comm]
            
#             if current_comm is None:
#                 # 新节点
#                 assign[node] = [best_comm]
#                 node_communities[node] = {best_comm}
#                 comm_size[best_comm] += 1
#             else:
#                 # 已有节点，考虑是否切换
#                 current_support = comm_counts.get(current_comm, 0)
#                 if best_support > current_support * 1.2:
#                     # 切换到新社区
#                     assign[node] = [best_comm]
#                     node_communities[node] = {best_comm}
#                     comm_size[current_comm] -= 1
#                     comm_size[best_comm] += 1
#     else:
#         # 重叠模式
#         for node, comm_counts in node_stats.items():
#             if not comm_counts:
#                 continue
                
#             total_hits = sum(comm_counts.values())
#             current_comms = node_communities.get(node, set())
            
#             # 计算每个社区的支持度比例
#             comm_ratios = [(count / total_hits, comm) for comm, count in comm_counts.items()]
#             comm_ratios.sort(reverse=True)
            
#             max_ratio, best_comm = comm_ratios[0]
#             k = len(comm_ratios)
            
#             if max_ratio >= DOMINANCE_THRESHOLD:
#                 selected = {best_comm}
#             else:
#                 avg_ratio = 1.0 / k if k > 0 else 0
#                 selected = {comm for ratio, comm in comm_ratios if ratio >= avg_ratio}
#                 if not selected:
#                     selected = {best_comm}
            
#             if selected != current_comms:
#                 # 更新社区大小
#                 for c in current_comms - selected:
#                     comm_size[c] -= 1
#                 for c in selected - current_comms:
#                     comm_size[c] += 1
                
#                 # 更新节点社区分配
#                 assign[node] = sorted(selected)
#                 node_communities[node] = selected
    
#     return _format_output(assign, allow_overlap)

#2026129 23:40
# def optimize_community_structure(comm_dict, G, min_comm_size, allow_overlap=False):
#     """简洁高效版：合并小社区提升F1值"""
#     from collections import defaultdict
    
#     # 1. 快速构建数据结构
#     node_comms = {}
#     comm_nodes = defaultdict(set)
    
#     for node, comms in comm_dict.items():
#         if isinstance(comms, list):
#             comm_list = [c for c in comms if c != -1]
#         else:
#             comm_list = [comms] if comms != -1 else []
        
#         if comm_list:
#             comm_set = set(comm_list)
#             node_comms[node] = comm_set
#             for c in comm_set:
#                 comm_nodes[c].add(node)
    
#     # 2. 识别小社区（仅关注大小）
#     small_comms = {cid for cid, nodes in comm_nodes.items() 
#                    if len(nodes) < min_comm_size}
    
#     if not small_comms:
#         return _format_output(node_comms, allow_overlap)
    
#     # 3. 为每个小社区找到连接最紧密的合并目标
#     merge_map = {}
    
#     # 预计算邻居缓存（关键优化）
#     adj_cache = {node: set(G.neighbors(node)) for node in node_comms}
    
#     for small_cid in small_comms:
#         nodes = comm_nodes[small_cid]
        
#         # 计算与所有其他社区的连接数
#         neighbor_counts = defaultdict(int)
        
#         for node in nodes:
#             for nb in adj_cache.get(node, set()):
#                 for nb_cid in node_comms.get(nb, set()):
#                     if nb_cid != small_cid:
#                         neighbor_counts[nb_cid] += 1
        
#         # 选择连接最紧密的社区（提升F1的关键）
#         if neighbor_counts:
#             # 直接选择连接数最多的社区
#             best_target = max(neighbor_counts.items(), key=lambda x: x[1])[0]
#             merge_map[small_cid] = best_target
#         else:
#             # 没有连接，保持原样
#             merge_map[small_cid] = small_cid
    
#     # 4. 应用合并（处理合并链）
#     # 解决 A->B, B->C 的情况
#     final_merge = {}
#     for cid in merge_map:
#         target = merge_map[cid]
#         # 追溯最终目标
#         while target in merge_map and merge_map[target] != target:
#             target = merge_map[target]
#         final_merge[cid] = target
    
#     # 5. 更新节点分配
#     new_node_comms = {}
#     for node, comm_set in node_comms.items():
#         # 应用合并映射
#         new_set = {final_merge.get(c, c) for c in comm_set}
#         # 去重：确保同一节点不重复分配同一社区
#         new_node_comms[node] = new_set
    
#     # 6. 非重叠模式处理（如果需要）
#     if not allow_overlap:
#         final_assign = {}
#         for node, comm_set in new_node_comms.items():
#             if not comm_set:
#                 final_assign[node] = -1
#             elif len(comm_set) == 1:
#                 final_assign[node] = next(iter(comm_set))
#             else:
#                 # 有多个社区，选择邻居支持最多的
#                 nb_support = defaultdict(int)
#                 for nb in adj_cache.get(node, set()):
#                     nb_comms = node_comms.get(nb, set())
#                     # 只考虑当前节点的社区
#                     for c in comm_set & nb_comms:
#                         nb_support[c] += 1
                
#                 if nb_support:
#                     final_assign[node] = max(nb_support.items(), key=lambda x: x[1])[0]
#                 else:
#                     final_assign[node] = next(iter(comm_set))
#         return final_assign
    
#     return {node: sorted(cs) for node, cs in new_node_comms.items()}


# def optimize_community_structure(comm_dict, G, min_comm_size, overlap_threshold=1):
#     """
#     High-performance version for large graphs (300k+ nodes).
#     Optimized to avoid nested loops and redundant lookups.
#     """
#     from collections import defaultdict

#     # --- Step 1: Normalize assignment ---
#     assign = {}
#     for node, comms in comm_dict.items():
#         if isinstance(comms, list):
#             clean = [c for c in comms if c != -1]
#             if overlap_threshold == 1:
#                 assign[node] = [clean[0]] if clean else [-1]
#             else:
#                 assign[node] = clean[:overlap_threshold]
#         else:
#             assign[node] = [comms] if comms != -1 else [-1]

#     # --- Step 2: Build community -> nodes and sizes ---
#     comm_to_nodes = defaultdict(list)
#     for node, comm_list in assign.items():
#         for cid in comm_list:
#             if cid != -1:
#                 comm_to_nodes[cid].append(node)

#     comm_sizes = {cid: len(nodes) for cid, nodes in comm_to_nodes.items()}
#     small_comms = {cid for cid, size in comm_sizes.items() if size < min_comm_size}
    
#     if not small_comms:
#         if overlap_threshold == 1:
#             return {n: cs[0] if cs and cs[0] != -1 else -1 for n, cs in assign.items()}
#         else:
#             return {n: [c for c in cs if c != -1] for n, cs in assign.items()}

#     large_comms = set(comm_sizes.keys()) - small_comms

#     # Precompute largest small community (for fallback)
#     largest_small = max(small_comms, key=lambda c: comm_sizes[c])

#     # Pre-cache node -> communities for fast access
#     node_to_comms = assign  # alias for clarity

#     # --- Step 3: For each small community, count external connections ---
#     merge_map = {}
#     for small_cid in small_comms:
#         neighbor_comm_count = defaultdict(int)
#         nodes_in_small = comm_to_nodes[small_cid]

#         # Traverse all neighbors of all nodes in this small community
#         for node in nodes_in_small:
#             for nb in G.neighbors(node):
#                 nb_comms = node_to_comms.get(nb)
#                 if nb_comms is None:
#                     continue
#                 for nb_cid in nb_comms:
#                     if nb_cid != -1 and nb_cid != small_cid:
#                         neighbor_comm_count[nb_cid] += 1

#         best_target = small_cid
#         best_score = -1

#         # Check large communities first
#         for candidate in large_comms:
#             score = neighbor_comm_count[candidate]
#             if score > best_score:
#                 best_score = score
#                 best_target = candidate

#         # If no good large target, check other small communities
#         if best_score <= 0:
#             for candidate in small_comms:
#                 if candidate == small_cid:
#                     continue
#                 score = neighbor_comm_count[candidate]
#                 if score > best_score:
#                     best_score = score
#                     best_target = candidate

#         # Final fallback: merge into the largest small community
#         if best_score <= 0:
#             best_target = largest_small

#         merge_map[small_cid] = best_target

#     # --- Step 4: Apply merging ---
#     new_assign = {}
#     for node, comm_list in assign.items():
#         new_list = []
#         seen = set()
#         for cid in comm_list:
#             if cid == -1:
#                 continue
#             target = merge_map.get(cid, cid)
#             if target != -1 and target not in seen:
#                 seen.add(target)
#                 new_list.append(target)
        
#         if overlap_threshold == 1:
#             new_assign[node] = [new_list[0]] if new_list else [-1]
#         else:
#             new_assign[node] = new_list[:overlap_threshold]

#     # --- Step 5: Format output ---
#     if overlap_threshold == 1:
#         return {node: comm_list[0] if comm_list and comm_list[0] != -1 else -1
#                 for node, comm_list in new_assign.items()}
#     else:
#         return {node: [c for c in comm_list if c != -1] 
#                 for node, comm_list in new_assign.items()}


def _format_output(node_comms, allow_overlap):
    """格式化输出"""
    if not allow_overlap:
        return {n: next(iter(cs)) if cs else -1 for n, cs in node_comms.items()}
    else:
        return {n: sorted(cs) for n, cs in node_comms.items()}

# def optimize_community_structure(comm_dict, G, min_comm_size, allow_overlap=False):
#     """简洁高效版：合并低内聚小社区"""
#     from collections import defaultdict
    
#     # 1. 快速预处理：构建基本数据结构
#     node_comms = {}          # 节点 -> 社区集合
#     comm_nodes = defaultdict(set)  # 社区 -> 节点集合
#     comm_sizes = {}          # 社区大小
    
#     for node, comms in comm_dict.items():
#         if isinstance(comms, list):
#             comm_list = [c for c in comms if c != -1]
#         else:
#             comm_list = [comms] if comms != -1 else []
        
#         if comm_list:
#             comm_set = set(comm_list)
#             node_comms[node] = comm_set
#             for c in comm_set:
#                 comm_nodes[c].add(node)
    
#     # 计算社区大小
#     for cid, nodes in comm_nodes.items():
#         comm_sizes[cid] = len(nodes)
    
#     # 2. 识别需要合并的低内聚小社区
#     small_comms = [cid for cid, size in comm_sizes.items() 
#                   if size < min_comm_size and size > 0]
    
#     if not small_comms:
#         return _format_output(node_comms, allow_overlap)
    
#     # 预缓存邻接关系（关键优化）
#     adj_cache = {node: set(G.neighbors(node)) for node in node_comms}
    
#     # 3. 批量计算合并目标
#     merge_map = {}
#     large_comms = {cid for cid, size in comm_sizes.items() if size >= min_comm_size}
    
#     for small_cid in small_comms:
#         nodes = comm_nodes[small_cid]
#         size = len(nodes)
        
#         # 快速评估内聚度（避免详细计算除非必要）
#         if size == 1:
#             # 单节点社区直接标记为低内聚
#             density = 0
#         else:
#             # 快速近似密度计算
#             node_set = nodes
#             internal_edges = 0
#             sample_nodes = list(nodes)[:min(10, size)]  # 采样计算
            
#             for u in sample_nodes:
#                 internal_edges += len(adj_cache.get(u, set()) & node_set)
            
#             max_possible = len(sample_nodes) * (size - 1)
#             density = internal_edges / max_possible if max_possible > 0 else 0
        
#         # 只合并低内聚社区
#         if density >= 0.15:
#             merge_map[small_cid] = small_cid
#             continue
        
#         # 寻找最佳合并目标
#         neighbor_counts = defaultdict(int)
#         for node in nodes:
#             for nb in adj_cache.get(node, set()):
#                 for nb_cid in node_comms.get(nb, []):
#                     if nb_cid != small_cid:
#                         neighbor_counts[nb_cid] += 1
        
#         if not neighbor_counts:
#             merge_map[small_cid] = small_cid
#             continue
        
#         # 优先合并到大社区
#         candidates = large_comms if large_comms else set(neighbor_counts.keys())
#         best_target = max(candidates, key=lambda c: neighbor_counts.get(c, 0))
#         merge_map[small_cid] = best_target
    
#     # 4. 应用合并
#     new_node_comms = {}
#     for node, comm_set in node_comms.items():
#         new_set = {merge_map.get(c, c) for c in comm_set}
#         new_node_comms[node] = new_set
    
#     # 5. 非重叠模式处理
#     if not allow_overlap:
#         final_assign = {}
#         for node, comm_set in new_node_comms.items():
#             if not comm_set:
#                 final_assign[node] = -1
#             elif len(comm_set) == 1:
#                 final_assign[node] = next(iter(comm_set))
#             else:
#                 # 选择邻居支持最多的社区
#                 nb_support = defaultdict(int)
#                 for nb in adj_cache.get(node, set()):
#                     for c in comm_set & node_comms.get(nb, set()):
#                         nb_support[c] += 1
                
#                 if nb_support:
#                     final_assign[node] = max(nb_support.items(), key=lambda x: x[1])[0]
#                 else:
#                     final_assign[node] = next(iter(comm_set))
#         return final_assign
#     else:
#         return {node: sorted(cs) for node, cs in new_node_comms.items()}


# def _format_output(node_comms, allow_overlap):
#     """格式化输出"""
#     if not allow_overlap:
#         return {n: next(iter(cs)) if cs else -1 for n, cs in node_comms.items()}
#     else:
#         return {n: sorted(cs) for n, cs in node_comms.items()}

                                                 

#20206.1.19 18:50
def global_optimization_with_overlap1(G, comm_dict, new_nodes, allow_overlap=False):
    """
    Assign or reassign communities to new nodes.
    
    Args:
        G: NetworkX graph
        comm_dict: dict {node: comm_id or [comm_ids]}
        new_nodes: iterable of nodes to optimize
        allow_overlap (bool): 
            - If False: each node assigned to exactly one community.
            - If True: node can belong to multiple communities *if supported*;
                      otherwise, automatically falls back to single assignment.

    Returns:
        Updated community dict:
            - {node: int} if allow_overlap=False
            - {node: [int, ...]} if allow_overlap=True (may be singleton list)
    """
    from collections import defaultdict
    import math

    # --- Step 1: Normalize assignment to list format (remove -1) ---
    assign = {}
    for node, comms in comm_dict.items():
        if isinstance(comms, list):
            clean = [c for c in comms if c != -1]
            assign[node] = clean
        else:
            assign[node] = [comms] if comms != -1 else []

    # Filter valid new nodes
    new_nodes = [n for n in new_nodes if n in G and G.degree(n) > 0]
    if not new_nodes:
        return _format_output(assign, allow_overlap)

    # --- Step 2: Compute current community sizes ---
    comm_size = defaultdict(int)
    for comms in assign.values():
        for c in comms:
            comm_size[c] += 1

    # --- Step 3: Process each new node ---
    for node in new_nodes:
        nb_comms = defaultdict(int)
        for nb in G.neighbors(node):
            nb_list = assign.get(nb)
            if nb_list:
                for c in nb_list:
                    nb_comms[c] += 1

        if not nb_comms:
            continue

        current = set(assign.get(node, []))

        if not allow_overlap:
            # Non-overlapping: pick the single best community
            best_comm = max(nb_comms, key=nb_comms.get)
            if not current or best_comm not in current:
                if current:
                    old = current.pop()
                    comm_size[old] -= 1
                comm_size[best_comm] += 1
                assign[node] = [best_comm]
        else:
            total_hits = sum(nb_comms.values())
            if total_hits == 0:
                continue

            # Convert to support ratios
            support_items = [(cnt / total_hits, c) for c, cnt in nb_comms.items()]
            support_items.sort(reverse=True)  # descending by ratio
            max_ratio, best_comm = support_items[0]
            k = len(support_items)

            # 🌟 Parameter-free dominance test: is support highly concentrated?
            dominance_threshold = 1.0 / math.sqrt(k) if k > 0 else 1.0
            # 🌟 改进：添加最小阈值和缓和系数
            # if k <= 1:
            #     dominance_threshold = 1.0
            # elif k <= 3:
            #     # 社区较少时，要求较高
            #     dominance_threshold = max(0.4, 1.0 / math.sqrt(k))
            # else:
            #     # 社区较多时，避免阈值过低
            #     # 使用对数而不是平方根，衰减更慢
            #     dominance_threshold = max(0.3, 1.0 / math.log(k + 1, 3))

            if max_ratio >= dominance_threshold:
                # Strong dominance → assign only the best community (non-overlapping behavior)
                selected = [best_comm]
            else:
                # Support is dispersed → allow overlap, but prune tail
                # Keep communities until cumulative support >= max_ratio (Pareto-like, no param)
                cumsum = 0.0
                selected = []
                for ratio, c in support_items:
                    selected.append(c)
                    cumsum += ratio
                    if cumsum >= max_ratio:
                        break

            selected_set = set(selected)
            if selected_set != current:
                # Update community sizes
                for c in current - selected_set:
                    comm_size[c] -= 1
                for c in selected_set - current:
                    comm_size[c] += 1
                assign[node] = sorted(selected)

    return _format_output(assign, allow_overlap)
#2026.1.31 12:02
# def global_optimization_with_overlap(G, comm_dict, new_nodes, allow_overlap=False):
#     """
#     Assign or reassign communities to new nodes.
    
#     Args:
#         G: NetworkX graph
#         comm_dict: dict {node: comm_id or [comm_ids]}
#         new_nodes: iterable of nodes to optimize
#         allow_overlap (bool): 
#             - If False: each node assigned to exactly one community.
#             - If True: node can belong to multiple communities *if supported*;
#                       otherwise, automatically falls back to single assignment.

#     Returns:
#         Updated community dict:
#             - {node: int} if allow_overlap=False
#             - {node: [int, ...]} if allow_overlap=True (may be singleton list)
#     """
#     from collections import defaultdict
#     import math

#     # --- Step 1: Normalize assignment to list format (remove -1) ---
#     assign = {}
#     for node, comms in comm_dict.items():
#         if isinstance(comms, list):
#             clean = [c for c in comms if c != -1]
#             assign[node] = clean
#         else:
#             assign[node] = [comms] if comms != -1 else []

#     # Filter valid new nodes
#     new_nodes = [n for n in new_nodes if n in G and G.degree(n) > 0]
#     if not new_nodes:
#         return _format_output(assign, allow_overlap)

#     # --- Step 2: Compute current community sizes ---
#     comm_size = defaultdict(int)
#     for comms in assign.values():
#         for c in comms:
#             comm_size[c] += 1

#     # --- Step 3: Process each new node ---
#     for node in new_nodes:
#         nb_comms = defaultdict(int)
#         for nb in G.neighbors(node):
#             nb_list = assign.get(nb)
#             if nb_list:
#                 for c in nb_list:
#                     nb_comms[c] += 1

#         if not nb_comms:
#             continue

#         current = set(assign.get(node, []))

#         if not allow_overlap:
#             # Non-overlapping: pick the single best community with tie-breaking
#             max_votes = max(nb_comms.values())
#             # Find all communities with max votes
#             candidates = [c for c, v in nb_comms.items() if v == max_votes]
            
#             if len(candidates) == 1:
#                 best_comm = candidates[0]
#             else:
#                 # Tie-breaking: choose the smallest community
#                 best_comm = min(candidates, key=lambda c: comm_size.get(c, 0))
            
#             if not current or best_comm not in current:
#                 if current:
#                     old = current.pop()
#                     comm_size[old] -= 1
#                 comm_size[best_comm] += 1
#                 assign[node] = [best_comm]
#         else:
#             total_hits = sum(nb_comms.values())
#             if total_hits == 0:
#                 continue

#             # Sort communities by support ratio, with tie-breaking
#             support_items = []
#             for c, cnt in nb_comms.items():
#                 ratio = cnt / total_hits
#                 # Add community size as secondary sort key (smaller is better)
#                 support_items.append((ratio, -comm_size.get(c, 0), c))
            
#             support_items.sort(reverse=True)  # descending by ratio, then by community size
            
#             max_ratio, _, best_comm = support_items[0]
#             k = len(support_items)

#             # 🌟 Parameter-free dominance test: is support highly concentrated?
#             dominance_threshold = 1.0 / math.sqrt(k) if k > 0 else 1.0
            
#             # 🌟 改进：添加最小阈值
#             if k <= 1:
#                 dominance_threshold = 1.0
#             elif k <= 3:
#                 dominance_threshold = max(0.4, dominance_threshold)
#             else:
#                 dominance_threshold = max(0.3, dominance_threshold)

#             if max_ratio >= dominance_threshold:
#                 # Strong dominance → assign only the best community
#                 selected = [best_comm]
#             else:
#                 # Support is dispersed → allow overlap
#                 cumsum = 0.0
#                 selected = []
#                 for ratio, _, c in support_items:
#                     selected.append(c)
#                     cumsum += ratio
#                     if cumsum >= max_ratio:
#                         break

#             selected_set = set(selected)
#             if selected_set != current:
#                 # Update community sizes
#                 for c in current - selected_set:
#                     comm_size[c] -= 1
#                 for c in selected_set - current:
#                     comm_size[c] += 1
#                 assign[node] = sorted(selected)

#     return _format_output(assign, allow_overlap)


def _format_output(assign, allow_overlap):
    if not allow_overlap:
        return {n: cs[0] if cs else -1 for n, cs in assign.items()}
    else:
        return {n: cs[:] for n, cs in assign.items()}  # return copy of list
#非重叠时结果一样
# def global_optimization_with_overlap(G, comm_dict, new_nodes, allow_overlap=False):
#     """
#     Improved version based on the better-performing algorithm.
#     """
#     from collections import defaultdict
#     import math

#     # --- Step 1: Normalize assignment ---
#     assign = {}
#     for node, comms in comm_dict.items():
#         if isinstance(comms, list):
#             clean = [c for c in comms if c != -1]
#             assign[node] = clean
#         else:
#             assign[node] = [comms] if comms != -1 else []

#     # 过滤节点（保持原逻辑）
#     new_nodes = [n for n in new_nodes if n in G and G.degree(n) > 0]
#     if not new_nodes:
#         return _format_output(assign, allow_overlap)

#     # --- Step 2: 预计算社区大小（关键改进：使用defaultdict避免key错误）---
#     comm_size = defaultdict(int)
#     for comms in assign.values():
#         for c in comms:
#             comm_size[c] += 1

#     # --- Step 3: 处理每个新节点（采用上面的策略）---
#     for node in new_nodes:
#         # 获取邻居社区统计
#         nb_comms = defaultdict(int)
#         for nb in G.neighbors(node):
#             nb_list = assign.get(nb)
#             if nb_list:
#                 for c in nb_list:
#                     nb_comms[c] += 1

#         if not nb_comms:
#             continue

#         # 当前节点所属社区
#         current = set(assign.get(node, []))

#         if not allow_overlap:
#             # 采用上面函数的策略：智能选择 + 条件移动
#             if current:
#                 current_comm = list(current)[0]  # 非重叠模式下只有一个社区
#             else:
#                 current_comm = -1
            
#             # 智能选择：邻居票数优先，小社区优先
#             if nb_comms:
#                 best_comm = max(
#                     nb_comms.items(),
#                     key=lambda x: (x[1], -comm_size.get(x[0], 0))
#                 )[0]
                
#                 # 检查是否需要移动（关键改进）
#                 current_conn = nb_comms.get(current_comm, 0)
#                 best_conn = nb_comms[best_comm]
                
#                 should_move = (
#                     best_comm != current_comm and 
#                     best_conn > current_conn  # 只有更好时才移动
#                 )
                
#                 if should_move:
#                     # 更新社区大小
#                     if current_comm != -1:
#                         comm_size[current_comm] = max(0, comm_size[current_comm] - 1)
#                     comm_size[best_comm] += 1
                    
#                     # 更新分配
#                     assign[node] = [best_comm]
        
#         else:
#             # 重叠模式的改进
#             total_hits = sum(nb_comms.values())
#             if total_hits == 0:
#                 continue

#             # 转换为支持率
#             support_items = [(cnt / total_hits, c) for c, cnt in nb_comms.items()]
#             support_items.sort(reverse=True)
            
#             max_ratio, best_comm = support_items[0]
#             k = len(support_items)

#             # 优势度检验（保持原逻辑）
#             dominance_threshold = 1.0 / math.sqrt(k) if k > 0 else 1.0
            
#             # 添加最小阈值保护（改进）
#             if k <= 1:
#                 dominance_threshold = 1.0
#             elif k <= 3:
#                 dominance_threshold = max(0.4, dominance_threshold)
#             else:
#                 dominance_threshold = max(0.3, dominance_threshold)

#             if max_ratio >= dominance_threshold:
#                 selected = [best_comm]
#             else:
#                 # 累积选择
#                 cumsum = 0.0
#                 selected = []
#                 for ratio, c in support_items:
#                     selected.append(c)
#                     cumsum += ratio
#                     if cumsum >= max_ratio:
#                         break
            
#             # 应用重叠分配（添加条件检查）
#             selected_set = set(selected)
#             if selected_set != current:
#                 # 计算变化
#                 to_remove = current - selected_set
#                 to_add = selected_set - current
                
#                 # 只有当改进时才更新
#                 should_update = True
                
#                 # 简单启发式：检查新分配是否更优
#                 if to_add:
#                     # 计算新分配的连接强度
#                     new_strength = sum(nb_comms.get(c, 0) for c in selected_set)
#                     old_strength = sum(nb_comms.get(c, 0) for c in current)
                    
#                     # 只有当新分配更好时才更新
#                     if new_strength <= old_strength and len(selected_set) >= len(current):
#                         should_update = False
                
#                 if should_update:
#                     # 更新社区大小
#                     for c in to_remove:
#                         comm_size[c] -= 1
#                     for c in to_add:
#                         comm_size[c] += 1
#                     assign[node] = sorted(selected)

#     return _format_output(assign, allow_overlap)


# def _format_output(assign, allow_overlap):
#     """保持原格式"""
#     if not allow_overlap:
#         return {n: cs[0] if cs else -1 for n, cs in assign.items()}
#     else:
#         return {n: cs[:] for n, cs in assign.items()}
    


def global_optimization_with_overlap(G, comm_dict, new_nodes, allow_overlap=False):
    """
    简单直接的重叠逻辑：非重叠选一个最好的，重叠选多个好的
    """
    from collections import defaultdict
    
    # 转换格式
    assign = {}
    for node, comms in comm_dict.items():
        if isinstance(comms, list):
            assign[node] = [c for c in comms if c != -1]
        else:
            assign[node] = [comms] if comms != -1 else []
    
    # 过滤节点
    new_nodes = [n for n in new_nodes if n in G and G.degree(n) > 0]
    if not new_nodes:
        return _simple_format(assign, allow_overlap)
    
    # 计算社区大小
    comm_size = defaultdict(int)
    for comms in assign.values():
        for c in comms:
            comm_size[c] += 1
    
    # 处理每个节点
    for node in new_nodes:
        # 获取邻居投票
        nb_comms = defaultdict(int)
        for nb in G.neighbors(node):
            for c in assign.get(nb, []):
                nb_comms[c] += 1
        
        if not nb_comms:
            continue
        
        current = set(assign.get(node, []))
        
        if not allow_overlap:
            # --- 非重叠：选最好的一个 ---
            # 使用相同的评分标准（票数优先，小社区优先）
            if nb_comms:
                best_comm, best_votes = max(
                    nb_comms.items(),
                    key=lambda x: (x[1], -comm_size.get(x[0], 0))
                )
                
                # 检查当前社区
                current_comm = list(current)[0] if current else -1
                current_votes = nb_comms.get(current_comm, 0)
                
                # 只有新的明显更好才移动
                if best_comm != current_comm and best_votes > current_votes:
                    if current_comm != -1:
                        comm_size[current_comm] = max(0, comm_size[current_comm] - 1)
                    comm_size[best_comm] += 1
                    assign[node] = [best_comm]
        
        else:
            # --- 重叠：选多个好的社区 ---
            # 1. 先找出最好的社区（像非重叠一样）
            best_comm, best_votes = max(
                nb_comms.items(),
                key=lambda x: (x[1], -comm_size.get(x[0], 0))
            )
            
            # 2. 确定哪些社区是"好"的（达到最佳的一定比例）
            selected = set()
            
            # 绝对阈值：至少要有1票（防止噪声）
            min_absolute = 1
            
            # 相对阈值：达到最佳票数的百分比
            # 可以根据需要调整这个比例
            # relative_threshold=best_votes
            if best_votes >= 3:
                relative_threshold = best_votes * 0.5  # 50%
            else:  # best_votes == 1
                relative_threshold = best_votes # 
  
            
            # 选择所有达到阈值的社区
            for comm, votes in nb_comms.items():
                if votes >= min_absolute and votes >= relative_threshold:
                    selected.add(comm)
            
            # 3. 确保我们至少有一个社区
            if not selected:
                selected.add(best_comm)
            
            # 4. 限制最大社区数（防止过度重叠）
            if len(selected) > 3:
                # 保留票数最多的3个
                top_comms = sorted(nb_comms.items(), key=lambda x: x[1], reverse=True)[:3]
                selected = set(comm for comm, _ in top_comms)
            
            # 5. 更新分配
            if selected != current:
                # 更新社区大小
                for c in current - selected:
                    comm_size[c] -= 1
                for c in selected - current:
                    comm_size[c] += 1
                
                assign[node] = sorted(selected)
    
    return _simple_format(assign, allow_overlap)


def _simple_format(assign, allow_overlap):
    if not allow_overlap:
        return {n: cs[0] if cs else -1 for n, cs in assign.items()}
    else:
        return {n: cs[:] for n, cs in assign.items()}
    

# def optimize_community_structure(comm_dict, G, min_comm_size, allow_overlap=False):
#     """
#     Optimize community structure by merging small communities.
    
#     Args:
#         comm_dict (dict): Community assignment. Values can be int or list of ints.
#         G (networkx.Graph): The graph.
#         min_comm_size (int): Communities smaller than this will be merged.
#         allow_overlap (bool): If True, nodes can belong to multiple communities 
#                               based on connectivity (no hard limit). 
#                               If False, each node belongs to exactly one community.
    
#     Returns:
#         dict: Optimized community assignment.
#               - If allow_overlap=False: {node: int}
#               - If allow_overlap=True:  {node: [int, ...]} (variable length)
#     """
#     from collections import defaultdict
#     import math

#     # --- Step 1: Normalize to list format (remove -1) ---
#     assign = {}
#     for node, comms in comm_dict.items():
#         if isinstance(comms, list):
#             clean = [c for c in comms if c != -1]
#             assign[node] = clean
#         else:
#             assign[node] = [comms] if comms != -1 else []

#     # --- Step 2: Build community -> nodes mapping ---
#     comm_to_nodes = defaultdict(list)
#     for node, comm_list in assign.items():
#         for cid in comm_list:
#             if cid != -1:
#                 comm_to_nodes[cid].append(node)

#     comm_sizes = {cid: len(nodes) for cid, nodes in comm_to_nodes.items()}
#     small_comms = {cid for cid, size in comm_sizes.items() if size < min_comm_size}
    
#     if not small_comms:
#         # Return in desired format
#         if not allow_overlap:
#             return {n: cs[0] if cs else -1 for n, cs in assign.items()}
#         else:
#             return {n: cs[:] for n, cs in assign.items()}  # copy

#     large_comms = set(comm_sizes.keys()) - small_comms
#     largest_small = max(small_comms, key=lambda c: comm_sizes[c])
#     node_to_comms = assign

#     # --- Step 3: 优化的确定合并目标 ---
#     merge_map = {}

#     # 预计算每个节点的社区（避免重复查询）
#     node_comms_cache = assign

#     # 预计算大社区的集合，用于快速检查
#     large_comms_set = set(large_comms)

#     for small_cid in small_comms:
#         neighbor_comm_count = defaultdict(int)
#         nodes = comm_to_nodes[small_cid]
        
#         # 遍历当前小社区的所有节点
#         for node in nodes:
#             # 获取节点的所有邻居（一次查询）
#             neighbors = list(G.neighbors(node))
            
#             for nb in neighbors:
#                 # 直接从缓存获取邻居的社区
#                 nb_comms = node_comms_cache.get(nb)
#                 if not nb_comms:
#                     continue
                    
#                 # 快速处理邻居社区
#                 for nb_cid in nb_comms:
#                     if nb_cid != -1 and nb_cid != small_cid:
#                         neighbor_comm_count[nb_cid] += 1
        
#         if not neighbor_comm_count:
#             merge_map[small_cid] = largest_small
#             continue
        
#         # 优先选择大社区
#         best_target = small_cid
#         best_score = -1
        
#         # 只检查连接数>0的大社区
#         for candidate in large_comms_set:
#             score = neighbor_comm_count.get(candidate, 0)
#             if score > best_score:
#                 best_score = score
#                 best_target = candidate
        
#         # 如果没有合适的大社区，检查其他小社区
#         if best_score <= 0:
#             for candidate, score in neighbor_comm_count.items():
#                 if candidate in small_comms and candidate != small_cid and score > best_score:
#                     best_score = score
#                     best_target = candidate
        
#         # 最终回退
#         if best_score <= 0:
#             best_target = largest_small
        
#         merge_map[small_cid] = best_target

#     # --- Step 4: Apply merging ---
#     new_assign = {}
#     for node, comm_list in assign.items():
#         new_list = []
#         seen = set()
#         for cid in comm_list:
#             if cid == -1:
#                 continue
#             target = merge_map.get(cid, cid)
#             if target != -1 and target not in seen:
#                 seen.add(target)
#                 new_list.append(target)
#         new_assign[node] = new_list

#     # --- Step 5: Post-process for overlap semantics ---
#     if not allow_overlap:
#         # Non-overlapping: pick the "best" community among merged ones
#         final_assign = {}
#         for node, comm_list in new_assign.items():
#             if not comm_list:
#                 final_assign[node] = -1
#             elif len(comm_list) == 1:
#                 final_assign[node] = comm_list[0]
#             else:
#                 # Score each candidate by neighbor support in original graph
#                 nb_support = defaultdict(int)
#                 for nb in G.neighbors(node):
#                     for c in node_to_comms.get(nb, []):
#                         if c in comm_list:
#                             nb_support[c] += 1
#                 if nb_support:
#                     best = max(nb_support, key=nb_support.get)
#                     final_assign[node] = best
#                 else:
#                     final_assign[node] = comm_list[0]  # fallback
#         return final_assign

#     else:
#         # Overlapping: keep all merged communities (no artificial cap)
#         # Optional: you could apply a soft filter here (e.g., min connection count),
#         # but per your request, we keep all that survived merging.
#         return {node: sorted(cs) for node, cs in new_assign.items()}


#2026.1.31 13:49  #格式不统一
# def optimize_community_structure(comm_dict, G, min_comm_size, allow_overlap=False):
#     """
#     Optimize community structure by merging small communities.
#     SIMPLIFIED VERSION based on merge_small_communities_fast logic.
#     """
#     from collections import defaultdict
    
#     # --- Step 1: Convert to community-centric format ---
#     comm_to_nodes = defaultdict(set)
#     node_to_primary = {}  # Primary community for each node (for connection calculation)
    
#     for node, comms in comm_dict.items():
#         if isinstance(comms, list):
#             comm_list = [c for c in comms if c != -1]
#         else:
#             comm_list = [comms] if comms != -1 else []
        
#         if not comm_list:
#             continue
            
#         # Track all communities for this node
#         for cid in comm_list:
#             comm_to_nodes[cid].add(node)
        
#         # Choose primary community (largest one this node belongs to)
#         if comm_list:
#             # Find the largest community
#             primary = max(comm_list, key=lambda c: len(comm_to_nodes[c]))
#             node_to_primary[node] = primary
    
#     # --- Step 2: Identify small and large communities ---
#     small_comms = {}
#     large_comms = {}
    
#     for cid, nodes in comm_to_nodes.items():
#         node_list = list(nodes)
#         if len(node_list) < min_comm_size:
#             small_comms[cid] = node_list
#         else:
#             large_comms[cid] = node_list
    
#     if not small_comms:
#         # No merging needed
#         if not allow_overlap:
#             return {node: node_to_primary.get(node, -1) for node in comm_dict}
#         else:
#             # Reconstruct overlap format
#             result = {}
#             for node in comm_dict:
#                 comms = [cid for cid, nodes in comm_to_nodes.items() if node in nodes]
#                 result[node] = sorted(comms) if comms else []
#             return result
    
#     # --- Step 3: Precompute connections (FAST method) ---
#     comm_connections = defaultdict(lambda: defaultdict(int))
    
#     for u, v in G.edges():
#         comm_u = node_to_primary.get(u, -1)
#         comm_v = node_to_primary.get(v, -1)
        
#         if comm_u != -1 and comm_v != -1 and comm_u != comm_v:
#             # Only store if at least one is small community
#             if comm_u in small_comms or comm_v in small_comms:
#                 comm_connections[comm_u][comm_v] += 1
#                 comm_connections[comm_v][comm_u] += 1
    
#     # --- Step 4: Merge small communities ---
#     merged_result = {cid: set(nodes) for cid, nodes in large_comms.items()}
#     comm_size = {cid: len(nodes) for cid, nodes in merged_result.items()}
    
#     # For overlap: track multiple communities per node
#     node_to_final_comms = defaultdict(set)
#     for cid, nodes in large_comms.items():
#         for node in nodes:
#             node_to_final_comms[node].add(cid)
    
#     # Process each small community
#     for small_cid, small_nodes in small_comms.items():
#         connections = comm_connections.get(small_cid, {})
        
#         # Find candidate large communities
#         candidate_large = {c: w for c, w in connections.items() if c in merged_result}
        
#         if candidate_large:
#             if allow_overlap:
#                 # Overlap: merge to communities with strong connections
#                 max_weight = max(candidate_large.values())
#                 threshold = max_weight * 0.5  # 40% of max connection
                
#                 for target_comm, weight in candidate_large.items():
#                     if weight >= threshold:
#                         merged_result[target_comm].update(small_nodes)
#                         comm_size[target_comm] += len(small_nodes)
#                         for node in small_nodes:
#                             node_to_final_comms[node].add(target_comm)
#             else:
#                 # Non-overlap: merge to best community
#                 best_comm = max(
#                     candidate_large.items(),
#                     key=lambda x: (x[1], -comm_size.get(x[0], 0))
#                 )[0]
                
#                 merged_result[best_comm].update(small_nodes)
#                 comm_size[best_comm] += len(small_nodes)
#                 for node in small_nodes:
#                     node_to_primary[node] = best_comm  # Update primary
#         else:
#             # No connections to large communities
#             if not allow_overlap:
#                 # Non-overlap: merge to smallest existing community
#                 if comm_size:
#                     best_comm = min(comm_size.items(), key=lambda x: x[1])[0]
#                 else:
#                     best_comm = 0
#                     merged_result[best_comm] = set()
#                     comm_size[best_comm] = 0
                
#                 merged_result[best_comm].update(small_nodes)
#                 comm_size[best_comm] += len(small_nodes)
#                 for node in small_nodes:
#                     node_to_primary[node] = best_comm
    
#     # --- Step 5: Format output ---
#     if not allow_overlap:
#         # Non-overlap: single community per node
#         final_assign = {}
#         for node in comm_dict:
#             final_assign[node] = node_to_primary.get(node, -1)
#         return final_assign
    
#     else:
#         # Overlap: multiple communities per node
#         final_assign = {}
#         for node in comm_dict:
#             comms = list(node_to_final_comms.get(node, set()))
#             # If no communities from merging, use original ones
#             if not comms:
#                 orig = comm_dict.get(node)
#                 if isinstance(orig, list):
#                     comms = [c for c in orig if c != -1]
#                 elif orig != -1:
#                     comms = [orig]
#             final_assign[node] = sorted(comms)
#         return final_assign
#2026.2.4 18:00
# def optimize_community_structure(comm_dict, G, min_comm_size, allow_overlap=False):
#     """
#     Optimize community structure by merging small communities.
#     MODIFIED VERSION with proper duplicate handling and efficient merging.
#     """
#     from collections import defaultdict
    
#     # --- Step 1: Convert to community-centric format ---
#     comm_to_nodes = defaultdict(set)
#     node_to_primary = {}  # Primary community for each node (for connection calculation)
#     node_all_comms = defaultdict(set)  # All communities each node belongs to (for overlap)
    
#     for node, comms in comm_dict.items():
#         if isinstance(comms, list):
#             comm_list = [c for c in comms if c != -1]
#         else:
#             comm_list = [comms] if comms != -1 else []
        
#         if not comm_list:
#             continue
            
#         # Track all communities for this node
#         for cid in comm_list:
#             comm_to_nodes[cid].add(node)
#             node_all_comms[node].add(cid)
        
#         # Choose primary community (largest one this node belongs to)
#         if comm_list:
#             # Find the largest community this node belongs to
#             primary = max(comm_list, key=lambda c: len(comm_to_nodes[c]))
#             node_to_primary[node] = primary
    
#     # --- Step 2: Identify small and large communities ---
#     small_comms = {}
#     large_comms = {}
    
#     for cid, nodes in comm_to_nodes.items():
#         node_list = list(nodes)
#         if len(node_list) < min_comm_size:
#             small_comms[cid] = set(node_list)  # Store as set for efficiency
#         else:
#             large_comms[cid] = set(node_list)  # Store as set for efficiency
    
#     if not small_comms:
#         # No merging needed
#         if not allow_overlap:
#             return {node: node_to_primary.get(node, -1) for node in comm_dict}
#         else:
#             # Reconstruct overlap format
#             result = {}
#             for node in comm_dict:
#                 comms = list(node_all_comms.get(node, set()))
#                 result[node] = sorted(comms) if comms else []
#             return result
    
#     # --- Step 3: Precompute connections (FAST method) ---
#     comm_connections = defaultdict(lambda: defaultdict(int))
    
#     for u, v in G.edges():
#         comm_u = node_to_primary.get(u, -1)
#         comm_v = node_to_primary.get(v, -1)
        
#         if comm_u != -1 and comm_v != -1 and comm_u != comm_v:
#             # Only store if at least one is small community
#             if comm_u in small_comms or comm_v in small_comms:
#                 comm_connections[comm_u][comm_v] += 1
#                 comm_connections[comm_v][comm_u] += 1
    
#     # --- Step 4: Merge small communities with duplicate handling ---
#     merged_result = {cid: set(nodes) for cid, nodes in large_comms.items()}
#     comm_size = {cid: len(nodes) for cid, nodes in merged_result.items()}
    
#     # For overlap: track multiple communities per node
#     node_to_final_comms = defaultdict(set)
#     for cid, nodes in large_comms.items():
#         for node in nodes:
#             node_to_final_comms[node].add(cid)
    
#     # Keep track of nodes that have been processed (for non-overlap mode)
#     processed_nodes = set()
    
#     # Process each small community
#     for small_cid, small_nodes in small_comms.items():
#         connections = comm_connections.get(small_cid, {})
        
#         # Find candidate large communities
#         candidate_large = {c: w for c, w in connections.items() if c in merged_result}
        
#         if candidate_large:
#             if allow_overlap:
#                 # Overlap: merge to communities with strong connections
#                 max_weight = max(candidate_large.values())
#                 threshold = max_weight   # 50% of max connection
                
#                 for target_comm, weight in candidate_large.items():
#                     if weight >= threshold:
#                         # Calculate truly new nodes for this target community
#                         existing_nodes = merged_result[target_comm]
#                         new_nodes = small_nodes - existing_nodes
                        
#                         if new_nodes:
#                             merged_result[target_comm].update(new_nodes)
#                             comm_size[target_comm] += len(new_nodes)
#                             for node in new_nodes:
#                                 node_to_final_comms[node].add(target_comm)
#             else:
#                 # Non-overlap: merge to best community
#                 best_comm = max(
#                     candidate_large.items(),
#                     key=lambda x: (x[1], -comm_size.get(x[0], 0))
#                 )[0]
                
#                 # Find nodes from this small community that haven't been assigned yet
#                 unassigned_nodes = small_nodes - processed_nodes
                
#                 if unassigned_nodes:
#                     # Calculate truly new nodes for the target community
#                     existing_nodes = merged_result[best_comm]
#                     truly_new_nodes = unassigned_nodes - existing_nodes
                    
#                     if truly_new_nodes:
#                         merged_result[best_comm].update(truly_new_nodes)
#                         comm_size[best_comm] += len(truly_new_nodes)
#                         for node in truly_new_nodes:
#                             node_to_primary[node] = best_comm
                    
#                     # Mark all nodes from this small community as processed
#                     processed_nodes.update(small_nodes)
#         else:
#             # No connections to large communities
#             if not allow_overlap:
#                 # Find nodes from this small community that haven't been assigned yet
#                 unassigned_nodes = small_nodes - processed_nodes
                
#                 if unassigned_nodes:
#                     # Non-overlap: merge to smallest existing community
#                     if comm_size:
#                         best_comm = min(comm_size.items(), key=lambda x: x[1])[0]
#                     else:
#                         best_comm = 0
#                         merged_result[best_comm] = set()
#                         comm_size[best_comm] = 0
                    
#                     # Calculate truly new nodes for the target community
#                     existing_nodes = merged_result[best_comm]
#                     truly_new_nodes = unassigned_nodes - existing_nodes
                    
#                     if truly_new_nodes:
#                         merged_result[best_comm].update(truly_new_nodes)
#                         comm_size[best_comm] += len(truly_new_nodes)
#                         for node in truly_new_nodes:
#                             node_to_primary[node] = best_comm
                    
#                     # Mark all nodes from this small community as processed
#                     processed_nodes.update(small_nodes)
    
#     # --- Step 5: Handle orphaned nodes (nodes not assigned to any large community) ---
#     # Collect all nodes from small communities that need to be assigned
#     all_small_community_nodes = set()
#     for nodes in small_comms.values():
#         all_small_community_nodes.update(nodes)
    
#     # Find orphaned nodes
#     if not allow_overlap:
#         orphaned_nodes = all_small_community_nodes - processed_nodes
#     else:
#         # In overlap mode, check if nodes have any final community
#         orphaned_nodes = set()
#         for node in all_small_community_nodes:
#             if node not in node_to_final_comms:
#                 orphaned_nodes.add(node)
    
#     # Assign orphaned nodes
#     if orphaned_nodes:
#         if not allow_overlap:
#             # For non-overlap, assign to smallest existing community
#             if comm_size:
#                 best_comm = min(comm_size.items(), key=lambda x: x[1])[0]
#             else:
#                 best_comm = 0
#                 merged_result[best_comm] = set()
#                 comm_size[best_comm] = 0
            
#             # Calculate truly new nodes
#             existing_nodes = merged_result[best_comm]
#             truly_new_orphans = orphaned_nodes - existing_nodes
            
#             if truly_new_orphans:
#                 merged_result[best_comm].update(truly_new_orphans)
#                 comm_size[best_comm] += len(truly_new_orphans)
#                 for node in truly_new_orphans:
#                     node_to_primary[node] = best_comm
#         else:
#             # For overlap, assign orphaned nodes to their original communities
#             for node in orphaned_nodes:
#                 original_comms = node_all_comms.get(node, set())
#                 for comm in original_comms:
#                     if comm not in merged_result:
#                         merged_result[comm] = set()
#                         comm_size[comm] = 0
                    
#                     if node not in merged_result[comm]:
#                         merged_result[comm].add(node)
#                         comm_size[comm] += 1
#                         node_to_final_comms[node].add(comm)
    
#     # --- Step 6: Format output ---
#     if not allow_overlap:
#         # Non-overlap: single community per node
#         final_assign = {}
#         for node in comm_dict:
#             if node in node_to_primary:
#                 final_assign[node] = node_to_primary[node]
#             else:
#                 # Check if node was in a large community originally
#                 if node in node_all_comms:
#                     # Use the largest original community
#                     original_comms = node_all_comms[node]
#                     if original_comms:
#                         # Find the largest community
#                         largest_comm = max(original_comms, 
#                                          key=lambda c: len(comm_to_nodes.get(c, set())))
#                         final_assign[node] = largest_comm
#                     else:
#                         final_assign[node] = -1
#                 else:
#                     final_assign[node] = -1
#         return final_assign
    
#     else:
#         # Overlap: multiple communities per node
#         final_assign = {}
#         for node in comm_dict:
#             comms = list(node_to_final_comms.get(node, set()))
#             # If no communities from merging, use original ones
#             if not comms:
#                 comms = list(node_all_comms.get(node, set()))
#             final_assign[node] = sorted(comms)
#         return final_assign

# def optimize_community_structure(comm_dict, G, min_comm_size, allow_overlap=False):
#     """
#     Optimize community structure by merging small communities.
#     - Non-overlap mode: each node belongs to exactly one community.
#     - Overlap mode: nodes can belong to multiple communities, but ALL final communities must have size >= min_comm_size.
#     """
#     from collections import defaultdict

#     # --- Step 1: Build community-centric mappings ---
#     comm_to_nodes = defaultdict(set)
#     node_all_comms = defaultdict(set)

#     for node, comms in comm_dict.items():
#         if isinstance(comms, list):
#             comm_list = [c for c in comms if c != -1]
#         else:
#             comm_list = [comms] if comms != -1 else []
#         if not comm_list:
#             continue
#         for cid in comm_list:
#             comm_to_nodes[cid].add(node)
#             node_all_comms[node].add(cid)

#     # --- Step 2: Split into small and large communities ---
#     small_comms = {}
#     large_comms = {}
#     for cid, nodes in comm_to_nodes.items():
#         node_set = set(nodes)
#         if len(node_set) < min_comm_size:
#             small_comms[cid] = node_set
#         else:
#             large_comms[cid] = node_set

#     if not small_comms:
#         # No small communities → return as-is
#         if not allow_overlap:
#             # Choose largest community per node
#             result = {}
#             for node in comm_dict:
#                 comms = node_all_comms.get(node, [])
#                 if comms:
#                     primary = max(comms, key=lambda c: len(comm_to_nodes[c]))
#                     result[node] = primary
#                 else:
#                     result[node] = -1
#             return result
#         else:
#             return {node: sorted(node_all_comms.get(node, [])) for node in comm_dict}

#     # --- Step 3: Precompute connections ---
#     comm_connections = defaultdict(lambda: defaultdict(int))

#     if not allow_overlap:
#         # Non-overlap: use primary community for efficiency
#         node_to_primary = {}
#         for node, comms in node_all_comms.items():
#             primary = max(comms, key=lambda c: len(comm_to_nodes[c]))
#             node_to_primary[node] = primary

#         for u, v in G.edges():
#             cu = node_to_primary.get(u, -1)
#             cv = node_to_primary.get(v, -1)
#             if cu != -1 and cv != -1 and cu != cv:
#                 if cu in small_comms or cv in small_comms:
#                     comm_connections[cu][cv] += 1
#                     comm_connections[cv][cu] += 1
#     else:
#         # Overlap: compute TRUE inter-community edges
#         # Build node -> communities mapping (already have node_all_comms)
#         for u, v in G.edges():
#             comms_u = node_all_comms.get(u, set())
#             comms_v = node_all_comms.get(v, set())
#             for cu in comms_u:
#                 for cv in comms_v:
#                     if cu != cv and (cu in small_comms or cv in small_comms):
#                         comm_connections[cu][cv] += 1
#                         # Note: we don't double-add; symmetric access is fine

#     # --- Step 4: Initialize merged result with large communities ---
#     merged_result = {cid: set(nodes) for cid, nodes in large_comms.items()}
#     comm_size = {cid: len(nodes) for cid, nodes in merged_result.items()}

#     # Track final community assignments
#     if allow_overlap:
#         node_to_final_comms = defaultdict(set)
#         for cid, nodes in merged_result.items():
#             for node in nodes:
#                 node_to_final_comms[node].add(cid)
#     else:
#         node_to_primary = {}  # Will rebuild during merge
#         processed_nodes = set()

#     # --- Step 5: Merge each small community ---
#     for small_cid, small_nodes in small_comms.items():
#         connections = comm_connections.get(small_cid, {})
#         # Only consider target communities that are already "large" (to avoid creating new small ones)
#         candidate_targets = {c: w for c, w in connections.items() if c in merged_result}

#         if candidate_targets:
#             if allow_overlap:
#                 # Merge to all strong-connected large communities
#                 max_w = max(candidate_targets.values())
#                 threshold = max_w* 0.7  # or: max_w * 0.7 for more flexibility
#                 for target, w in candidate_targets.items():
#                     if w >= threshold:
#                         new_nodes = small_nodes - merged_result[target]
#                         if new_nodes:
#                             merged_result[target].update(new_nodes)
#                             comm_size[target] += len(new_nodes)
#                             for node in new_nodes:
#                                 node_to_final_comms[node].add(target)
#             else:
#                 # Non-overlap: pick best single target
#                 best_target = max(candidate_targets.items(), key=lambda x: (x[1], -comm_size[x[0]]))[0]
#                 unassigned = small_nodes - processed_nodes
#                 if unassigned:
#                     truly_new = unassigned - merged_result[best_target]
#                     if truly_new:
#                         merged_result[best_target].update(truly_new)
#                         comm_size[best_target] += len(truly_new)
#                         for node in truly_new:
#                             node_to_primary[node] = best_target
#                     processed_nodes.update(small_nodes)
#         else:
#             # No connection to any large community → merge to smallest existing large community
#             if merged_result:
#                 best_target = min(comm_size, key=comm_size.get)
#             else:
#                 # Edge case: no large communities at all → create one
#                 best_target = 0
#                 merged_result[best_target] = set()
#                 comm_size[best_target] = 0

#             if allow_overlap:
#                 new_nodes = small_nodes - merged_result[best_target]
#                 if new_nodes:
#                     merged_result[best_target].update(new_nodes)
#                     comm_size[best_target] += len(new_nodes)
#                     for node in new_nodes:
#                         node_to_final_comms[node].add(best_target)
#             else:
#                 unassigned = small_nodes - processed_nodes
#                 if unassigned:
#                     truly_new = unassigned - merged_result[best_target]
#                     if truly_new:
#                         merged_result[best_target].update(truly_new)
#                         comm_size[best_target] += len(truly_new)
#                         for node in truly_new:
#                             node_to_primary[node] = best_target
#                     processed_nodes.update(small_nodes)

#     # --- Step 6: Final output formatting ---
#     if not allow_overlap:
#         result = {}
#         for node in comm_dict:
#             if node in node_to_primary:
#                 result[node] = node_to_primary[node]
#             else:
#                 # Fallback: assign to largest original community
#                 orig_comms = node_all_comms.get(node, [])
#                 if orig_comms:
#                     result[node] = max(orig_comms, key=lambda c: len(comm_to_nodes[c]))
#                 else:
#                     result[node] = -1
#         return result
#     else:
#         # In overlap mode, only communities in `merged_result` are kept (all ≥ min_comm_size)
#         result = {}
#         for node in comm_dict:
#             comms = sorted(node_to_final_comms.get(node, []))
#             result[node] = comms
#         return result

def optimize_community_structure(comm_dict, G, min_comm_size, allow_overlap=False):
    from collections import defaultdict

    # --- Step 1: Build mappings ---
    comm_to_nodes = defaultdict(set)
    node_all_comms = defaultdict(set)

    for node, comms in comm_dict.items():
        comm_list = [c for c in (comms if isinstance(comms, list) else [comms]) if c != -1]
        if not comm_list:
            continue
        for cid in comm_list:
            comm_to_nodes[cid].add(node)
            node_all_comms[node].add(cid)

    # --- Step 2: Split small/large ---
    small_comms = {}
    large_comms = {}
    for cid, nodes in comm_to_nodes.items():
        s = set(nodes)
        (small_comms if len(s) < min_comm_size else large_comms)[cid] = s

    if not small_comms:
        if not allow_overlap:
            # Non-overlap: pick largest community per node
            return {
                node: (
                    max(node_all_comms[node], key=lambda c: len(comm_to_nodes[c]))
                    if node_all_comms[node] else -1
                )
                for node in comm_dict
            }
        else:
            return {node: sorted(node_all_comms.get(node, [])) for node in comm_dict}

    # --- Step 3: Compute connections ---
    comm_connections = defaultdict(lambda: defaultdict(int))

    if not allow_overlap:
        # Only in non-overlap mode: use primary community
        node_to_primary = {
            node: max(comms, key=lambda c: len(comm_to_nodes[c]))
            for node, comms in node_all_comms.items()
        }
        for u, v in G.edges():
            cu, cv = node_to_primary.get(u, -1), node_to_primary.get(v, -1)
            if cu != -1 and cv != -1 and cu != cv and (cu in small_comms or cv in small_comms):
                comm_connections[cu][cv] += 1
                comm_connections[cv][cu] += 1
    else:
        # Overlap mode: NO PRIMARY COMMUNITY. Use full membership.
        for u, v in G.edges():
            for cu in node_all_comms.get(u, ()):
                for cv in node_all_comms.get(v, ()):
                    if cu != cv and (cu in small_comms or cv in small_comms):
                        comm_connections[cu][cv] += 1

    # --- Step 4: Initialize with large communities ---
    merged_result = {cid: set(nodes) for cid, nodes in large_comms.items()}
    comm_size = {cid: len(nodes) for cid, nodes in merged_result.items()}

    if not allow_overlap:
        node_to_primary = {}  # Final assignment
        processed = set()
    else:
        node_to_final_comms = defaultdict(set)
        for cid, nodes in merged_result.items():
            for node in nodes:
                node_to_final_comms[node].add(cid)

    # --- Step 5: Merge small communities ---
    for small_cid, small_nodes in small_comms.items():
        candidates = {c: w for c, w in comm_connections[small_cid].items() if c in merged_result}

        if not candidates and merged_result:
            # No strong connection → fall back to smallest large community
            best_target = min(comm_size, key=comm_size.get)
            candidates = {best_target: 0}  # dummy weight

        if not merged_result:
            # No large community at all → create one
            best_target = 0
            merged_result[best_target] = set()
            comm_size[best_target] = 0
            candidates = {best_target: 0}

        targets = []
        if allow_overlap and candidates:
            max_w = max(candidates.values())
            threshold = max_w
            targets = [c for c, w in candidates.items() if w >= threshold]
        elif candidates:
            # Non-overlap: pick best
            targets = [max(candidates.items(), key=lambda x: (x[1], -comm_size[x[0]]))[0]]

        for target in targets:
            if allow_overlap:
                new_nodes = small_nodes - merged_result[target]
                if new_coords := new_nodes:  # Python 3.8+
                    merged_result[target].update(new_coords)
                    comm_size[target] += len(new_coords)
                    for node in new_coords:
                        node_to_final_comms[node].add(target)
            else:
                unassigned = small_nodes - processed
                truly_new = unassigned - merged_result[target]
                if truly_new:
                    merged_result[target].update(truly_new)
                    comm_size[target] += len(truly_new)
                    for node in truly_new:
                        node_to_primary[node] = target
                processed.update(small_nodes)

    # --- Step 6: Output ---
    if not allow_overlap:
        result = {}
        for node in comm_dict:
            if node in node_to_primary:
                result[node] = node_to_primary[node]
            else:
                # Fallback (rare): assign to largest original community
                comms = node_all_comms.get(node, [])
                result[node] = max(comms, key=lambda c: len(comm_to_nodes[c])) if comms else -1
        return result
    else:
        # ONLY return communities in merged_result (all ≥ min_comm_size)
        return {node: sorted(node_to_final_comms.get(node, [])) for node in comm_dict}

def optimize_community_structure_non_overlap(comm_dict, G, min_comm_size):
    """非重叠社区结构优化"""
    from collections import defaultdict

    # --- Step 1: Build mappings ---
    comm_to_nodes = defaultdict(set)
    node_all_comms = defaultdict(set)

    for node, comms in comm_dict.items():
        comm_list = [c for c in (comms if isinstance(comms, list) else [comms]) if c != -1]
        if not comm_list:
            continue
        for cid in comm_list:
            comm_to_nodes[cid].add(node)
            node_all_comms[node].add(cid)

    # --- Step 2: Split small/large ---
    small_comms = {}
    large_comms = {}
    for cid, nodes in comm_to_nodes.items():
        s = set(nodes)
        (small_comms if len(s) < min_comm_size else large_comms)[cid] = s

    if not small_comms:
        # Non-overlap: pick largest community per node
        return {
            node: (
                max(node_all_comms[node], key=lambda c: len(comm_to_nodes[c]))
                if node_all_comms[node] else -1
            )
            for node in comm_dict
        }

    # --- Step 3: Compute connections ---
    comm_connections = defaultdict(lambda: defaultdict(int))

    # Only in non-overlap mode: use primary community
    node_to_primary = {
        node: max(comms, key=lambda c: len(comm_to_nodes[c]))
        for node, comms in node_all_comms.items()
    }
    for u, v in G.edges():
        cu, cv = node_to_primary.get(u, -1), node_to_primary.get(v, -1)
        if cu != -1 and cv != -1 and cu != cv and (cu in small_comms or cv in small_comms):
            comm_connections[cu][cv] += 1
            comm_connections[cv][cu] += 1

    # --- Step 4: Initialize with large communities ---
    merged_result = {cid: set(nodes) for cid, nodes in large_comms.items()}
    comm_size = {cid: len(nodes) for cid, nodes in merged_result.items()}

    node_to_primary_final = {}  # Final assignment
    processed = set()

    # --- Step 5: Merge small communities ---
    for small_cid, small_nodes in small_comms.items():
        candidates = {c: w for c, w in comm_connections[small_cid].items() if c in merged_result}

        if not candidates and merged_result:
            # No strong connection → fall back to smallest large community
            best_target = min(comm_size, key=comm_size.get)
            candidates = {best_target: 0}  # dummy weight

        if not merged_result:
            # No large community at all → create one
            best_target = 0
            merged_result[best_target] = set()
            comm_size[best_target] = 0
            candidates = {best_target: 0}

        targets = []
        if candidates:
            # Non-overlap: pick best
            targets = [max(candidates.items(), key=lambda x: (x[1], -comm_size[x[0]]))[0]]

        for target in targets:
            unassigned = small_nodes - processed
            truly_new = unassigned - merged_result[target]
            if truly_new:
                merged_result[target].update(truly_new)
                comm_size[target] += len(truly_new)
                for node in truly_new:
                    node_to_primary_final[node] = target
            processed.update(small_nodes)

    # --- Step 6: Output ---
    result = {}
    for node in comm_dict:
        if node in node_to_primary_final:
            result[node] = node_to_primary_final[node]
        else:
            # Fallback (rare): assign to largest original community
            comms = node_all_comms.get(node, [])
            result[node] = max(comms, key=lambda c: len(comm_to_nodes[c])) if comms else -1
    return result


def optimize_community_structure_overlap(comm_dict, G, min_comm_size):
    """重叠社区结构优化"""
    from collections import defaultdict

    # --- Step 1: Build mappings ---
    comm_to_nodes = defaultdict(set)
    node_all_comms = defaultdict(set)

    for node, comms in comm_dict.items():
        comm_list = [c for c in (comms if isinstance(comms, list) else [comms]) if c != -1]
        if not comm_list:
            continue
        for cid in comm_list:
            comm_to_nodes[cid].add(node)
            node_all_comms[node].add(cid)

    # --- Step 2: Split small/large ---
    small_comms = {}
    large_comms = {}
    for cid, nodes in comm_to_nodes.items():
        s = set(nodes)
        (small_comms if len(s) < min_comm_size else large_comms)[cid] = s

    if not small_comms:
        return {node: sorted(node_all_comms.get(node, [])) for node in comm_dict}

    # --- Step 3: Compute connections ---
    comm_connections = defaultdict(lambda: defaultdict(int))

    # Overlap mode: NO PRIMARY COMMUNITY. Use full membership.
    for u, v in G.edges():
        for cu in node_all_comms.get(u, ()):
            for cv in node_all_comms.get(v, ()):
                if cu != cv and (cu in small_comms or cv in small_comms):
                    comm_connections[cu][cv] += 1

    # --- Step 4: Initialize with large communities ---
    merged_result = {cid: set(nodes) for cid, nodes in large_comms.items()}
    comm_size = {cid: len(nodes) for cid, nodes in merged_result.items()}

    node_to_final_comms = defaultdict(set)
    for cid, nodes in merged_result.items():
        for node in nodes:
            node_to_final_comms[node].add(cid)

    # --- Step 5: Merge small communities ---
    for small_cid, small_nodes in small_comms.items():
        candidates = {c: w for c, w in comm_connections[small_cid].items() if c in merged_result}

        if not candidates and merged_result:
            # No strong connection → fall back to smallest large community
            best_target = min(comm_size, key=comm_size.get)
            candidates = {best_target: 0}  # dummy weight

        if not merged_result:
            # No large community at all → create one
            best_target = 0
            merged_result[best_target] = set()
            comm_size[best_target] = 0
            candidates = {best_target: 0}

        targets = []
        if candidates:
            max_w = max(candidates.values())
            threshold = max_w
            targets = [c for c, w in candidates.items() if w >= threshold]

        for target in targets:
            new_nodes = small_nodes - merged_result[target]
            if new_nodes:
                merged_result[target].update(new_nodes)
                comm_size[target] += len(new_nodes)
                for node in new_nodes:
                    node_to_final_comms[node].add(target)

    # --- Step 6: Output ---
    # ONLY return communities in merged_result (all ≥ min_comm_size)
    return {node: sorted(node_to_final_comms.get(node, [])) for node in comm_dict}

def get_adaptive_params(network_type,m,n):
    """Get adaptive parameters based on network type and size"""

    d_bar = 2 * m / n if n > 0 else 0
    
    if n < 50000:
        K = min(10, max(3, round(n / 2000)))
    elif d_bar >= 10:
        K = min(10, max(3, round(n / 16000)))
    else:
        K = min(2000, max(50, round(n / 3000)))
    
    if network_type == 'social':
        if n< 5000:#facebook
           tau = 2
        elif n < 300000:#lj1
            tau = 5
        else: #lj2
           tau = 4
        alpha = 2.0
        
    elif network_type == 'co-purchase':
        if n < 300000:#amazon1
            tau = 4
        else:
            tau = 3 #3 #amazon2
        alpha = 5.0
        
    elif network_type == 'collaboration':#dblp1 dblp2
            tau = 5
            alpha = 3.0
       
        
    
    return K, tau, alpha


from collections import defaultdict, deque

def spp_ultra_fast(edge_df, all_nodes, block_num, lambda_coeff=0.3):
    """超快SPP - 用于30万+节点"""
    print("启动超快SPP...")
    
    # 1. 快速构建邻接表
    adj = defaultdict(set)
    for u, v in edge_df[['u', 'v']].values:
        adj[u].add(v)
        adj[v].add(u)
    
    # 2. 简单估计聚类系数（基于度数）
    degrees = {node: len(adj[node]) for node in all_nodes}
    
    # 简单公式：cc ≈ 1/sqrt(degree+1)
    cc = {}
    for node, deg in degrees.items():
        if deg < 2:
            cc[node] = 0.0
        else:
            cc[node] = 1.0 / (deg ** 0.5 + 1)
    
    # 3. 选择种子（高度数节点）
    block_size = len(all_nodes) // block_num
    seeds = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:block_num * 2]
    seeds = [node for node, _ in seeds]
    
    visited = set()
    blocks = []
    node_seq = []
    
    # 4. 简单社区扩展（BFS变种）
    for seed in seeds:
        if seed in visited:
            continue
        
        community = []
        stack = [seed]
        
        while stack and len(community) < block_size:
            node = stack.pop()
            if node in visited:
                continue
            
            visited.add(node)
            community.append(node)
            node_seq.append(node)
            
            # 按度数排序邻居
            neighbors = list(adj[node])
            neighbors.sort(key=lambda x: degrees.get(x, 0), reverse=True)
            
            # 添加前几个邻居
            for nb in neighbors[:20]:  # 限制数量
                if nb not in visited and nb not in stack:
                    stack.append(nb)
        
        if community:
            blocks.append(community)
        
        if len(blocks) >= block_num:
            break
    
    # 5. 剩余节点直接分配
    remaining = [n for n in all_nodes if n not in visited]
    if remaining:
        # 批量分配到连接最多的块
        for i in range(0, len(remaining), 1000):  # 批量处理
            batch = remaining[i:i+1000]
            for node in batch:
                if node in visited:
                    continue
                    
                # 简单查找
                best_idx = 0
                best_conn = 0
                
                for idx in range(min(5, len(blocks))):  # 只检查前5个块
                    conn = len(adj[node] & set(blocks[idx]))
                    if conn > best_conn:
                        best_conn = conn
                        best_idx = idx
                
                blocks[best_idx].append(node)
                visited.add(node)
                node_seq.append(node)
    
    print(f"超快SPP完成: {len(blocks)} blocks")
    return blocks, node_seq

def execute_HIDC_pipeline_unsupervised(edge_file_path, comm_file_path, network_type):
    """Main unsupervised HIDC pipeline"""
    """Luyun"""
    print(mp.cpu_count())
    start_total_time1 = time.time()
    #print("="*50)
    #print(f"Main process started (unsupervised version): {time.strftime('%Y-%m-%d %H:%M:%S')}")
    #print("="*50)


    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR, exist_ok=True)
        #print(f"Created temporary directory: {TEMP_DIR}")
    #print("\n[1/6 Loading data]")
    num_node, num_edges, num_comm, all_nodes, edges, communties = load_data(edge_file_path, comm_file_path)
    edge_df = pd.DataFrame(edges, columns=['u', 'v'])
    
    #print("\n[2/6 Feature preprocessing]")
    node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
    u_idx = np.array([node_to_idx.get(u, -1) for u in edge_df['u']], dtype=np.int32)
    v_idx = np.array([node_to_idx.get(v, -1) for v in edge_df['v']], dtype=np.int32)
    valid_u = u_idx[u_idx != -1]
    valid_v = v_idx[v_idx != -1]
    u_degrees = np.bincount(valid_u, minlength=len(all_nodes))
    v_degrees = np.bincount(valid_v, minlength=len(all_nodes))
    node_degree = (u_degrees + v_degrees).astype(np.float32)
    node_degree_dict = dict(zip(all_nodes, node_degree))
    
    #print(f"  Node degree stats: mean={node_degree.mean():.2f}, non-zero nodes={np.sum(node_degree > 0)}")
    
    #print("\n[3/6 Calculating edge weights]")
    K, MIN_COMM_SIZE, cn_base_alpha = get_adaptive_params(network_type, len(edges),len(all_nodes))
    # if len(all_nodes)<5000:
    #     MIN_COMM_SIZE=3
    #     cn_base_alpha=0

    total_time1 = (time.time() - start_total_time1) / 60
    # print(f"\nTotal time: {total_time1:.2f} minutes")

    
    print(f"  K: ={K}, MIN_COMM_SIZE:{MIN_COMM_SIZE},cn_base_alpha:{cn_base_alpha}")
  
    # klist = [1,2,3,4,5,6,7,8,9,10,20,50,100,200,500,1000,2000]
    # klist = [0,1,2,3,4,5,6]
    # klist = [2,3,4,5,6]
    klist=[K]


    for K in klist:  
        print(f"  K: ={K}, MIN_COMM_SIZE:{MIN_COMM_SIZE},cn_base_alpha:{cn_base_alpha}")
        # blocks,new_all_nodes= split_data_by_node(all_nodes, node_degree_dict, K)
        start_total_time = time.time()
        blocks, new_all_nodes = split_data_by_connectivity(edge_df, all_nodes, node_degree_dict, K)
        # blocks, new_all_nodes = spp_ultra_fast(edge_df, all_nodes, K)
        
      
      
        num_blocks = len(blocks)
        
        if num_blocks == 0:
        
            raise ValueError("Data splitting failed")
        
        with Pool(processes=NUM_PROCESSES) as pool:
            block_args = [(block_id, block_nodes) for block_id, block_nodes in enumerate(blocks)]
            partial_func = partial(process_block, edge_df=edge_df, cn_base_alpha=cn_base_alpha)
            results = pool.imap_unordered(partial_func, block_args)
            
            weighted_edge_dict = {}
            for bid, bedge in results:
                weighted_edge_dict[bid] = bedge
                #print(f"Received edge weights for block {bid}, total {len(weighted_edge_dict)}/{num_blocks}")
        
        weighted_edge_list = []
        for i in sorted(weighted_edge_dict.keys()):
            df = weighted_edge_dict[i]
            if not df.empty and len(df) > 0:
                weighted_edge_list.append(df)
        
        if weighted_edge_list:
            weighted_edge_df = pd.concat(weighted_edge_list, ignore_index=True)
            weighted_edge_df = weighted_edge_df.astype({
                'u': int,
                'v': int,
                'weight': float
            })
        else:
            weighted_edge_df = pd.DataFrame(columns=['u', 'v', 'weight'])
            #print("Warning: No valid edge data generated")
        
        #print(f"Summarized edge weights: {len(weighted_edge_df)} edges")
        
        #print("\n[4/6 Training unsupervised node embeddings]")
        if not os.path.exists(TEMP_DIR):
            os.makedirs(TEMP_DIR, exist_ok=True)
        # ##print(f"✓ Created temporary directory: {TEMP_DIR}")
        for f in os.listdir(TEMP_DIR):
            os.remove(os.path.join(TEMP_DIR, f))
        
    
        #print("\n[5/6 Generating local communities]")
        comm_args = []
        for block_id, block_nodes in enumerate(blocks):
            block_edges = weighted_edge_df[weighted_edge_df['u'].isin(block_nodes) | weighted_edge_df['v'].isin(block_nodes)]
            comm_args.append((block_edges, block_nodes, block_id))
        
        global_comm_dict = {}
        with Pool(processes=NUM_PROCESSES) as pool:
            comm_results = pool.imap_unordered(generate_block_community, comm_args)
            
            received_blocks = 0
            for result in comm_results:
                bid, bcomm = result
                global_comm_dict.update(bcomm)
                received_blocks += 1
                #print(f"Received community results for block {bid}, total {received_blocks}/{len(blocks)} blocks")
        
        #print("\n[6/6 Global community optimization]")
        global_G = build_global_graph_from_original_optimized(edge_df, all_nodes)
        
        # final_comm_dict1 = unified_leiden_optimization(
        #     global_G, 
        #     global_comm_dict, 
        #     new_all_nodes,
        # )

        allow_overlap=True
        print("global_optimization_with_overlap")
        final_comm_dict1 = global_optimization_with_overlap(
            global_G, 
            global_comm_dict, 
            new_all_nodes,
            allow_overlap
        )
        print("global_optimization_with_overlap1")
        print("optimize_community_structure")
        # final_comm_dict = optimize_community_structure(final_comm_dict1, global_G, MIN_COMM_SIZE,allow_overlap)
        # final_comm_dict = optimize_community_structure(final_comm_dict1, global_G, MIN_COMM_SIZE,allow_overlap)
     
        if allow_overlap:
            final_comm_dict=optimize_community_structure_overlap(final_comm_dict1, global_G, MIN_COMM_SIZE)
        else:
            final_comm_dict=optimize_community_structure_non_overlap(final_comm_dict1, global_G, MIN_COMM_SIZE)

        print("optimize_community_structure1")
        # count_nodes_by_community_overlap(final_comm_dict)

        #print("\n[Performance evaluation]")
        if communties:
            metrics = evaluate_with_correct_format(communties, final_comm_dict)
        
        # shutil.rmtree(TEMP_DIR, ignore_errors=True)
        
        total_time = (time.time() - start_total_time) / 60
        print(f"\nTotal time: {total_time+total_time1:.2f} minutes")
    return global_comm_dict



def count_nodes_by_community_overlap(comm_dict):
    """
    统计属于不同数量社区的节点数，并按要求格式打印。
    
    Args:
        comm_dict (dict): 社区分配字典，值可以是 int 或 list[int]
    
    Returns:
        dict: {k: count}，k 表示社区数量，count 是节点数
    """
    overlap_counts = []
    
    for node, comms in comm_dict.items():
        if isinstance(comms, list):
            # 过滤掉 -1（未分配）
            valid_comms = [c for c in comms if c != -1]
            k = len(valid_comms)
        else:
            # 单值情况：-1 表示未分配（0 个社区），否则为 1 个
            k = 0 if comms == -1 else 1
        overlap_counts.append(k)
    
    from collections import Counter
    dist = Counter(overlap_counts)
    total_nodes = len(comm_dict)
    total_sum = sum(dist.values())
    
    # 按社区数量从高到低排序（可选，也可升序）
    for k in sorted(dist.keys(), reverse=True):
        if k == 0:
            print(f"未分配到任何社区的节点有 {dist[k]} 个")
        else:
            suffix = "个社区" if k >= 2 else "个社区"
            print(f"属于 {k} 个社区的节点有 {dist[k]} 个")
    
    print(f"\n所有类别节点数总和：{total_sum}，原始节点总数：{total_nodes}")
    
    if total_sum == total_nodes:
        print("✅ 节点总数一致，统计完整。")
    else:
        print("❌ 警告：节点总数不一致！")
    
    return dict(dist)


# Dataset configurations
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
    'twitter': {
        'edge_path': 'dataset/twitter-1.90.ungraph.txt',
        'community_path': 'dataset/twitter-1.90.cmty.txt',
        'description': 'LiveJournal social network',
        'network_type': 'social'
    },
    'lj3': {
        'edge_path': 'dataset/com-lj.ungraph.txt',
        'community_path': 'dataset/com-lj.all.cmty.txt',
        'description': 'LiveJournal social network',
        'network_type': 'social'
    },
     'youtube': {
        'edge_path': 'dataset/youtube-1.90.ungraph.txt',
        'community_path': 'dataset/youtube-1.90.cmty.txt',
        'description': 'LiveJournal social network',
        'network_type': 'social'
    },
}

if __name__ == "__main__":
    dataset_name = "amazon1"
    configds = DATASET_CONFIGS.get(dataset_name) 
    
    EDGE_FILE_PATH = configds["edge_path"]
    COMMUNITY_FILE_PATH = configds["community_path"]
    network_type = configds["network_type"]
    #print(f"CPU cores: {mp.cpu_count()}")
    
    if not os.path.exists(EDGE_FILE_PATH):
        raise FileNotFoundError(f"Edge file not found: {EDGE_FILE_PATH}")
    
    if not os.path.exists(COMMUNITY_FILE_PATH):
        print(f"Warning: Community file not found, running unsupervised community detection")
    
    execute_HIDC_pipeline_unsupervised(EDGE_FILE_PATH, COMMUNITY_FILE_PATH, network_type)