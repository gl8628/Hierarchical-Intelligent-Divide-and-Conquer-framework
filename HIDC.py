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
NUM_PROCESSES = min(mp.cpu_count() - 1, 4)
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
    
    print(f"Adjacency nodes: {len(set(adjacency.keys()))}")
    print(f"Common nodes: {len(adjacency.keys() & all_nodes)}")
    
    sorted_nodes = sorted(all_nodes, key=lambda x: node_degree_dict.get(x, 0), reverse=True)
    print(f"Sorted nodes: {len(sorted_nodes)}")
    
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
                print(f"Current block size: {len(current_block)}")
                current_block = []
            
            neighbors = adjacency.get(current, [])
            neighbors_sorted = sorted(neighbors, 
                                    key=lambda x: node_degree_dict.get(x, 0), 
                                    reverse=True)
            for neighbor in neighbors_sorted:
                queue.append(neighbor)
    
    if current_block:
        print(f"Final block: {len(current_block)}")
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
    
    print(f"Generated {len(blocks)} blocks")
    
    new_all_nodes = []
    for i, block in enumerate(blocks):
        print(f"{i}. Block size: {len(block)}")
        new_all_nodes.extend(block)
    
    new_all_nodes = list(set(new_all_nodes))
    
    return blocks, new_all_nodes

def calc_block_edge_weight_no_queue(edge_df, block_nodes, block_id, cn_base_alpha):
    """Calculate edge weights for a block"""
    try:
        start_time = time.time()
        block_node_set = set(block_nodes)
        
        block_mask = edge_df['u'].isin(block_node_set) & edge_df['v'].isin(block_node_set)
        block_edge = edge_df[block_mask].copy()
        edge_count = len(block_edge)
        
        if edge_count == 0:
            print(f"Warning: Block {block_id} has no valid edges")
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
        
        edge_counts['weight'] = edge_counts['count'] + cn_base_alpha * edge_counts['common']
        
        result_edge = block_edge[['u', 'v']].drop_duplicates()
        result_edge = result_edge.merge(
            edge_counts[['u_sorted', 'v_sorted', 'weight']],
            left_on=['u', 'v'],
            right_on=['u_sorted', 'v_sorted'],
            how='left'
        ).fillna(1)[['u', 'v', 'weight']]
        
        print(f"Process {os.getpid()}: Block {block_id} completed, {len(result_edge)} edges, time: {time.time()-start_time:.2f}s")
        return (block_id, result_edge)
    
    except Exception as e:
        print(f"Process {os.getpid()}: Block {block_id} edge weight calculation failed! Error: {str(e)}")
        return (block_id, pd.DataFrame(columns=['u', 'v', 'weight']))

def simplest_structural_embedding(edges, nodes, block_id):
    """Simple structural embedding based on graph structure"""
    degree = {}
    for idx, row in edges.iterrows():
        u = row['u']
        v = row['v']
        degree[u] = degree.get(u, 0) + 1
        degree[v] = degree.get(v, 0) + 1
        
    n_nodes = len(nodes)
    if n_nodes == 0:
        return (block_id, {})
    
    dim = HIDDEN_DIM
    embeddings = {}
    
    for node in nodes:
        deg = degree.get(node, 0)
        h = hash(str(node))
        
        vec = np.zeros(dim, dtype=np.float32)
        vec[0] = deg
        vec[1] = deg / max(n_nodes, 1)
        vec[2] = 1 if deg > 0 else 0
        vec[3] = np.sqrt(deg) if deg > 0 else 0
        
        for i in range(4, dim):
            x = h + i * 1000
            vec[i] = np.sin(x) * 0.5 + 0.5
        
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        
        embeddings[node] = vec
    
    return (block_id, embeddings)

def generate_block_community(args):
    """Generate communities for a block using Leiden algorithm"""
    try:
        weighted_edge_block, node_embed_dict, block_nodes, block_id = args
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
        max_possible_edges = num_nodes * (num_nodes - 1) / 2 if num_nodes > 1 else 1
        density = num_edges / max_possible_edges
        
        def run_leiden():
            try:
                node_list = sorted(block_G.nodes())
                node_to_idx = {node: idx for idx, node in enumerate(node_list)}
                idx_to_node = {idx: node for idx, node in enumerate(node_list)}
                
                block_ig = ig.Graph(directed=False)
                block_ig.add_vertices(len(node_list))
                
                edges, edge_weights = [], []
                for u, v, data in block_G.edges(data=True):
                    edges.append((node_to_idx[u], node_to_idx[v]))
                    edge_weights.append(float(data.get('weight', 1.0)))
                if edges:
                    block_ig.add_edges(edges)
                    block_ig.es['weight'] = edge_weights
                
                block_z = np.array([node_embed_dict.get(node, np.zeros(HIDDEN_DIM)) 
                                  for node in block_nodes], dtype=np.float32)
                
                k_init = auto_kmeans_elbow(block_z)
                print(f"Block {block_id}: KMeans initial clusters k_init={k_init}")
                
                init_comm = KMeans(n_clusters=k_init, random_state=RANDOM_SEED, n_init=3).fit_predict(block_z)
                node_to_comm = dict(zip(block_nodes, init_comm))
                init_membership = [node_to_comm.get(idx_to_node[idx], 0) for idx in range(len(node_list))]
                
                partition = la.find_partition(
                    block_ig,
                    la.ModularityVertexPartition,
                    weights='weight',
                    n_iterations=20,
                    initial_membership=init_membership,
                    seed=42
                )
                
                leiden_comm = np.array(partition.membership)
                return [leiden_comm[node_to_idx[node]] if node in node_to_idx else 0 
                        for node in block_nodes]
            
            except Exception as e:
                print(f"Block {block_id} Leiden failed, fallback to KMeans: {str(e)}")
                return run_kmeans()
        
        def run_kmeans():
            block_z = np.array([node_embed_dict.get(node, np.zeros(HIDDEN_DIM)) 
                              for node in block_nodes], dtype=np.float32)
            k_init = auto_kmeans_elbow(block_z)
            print(f"Block {block_id}: Using KMeans, clusters k_init={k_init}")
            kmeans = KMeans(n_clusters=k_init, random_state=RANDOM_SEED, n_init=5)
            return kmeans.fit_predict(block_z)
        
        print(f"  Block {block_id}: Dense network (density {density:.6f}), using standard CPM-Leiden")
        block_fine_comm = run_leiden()
        
        global_comm_prefix = block_id * 1000000
        block_comm_dict = {node: global_comm_prefix + comm 
                         for node, comm in zip(block_nodes, block_fine_comm)}
        
        print(f"Process {os.getpid()}: Block {block_id} generated {len(set(block_fine_comm))} communities, time: {time.time()-start_time:.2f}s")
        return (block_id, block_comm_dict)
    
    except Exception as e:
        print(f"Process {os.getpid()}: Block {block_id} community generation failed! Error: {str(e)}")
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
    print("  Building global graph...")
    start_time = time.time()
    
    global_G = nx.Graph()
    global_G.add_nodes_from(all_nodes)
    all_nodes_set = set(all_nodes)
    
    for row in weighted_edge_df.itertuples():
        u, v, weight = row.u, row.v, row.weight
        if u in all_nodes_set and v in all_nodes_set:
            global_G.add_edge(u, v, weight=float(weight))
    
    print(f"  Global graph: {global_G.number_of_nodes()} nodes, {global_G.number_of_edges()} edges")
    print(f"  Build time: {time.time()-start_time:.2f}s")
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

def build_global_graph_from_original(edge_df, nodes):
    """Build global graph from original edges"""
    print("  Building global graph from original edges...")
    start_time = time.time()
    
    edge_accumulated = {}
    for _, row in edge_df.iterrows():
        u, v = row['u'], row['v']
        edge_key = (min(u, v), max(u, v))
        edge_accumulated[edge_key] = edge_accumulated.get(edge_key, 0) + 1.0
    
    processed_edges = []
    for (u, v), weight in edge_accumulated.items():
        processed_edges.append({'u': u, 'v': v, 'weight': weight})
    
    processed_df = pd.DataFrame(processed_edges)
    print(f"  Processed edges: {len(processed_df)}")
    
    G = nx.Graph()
    all_nodes = np.unique(processed_df[['u', 'v']].values.flatten())
    G.add_nodes_from(all_nodes)
    print(f"  Nodes: {len(all_nodes)}")
    
    for _, row in processed_df.iterrows():
        G.add_edge(row['u'], row['v'], weight=row['weight'])
    
    print(f"  Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
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
    
    print(f"Small communities to merge: {len(small_comms)}")
    
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
    
    print(f"Merged communities: {len(final_result)}")
    print(f"Remaining small communities: {remaining_small}")
    if sizes:
        print(f"Community sizes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.2f}")
    
    original_nodes = sum(len(nodes) for nodes in communities.values())
    result_nodes = sum(len(nodes) for nodes in final_result.values())
    assert original_nodes == result_nodes, f"Node count mismatch: {original_nodes} != {result_nodes}"
    
    return final_result

def optimize_community_structure(node_to_community_dict, global_G, min_size=3):
    """Optimize community structure"""
    print("Optimizing community structure...")
    
    community_to_nodes = defaultdict(list)
    for node, comm_id in node_to_community_dict.items():
        community_to_nodes[comm_id].append(node)
    
    print(f"Converted communities: {len(community_to_nodes)}")
    
    comm_sizes = [len(nodes) for nodes in community_to_nodes.values()]
    print(f"Community size stats: min={min(comm_sizes)}, max={max(comm_sizes)}, avg={np.mean(comm_sizes):.2f}")
    
    merged_community_to_nodes = merge_small_communities_fast(community_to_nodes, global_G, min_size)
    
    print(f"Merged communities: {len(merged_community_to_nodes)}")
    
    final_node_to_community = {}
    for comm_id, nodes in merged_community_to_nodes.items():
        for node in nodes:
            final_node_to_community[node] = comm_id
    
    return final_node_to_community

def global_optimization_with_overlap(G, comm_dict, all_new_nodes):
    """Global optimization with overlap nodes"""
    from collections import defaultdict
    
    print(f"Starting global optimization with {len(all_new_nodes)} nodes")
    
    improved_comm_dict = comm_dict.copy()
    comm_sizes = defaultdict(int)
    for node, comm_id in improved_comm_dict.items():
        comm_sizes[comm_id] += 1
    
    for new_node in all_new_nodes:
        if new_node not in G:
            continue
            
        current_comm = improved_comm_dict[new_node]
        neighbors = list(G.neighbors(new_node))
        
        if not neighbors:
            continue
            
        neighbor_comms = defaultdict(int)
        for neighbor in neighbors:
            neighbor_comm = improved_comm_dict.get(neighbor, -1)
            if neighbor_comm != -1:
                neighbor_comms[neighbor_comm] += 1
        
        if neighbor_comms:
            best_comm = max(
                neighbor_comms.items(),
                key=lambda x: (x[1], -comm_sizes.get(x[0], 0))
            )[0]
            
            current_conn = neighbor_comms.get(current_comm, 0)
            best_conn = neighbor_comms[best_comm]
            
            should_move = (
                best_comm != current_comm and 
                best_conn > current_conn
            )
            
            if should_move:
                comm_sizes[current_comm] = max(0, comm_sizes.get(current_comm, 0) - 1)
                comm_sizes[best_comm] = comm_sizes.get(best_comm, 0) + 1
                
                improved_comm_dict[new_node] = best_comm
    
    print("Global optimization completed")
    return improved_comm_dict

def get_adaptive_params(network_type, node_count):
    """Get adaptive parameters based on network type and size"""
    if network_type == 'social':
        if node_count < 100000:
            K = 3
        else:
            K = 10
        tau = 4
        alpha = 1.0
        
    elif network_type == 'co-purchase':
        if node_count < 300000:
            K = 7
        else:
            K = 10
        tau = 4
        alpha = 5.0
        
    elif network_type == 'collaboration':
        if node_count < 300000:
            K = 7
        else:
            K = 10
        tau = 5
        alpha = 4.0
        
    else:
        raise ValueError(f"Unknown network type: {network_type}")
    
    return K, tau, alpha

def execute_HIDC_pipeline_unsupervised(edge_file_path, comm_file_path, network_type):
    """Main unsupervised HIDC pipeline"""
    start_total_time = time.time()
    print("="*50)
    print(f"Main process started (unsupervised version): {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)

    os.makedirs(TEMP_DIR, exist_ok=True)
    
    print("\n[1/6 Loading data]")
    num_node, num_edges, num_comm, all_nodes, edges, communties = load_data(edge_file_path, comm_file_path)
    edge_df = pd.DataFrame(edges, columns=['u', 'v'])
    
    print("\n[2/6 Feature preprocessing]")
    node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
    u_idx = np.array([node_to_idx.get(u, -1) for u in edge_df['u']], dtype=np.int32)
    v_idx = np.array([node_to_idx.get(v, -1) for v in edge_df['v']], dtype=np.int32)
    valid_u = u_idx[u_idx != -1]
    valid_v = v_idx[v_idx != -1]
    u_degrees = np.bincount(valid_u, minlength=len(all_nodes))
    v_degrees = np.bincount(valid_v, minlength=len(all_nodes))
    node_degree = (u_degrees + v_degrees).astype(np.float32)
    node_degree_dict = dict(zip(all_nodes, node_degree))
    
    print(f"  Node degree stats: mean={node_degree.mean():.2f}, non-zero nodes={np.sum(node_degree > 0)}")
    
    print("\n[3/6 Calculating edge weights]")
    K, MIN_COMM_SIZE, cn_base_alpha = get_adaptive_params(network_type, len(all_nodes))
    blocks, new_all_nodes = split_data_by_connectivity(edge_df, all_nodes, node_degree_dict, K)
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
            print(f"Received edge weights for block {bid}, total {len(weighted_edge_dict)}/{num_blocks}")
    
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
        print("Warning: No valid edge data generated")
    
    print(f"Summarized edge weights: {len(weighted_edge_df)} edges")
    
    print("\n[4/6 Training unsupervised node embeddings]")
    for f in os.listdir(TEMP_DIR):
        os.remove(os.path.join(TEMP_DIR, f))
    
    global_embed = {}
    with Pool(processes=NUM_PROCESSES) as pool:
        embed_results = []
        for block_id, block_nodes in enumerate(blocks):
            block_edges = weighted_edge_df[weighted_edge_df['u'].isin(block_nodes) | weighted_edge_df['v'].isin(block_nodes)]
            res = pool.apply_async(
                simplest_structural_embedding,
                args=(block_edges, block_nodes, block_id)
            )
            embed_results.append(res)
        
        for res in embed_results:
            bid, emb = res.get()
            for node, vec in emb.items():
                vec = enforce_array_type(vec, dtype=np.float32, shape=HIDDEN_DIM)
                global_embed[node] = vec
    
    print(f"  Embedding summary: {len(global_embed)} nodes")
    missing_embed = [node for node in all_nodes if node not in global_embed]
    if missing_embed:
        print(f"  Filling embeddings for {len(missing_embed)} missing nodes")
        for node in missing_embed:
            global_embed[node] = np.zeros(HIDDEN_DIM, dtype=np.float32)
    
    print("\n[5/6 Generating local communities]")
    comm_args = []
    for block_id, block_nodes in enumerate(blocks):
        block_edges = weighted_edge_df[weighted_edge_df['u'].isin(block_nodes) | weighted_edge_df['v'].isin(block_nodes)]
        comm_args.append((block_edges, global_embed, block_nodes, block_id))
    
    global_comm_dict = {}
    with Pool(processes=NUM_PROCESSES) as pool:
        comm_results = pool.imap_unordered(generate_block_community, comm_args)
        
        received_blocks = 0
        for result in comm_results:
            bid, bcomm = result
            global_comm_dict.update(bcomm)
            received_blocks += 1
            print(f"Received community results for block {bid}, total {received_blocks}/{len(blocks)} blocks")
    
    print("\n[6/6 Global community optimization]")
    global_G = build_global_graph_from_original(edge_df, all_nodes)
    
    final_comm_dict1 = global_optimization_with_overlap(
        global_G, 
        global_comm_dict, 
        new_all_nodes,
    )
    final_comm_dict = optimize_community_structure(final_comm_dict1, global_G, MIN_COMM_SIZE)
    
    print("\n[Performance evaluation]")
    if communties:
        metrics = evaluate_with_correct_format(communties, final_comm_dict)
    
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    
    total_time = (time.time() - start_total_time) / 60
    print(f"\nTotal time: {total_time:.2f} minutes")
    return global_comm_dict

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
}

if __name__ == "__main__":
    dataset_name = "amazon1"
    configds = DATASET_CONFIGS.get(dataset_name)
    
    EDGE_FILE_PATH = configds["edge_path"]
    COMMUNITY_FILE_PATH = configds["community_path"]
    network_type = configds["network_type"]
    print(f"CPU cores: {mp.cpu_count()}")
    
    if not os.path.exists(EDGE_FILE_PATH):
        raise FileNotFoundError(f"Edge file not found: {EDGE_FILE_PATH}")
    
    if not os.path.exists(COMMUNITY_FILE_PATH):
        print(f"Warning: Community file not found, running unsupervised community detection")
    
    execute_HIDC_pipeline_unsupervised(EDGE_FILE_PATH, COMMUNITY_FILE_PATH, network_type)