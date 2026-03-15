import random
import numpy as np
import re
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from embedding_cache import get_embedding_cache


def clean_step_texts(step_texts: Optional[List[str]]) -> Optional[List[str]]:
    if step_texts is None:
        return None
    cleaned = []
    for s in step_texts:
        if not s:
            cleaned.append("")
            continue
        t = str(s).strip()
        t = re.sub(r"^\s*Step\s*\d+\s*:\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"^\s*\d+\s*:\s*", "", t)
        t = " ".join(t.split())
        cleaned.append(t)
    return cleaned


def links_to_edges(tools: List[str], links) -> List[Tuple[int, int]]:
    if not tools or not links:
        return []
    if isinstance(links, list) and links and isinstance(links[0], dict):
        if "source" in links[0] and "target" in links[0]:
            edges = []
            if isinstance(links[0].get("source"), int):
                for x in links:
                    u, v = x.get("source"), x.get("target")
                    if isinstance(u, int) and isinstance(v, int) and 0 <= u < len(tools) and 0 <= v < len(tools):
                        edges.append((u, v))
                return edges
            pos_map = {}
            for i, t in enumerate(tools):
                pos_map.setdefault(t, []).append(i)
            for x in links:
                u_name, v_name = x.get("source"), x.get("target")
                if not isinstance(u_name, str) or not isinstance(v_name, str):
                    continue
                if u_name not in pos_map or v_name not in pos_map:
                    continue
                u_positions = pos_map[u_name]
                v_positions = pos_map[v_name]
                best = None
                best_score = float("inf")
                for u in u_positions:
                    for v in v_positions:
                        if u == v:
                            continue
                        if v > u:
                            score = v - u
                        else:
                            score = 10000 + abs(v - u)
                        if score < best_score:
                            best_score = score
                            best = (u, v)
                if best is not None:
                    edges.append(best)
            return edges
    if isinstance(links, list) and links and isinstance(links[0], (list, tuple)):
        edges = []
        for e in links:
            if len(e) == 2:
                u, v = e[0], e[1]
                if isinstance(u, int) and isinstance(v, int) and 0 <= u < len(tools) and 0 <= v < len(tools):
                    edges.append((u, v))
        return edges
    if isinstance(links, list) and links and isinstance(links[0], str):
        pos_map = {}
        for i, t in enumerate(tools):
            pos_map.setdefault(t, []).append(i)
        edges = []
        for link in links:
            if not isinstance(link, str):
                continue
            parts = link.split(", ")
            if len(parts) != 2:
                continue
            u_name, v_name = parts[0].strip(), parts[1].strip()
            if u_name not in pos_map or v_name not in pos_map:
                continue
            u_positions = pos_map[u_name]
            v_positions = pos_map[v_name]
            best = None
            best_score = float("inf")
            for u in u_positions:
                for v in v_positions:
                    if u == v:
                        continue
                    if v > u:
                        score = v - u
                    else:
                        score = 10000 + abs(v - u)
                    if score < best_score:
                        best_score = score
                        best = (u, v)
            if best is not None:
                edges.append(best)
        return edges
    return []


def order_chain_by_edges(tools: List[str], links) -> Tuple[List[str], List[int]]:
    if not tools:
        return [], []
    if len(tools) == 1:
        return tools[:], [0]
    edges = links_to_edges(tools, links)
    if not edges:
        return tools[:], list(range(len(tools)))
    adj = defaultdict(list)
    indeg = [0] * len(tools)
    for u, v in edges:
        if u == v:
            continue
        adj[u].append(v)
        indeg[v] += 1
    roots = [i for i, d in enumerate(indeg) if d == 0]
    if len(roots) != 1:
        return tools[:], list(range(len(tools)))
    order = []
    visited = set()
    cur = roots[0]
    while cur is not None and cur not in visited:
        order.append(cur)
        visited.add(cur)
        nxts = adj.get(cur, [])
        if len(nxts) != 1:
            break
        cur = nxts[0]
    if len(order) != len(tools):
        return tools[:], list(range(len(tools)))
    ordered_tools = [tools[i] for i in order]
    return ordered_tools, order


def edges_to_links(nodes: List[str], edges: List[Tuple[int, int]]) -> List[str]:
    links = []
    for (u, v) in edges or []:
        if isinstance(u, int) and isinstance(v, int) and 0 <= u < len(nodes) and 0 <= v < len(nodes):
            links.append(f"{nodes[u]}, {nodes[v]}")
    return links


def order_chain_with_steps_and_edges(tools: List[str], steps: List[str], links):
    ordered_tools, order = order_chain_by_edges(tools, links)
    ordered_steps = steps

    edges_raw = links_to_edges(tools, links)
    if edges_raw:
        old_to_new = {old: new for new, old in enumerate(order)}
        ordered_edges = []
        for u, v in edges_raw:
            if u in old_to_new and v in old_to_new and u != v:
                ordered_edges.append((old_to_new[u], old_to_new[v]))
        ordered_edges = sorted(set(ordered_edges))
    else:
        ordered_edges = [(i, i + 1) for i in range(len(ordered_tools) - 1)] if len(ordered_tools) > 1 else []

    return ordered_tools, ordered_steps, ordered_edges


def build_io_types_vocab(tool_meta: Dict) -> Tuple[List[str], Dict[str, int]]:
    io_types_set = set()

    for tool_info in tool_meta.get("nodes", []):
        input_types = tool_info.get("input-type", [])
        if isinstance(input_types, list):
            for t in input_types:
                if isinstance(t, str):
                    io_types_set.add(t)
        elif isinstance(input_types, str):
            io_types_set.add(input_types)

        output_types = tool_info.get("output-type", [])
        if isinstance(output_types, list):
            for t in output_types:
                if isinstance(t, str):
                    io_types_set.add(t)
        elif isinstance(output_types, str):
            io_types_set.add(output_types)

    IO_TYPES = sorted(list(io_types_set))
    IO_TYPE2IDX = {t: i for i, t in enumerate(IO_TYPES)}
    return IO_TYPES, IO_TYPE2IDX


def build_io_multihot(tool_name: str, tool_meta: Dict, IO_TYPE2IDX: Dict[str, int], 
                      num_io_types: int) -> Tuple[List[int], List[int]]:
    x_in_IO = [0] * num_io_types
    x_out_IO = [0] * num_io_types

    tool_info = None
    if "nodes" in tool_meta:
        for node in tool_meta["nodes"]:
            if node.get("id") == tool_name:
                tool_info = node
                break
    
    if tool_info is None:
        return x_in_IO, x_out_IO

    input_types = tool_info.get("input-type", [])
    if isinstance(input_types, list):
        for t in input_types:
            if isinstance(t, str) and t in IO_TYPE2IDX:
                x_in_IO[IO_TYPE2IDX[t]] = 1

    output_types = tool_info.get("output-type", [])
    if isinstance(output_types, list):
        for t in output_types:
            if isinstance(t, str) and t in IO_TYPE2IDX:
                x_out_IO[IO_TYPE2IDX[t]] = 1
    
    return x_in_IO, x_out_IO


def build_typed_ngrams(train_data: List[Dict], tool_meta: Dict, n_range=(2, 4)) -> Dict:
    f2_counts = defaultdict(int)
    f3_counts = defaultdict(int)
    f4_counts = defaultdict(int)
    motif_counts = defaultdict(lambda: defaultdict(int))
    
    for ex in train_data:
        chain = []
        for n in ex.get("task_nodes", []):
            task = n.get("task") if isinstance(n, dict) else n
            chain.append(task)
        
        if len(chain) < 2:
            continue

        for i in range(len(chain) - 1):
            f2_counts[(chain[i], chain[i+1])] += 1

        for i in range(len(chain) - 2):
            t_u, t_v = chain[i], chain[i+2]
            motif = tuple(chain[i:i+3])
            motif_counts[(t_u, t_v)][motif] += 1
            f3_counts[motif] += 1

        for i in range(len(chain) - 3):
            t_u, t_v = chain[i], chain[i+3]
            motif = tuple(chain[i:i+4])
            motif_counts[(t_u, t_v)][motif] += 1
            f4_counts[motif] += 1

    f_gt_motif = {}
    for (t_u, t_v), motifs in motif_counts.items():
        f_gt_motif[(t_u, t_v)] = max(motifs.values()) if motifs else 0

    nonzero_freqs = [f for f in f_gt_motif.values() if f > 0]
    if nonzero_freqs:
        tau_motif = float(np.percentile(nonzero_freqs, 80))
    else:
        tau_motif = 1.0
    
    result = {
        '__f2__': dict(f2_counts),
        '__f3__': dict(f3_counts),
        '__f4__': dict(f4_counts),
        '__f_gt_motif__': f_gt_motif,
        '__tau_motif__': tau_motif
    }
    return result


def build_confusion_prior(tool_meta: Dict, topk: int=10,
                          cache_dir: Optional[str] = None, device: Optional[str] = None,
                          lm_name: Optional[str] = None) -> Dict:
    try:
        tool_descs = {}
        for n in tool_meta["nodes"]:
            tool_id = n["id"]
            desc = n.get("desc", tool_id)
            tool_descs[tool_id] = desc

        cache = get_embedding_cache(cache_dir=cache_dir, device=device, lm_name=lm_name or "intfloat/e5-large")
        
        tool_ids = list(tool_descs.keys())
        if not tool_ids:
            return {'top_k': {}, 'p_confuse': {}}
        
        descs = [tool_descs[tid] for tid in tool_ids]
        embeddings = cache.encode_texts(descs, prefix="passage")
        s_all = embeddings @ embeddings.T
        np.fill_diagonal(s_all, -np.inf)
        
        top_k_confuse = {}
        p_confuse_dict = {}
        k = max(0, min(topk, len(tool_ids) - 1))
        
        for i, t_i in enumerate(tool_ids):
            if k == 0:
                top_k_confuse[t_i] = []
                p_confuse_dict[t_i] = 0.0
                continue
            
            row = s_all[i]
            if k < len(tool_ids) - 1:
                idx = np.argpartition(-row, k)[:k]
                idx = idx[np.argsort(-row[idx])]
            else:
                idx = np.argsort(-row)
            
            sims = [(tool_ids[j], float(row[j])) for j in idx if row[j] > 0]
            top_k_confuse[t_i] = sims
            p_confuse_dict[t_i] = max((s for _, s in sims), default=0.0)
        
        result = {
            'top_k': top_k_confuse,
            'p_confuse': p_confuse_dict
        }
        return result

    except Exception:
        return {'top_k': {}, 'p_confuse': {}}


def generate_perturbations_with_labels(gt_ex, confusion, tool_meta, typed_ngrams,
                                       embedding_cache=None, cache_dir=None,
                                       device=None, lm_name=None,
                                       align_proj=None, cost_tau=0.8):
    items = []
    rng = random.Random(hash(str(gt_ex["id"])) % 10**9)
    ETA_CONF_BASE = 0.7
    ETA_DROP_LEN = 0.25
    ETA_DROP_REL = 0.2
    ETA_CMP_LEN = 0.30
    ETA_CMP_COVER = 0.4
    TAU_COST = float(cost_tau)

    gt_tools_raw = []
    for n in gt_ex.get("task_nodes", []):
        task = n.get("task") if isinstance(n, dict) else n
        gt_tools_raw.append(task)
    
    if not gt_tools_raw:
        return items

    meta_map = {}
    tool_descs = {}
    for n in tool_meta["nodes"]:
        tool_id = n["id"]
        in_t = n.get("input-type", [])
        out_t = n.get("output-type", [])
        if not isinstance(in_t, list):
            in_t = [in_t] if in_t else []
        if not isinstance(out_t, list):
            out_t = [out_t] if out_t else []
        meta_map[tool_id] = (set(in_t), set(out_t))
        tool_descs[tool_id] = n.get("desc", tool_id)

    gt_links = gt_ex.get("task_links", [])
    gt_tools, gt_step_texts_orig, gt_edges = order_chain_with_steps_and_edges(
        gt_tools_raw,
        clean_step_texts(gt_ex.get("task_steps", [])),
        gt_links
    )
    
    user_request = gt_ex.get("user_request", "")

    in_index = defaultdict(set)
    out_index = defaultdict(set)
    for tool_id, (in_t, out_t) in meta_map.items():
        for t in in_t:
            in_index[t].add(tool_id)
        for t in out_t:
            out_index[t].add(tool_id)

    gt_tools_canon = gt_tools
    gt_edges_canon = gt_edges

    gt_step_texts = []
    for i, tool in enumerate(gt_tools):
        if i < len(gt_step_texts_orig):
            step_text = gt_step_texts_orig[i]
        else:
            step_text = tool_descs.get(tool, tool)
        gt_step_texts.append(step_text)

    gt_edges_gnn = [(u+1, v+1) for (u, v) in gt_edges_canon]
    gt_in_degree = [0] * len(gt_tools_canon)
    for (u, v) in gt_edges_canon:
        gt_in_degree[v] += 1
    
    root_idx = None
    for i in range(len(gt_tools_canon)):
        if gt_in_degree[i] == 0:
            root_idx = i
            break
    if root_idx is not None:
        gt_edges_gnn.insert(0, (0, root_idx + 1))
    
    gt_edges_gnn = sorted(list(set(gt_edges_gnn)))
    
    items.append({
        "id": gt_ex["id"],
        "tools": gt_tools_canon,
        "edges": gt_edges_gnn,
        "y_cons": 1.0,
        "cost": 0.0,
        "node_risk": [0] * len(gt_tools_canon),
        "gap_risk_edges": [],
        "step_texts": gt_step_texts, 
        "user_request": user_request,
        "is_gt": True,
        "label_type": "gt",
        "perturb_ops": []
    })

    def io_compat(a, b):
        if a not in meta_map or b not in meta_map:
            return False
        return len(meta_map[a][1] & meta_map[b][0]) > 0
    
    def can_connect(u, t, v):
        if u is not None and not io_compat(u, t):
            return False
        if v is not None and not io_compat(t, v):
            return False
        return True

    if embedding_cache is None:
        embedding_cache = get_embedding_cache(
            cache_dir=cache_dir or "./outputs/embedding_cache",
            device=device or "cuda:0",
            lm_name=lm_name or "intfloat/e5-large"
        )
    
    tool_embs = embedding_cache.get_all_tool_embeddings()
    if not tool_embs:
        embedding_cache.precompute_tool_embeddings(tool_meta)
        tool_embs = embedding_cache.get_all_tool_embeddings()

    K = rng.choices([2, 3, 4], weights=[0.25, 0.50, 0.25], k=1)[0]
    
    generated_tools_set = set()
    generated_tools_set.add(tuple(gt_tools))
    
    perturb_count = 0
    max_attempts = K * 50
    attempt = 0
    next_inst_id = 0

    while perturb_count < K and attempt < max_attempts:
        attempt += 1

        instances = []
        for i, tool in enumerate(gt_tools):
            step_text = gt_step_texts[i] if i < len(gt_step_texts) else tool_descs.get(tool, tool)
            step_text = clean_step_texts([step_text])[0] if step_text else ""
            instances.append({
                "inst_id": next_inst_id,
                "tool": tool,
                "step": step_text,
                "is_compress_inserted": False
            })
            next_inst_id += 1
        
        applied_ops = []
        total_cost = 0.0
        node_pos_ids = set()
        gap_pos_pairs = []

        B = rng.choices([1, 2, 3], weights=[0.60, 0.30, 0.10], k=1)[0]
        desired_op_types = []
        if B == 1:
            desired_op_types = ["CONFUSION" if rng.random() < 0.5 else "MISSING"]
        elif B == 2:
            desired_op_types = ["CONFUSION", "MISSING"]
            rng.shuffle(desired_op_types)
        else:
            if rng.random() < 0.5:
                desired_op_types = ["CONFUSION", "CONFUSION", "MISSING"]
            else:
                desired_op_types = ["CONFUSION", "MISSING", "MISSING"]
            rng.shuffle(desired_op_types)
        
        def try_confusion():
            nonlocal instances, total_cost
            if len(instances) == 0 or not tool_embs:
                return False
            confusion_top_k = confusion.get('top_k', {})
            p_confuse = confusion.get('p_confuse', {})
            candidate_positions = []
            for v_idx in range(len(instances)):
                v = instances[v_idx]["tool"]
                if v in confusion_top_k and len(confusion_top_k[v]) > 0:
                    w = max(float(p_confuse.get(v, 0.0)), 1e-6)
                    candidate_positions.append((v_idx, w))
            
            if not candidate_positions:
                return False

            positions = [p[0] for p in candidate_positions]
            weights = [p[1] for p in candidate_positions]
            filtered = [(i, w) for i, w in zip(positions, weights)
                        if not instances[i].get("is_compress_inserted")
                        and instances[i]["inst_id"] not in node_pos_ids]
            if not filtered:
                return False
            positions = [p[0] for p in filtered]
            weights = [p[1] for p in filtered]
            v_idx = rng.choices(positions, weights=weights, k=1)[0]
            v = instances[v_idx]["tool"]
            
            step_text = instances[v_idx]["step"]
            try:
                step_emb = embedding_cache.encode_texts([step_text], prefix="query")[0]
            except Exception:
                step_emb = None

            candidates_A, candidates_B, candidates_C = [], [], []
            candidates_all = []
            
            prev_tool = instances[v_idx-1]["tool"] if v_idx > 0 else None
            next_tool = instances[v_idx+1]["tool"] if v_idx+1 < len(instances) else None

            v_emb = tool_embs.get(v, None) if tool_embs else None
            v_in, v_out = meta_map.get(v, (set(), set()))
            q_self = None
            if step_emb is not None and v_emb is not None:
                q_self = float(np.dot(v_emb, step_emb))
            
            for t_prime, conf_score in confusion_top_k.get(v, []):
                if t_prime == v or t_prime not in meta_map:
                    continue

                if not can_connect(prev_tool, t_prime, next_tool):
                    continue

                if step_emb is not None and t_prime in tool_embs:
                    t_emb = tool_embs[t_prime]
                    q_score = float(np.dot(t_emb, step_emb))
                else:
                    q_score = 0.0

                if v_emb is not None and t_prime in tool_embs:
                    t_prime_emb = tool_embs[t_prime]
                    s_text = float(np.dot(v_emb, t_prime_emb))
                else:
                    s_text = conf_score

                t_prime_in, t_prime_out = meta_map[t_prime]
                in_union = len(v_in | t_prime_in)
                in_intersect = len(v_in & t_prime_in)
                out_union = len(v_out | t_prime_out)
                out_intersect = len(v_out & t_prime_out)
                s_io_in = in_intersect / max(1, in_union)
                s_io_out = out_intersect / max(1, out_union)
                s_io = 0.5 * (s_io_in + s_io_out)

                high_conf = 0.56
                high_q = 0.78
                low_q = 0.50
                io_thresh = 0.30
                s_text_thresh = 0.90
                q_rel = q_score / max(q_self, 1e-6) if q_self is not None else 1.0
                q_gap = (q_self - q_score) if q_self is not None else 0.0
                q_rel_thresh = 0.92
                q_gap_thresh = 0.10
                if conf_score >= high_conf and (q_score >= high_q or (q_rel >= q_rel_thresh and q_gap <= q_gap_thresh)):
                    candidates_A.append((t_prime, conf_score, q_score))
                elif conf_score >= high_conf and (q_score < high_q or q_rel < q_rel_thresh or q_gap > q_gap_thresh):
                    candidates_B.append((t_prime, conf_score, q_score))
                elif (q_score < low_q and conf_score < high_conf and s_text < s_text_thresh):
                    candidates_C.append((t_prime, conf_score, q_score))
                
                candidates_all.append((t_prime, conf_score, q_score))

            if not candidates_C and candidates_all:
                pool_sorted = sorted(candidates_all, key=lambda x: x[2])
                take_k = max(1, int(len(pool_sorted) * 0.02))
                candidates_C.extend(pool_sorted[:take_k])

            bucket_specs = [
                ("A", candidates_A, 0.50),
                ("B", candidates_B, 0.30),
                ("C", candidates_C, 0.20),
            ]
            available = [(bid, b, w) for bid, b, w in bucket_specs if b]
            if not available:
                return False
            weights = [w for _, _, w in available]
            weights = (np.array(weights) / max(1e-12, sum(weights))).tolist()
            selected_idx = rng.choices(range(len(available)), weights=weights, k=1)[0]
            selected_bucket_id, selected_bucket, _ = available[selected_idx]

            lambda_conf = 2.0
            bucket_weights = [max(c[1], 0) * np.exp(lambda_conf * c[2]) for c in selected_bucket]
            if sum(bucket_weights) <= 0:
                return False
            bucket_weights_normalized = np.array(bucket_weights) / sum(bucket_weights)
            selected_idx = rng.choices(range(len(selected_bucket)), weights=bucket_weights_normalized.tolist(), k=1)[0]
            t_prime, final_conf_score, final_q_score = selected_bucket[selected_idx]
            final_sampling_weight = bucket_weights_normalized[selected_idx]
            
            instances[v_idx]["tool"] = t_prime
            
            q_norm = None
            if align_proj is not None and step_emb is not None:
                try:
                    step_proj = align_proj.get("step_proj")
                    tool_proj = align_proj.get("tool_proj")
                    tau = float(align_proj.get("tau", 0.07))
                    if step_proj is not None and tool_proj is not None:
                        step_t = torch.tensor(step_emb, dtype=torch.float32).unsqueeze(0)
                        tool_t = torch.tensor(tool_embs[t_prime], dtype=torch.float32).unsqueeze(0)
                        with torch.no_grad():
                            q = step_proj(step_t)
                            k = tool_proj(tool_t)
                            q = F.normalize(q, p=2, dim=-1)
                            k = F.normalize(k, p=2, dim=-1)
                            g = torch.sum(q * k, dim=-1) / max(tau, 1e-6)
                            q_norm = float(torch.sigmoid(g).item())
                except Exception:
                    q_norm = None
            if q_norm is None and step_emb is not None:
                q_norm = float(np.clip(final_q_score, 0.0, 1.0))
            if q_norm is None:
                conf_cost = ETA_CONF_BASE
            else:
                conf_cost = ETA_CONF_BASE * (1.0 - q_norm)
                conf_cost = max(0.05, float(conf_cost))
            total_cost += conf_cost

            node_pos_ids.add(instances[v_idx]["inst_id"])
            
            applied_ops.append({
                "type": "CONFUSION",
                "inst_id": instances[v_idx]["inst_id"],
                "original": v,
                "replaced": t_prime,
                "conf_score": float(final_conf_score),
                "q_score": float(final_q_score),
                "bucket_id": selected_bucket_id,
                "sampling_weight": float(final_sampling_weight),
                "cost": float(conf_cost)
            })
            return True
        
        def try_missing():
            nonlocal instances, total_cost, next_inst_id
            if len(instances) < 2:
                return False
            subtype_order = ["DROP", "COMPRESS"] if rng.random() < 0.75 else ["COMPRESS", "DROP"]
            
            for missing_subtype in subtype_order:
                if missing_subtype == "DROP":
                    m_probs = [0.55, 0.25, 0.10, 0.07, 0.03]
                    m_values = [1, 2, 3, 4, 5]
                    m = rng.choices(m_values, weights=m_probs, k=1)[0]
                    m = min(m, len(instances) - 1)
                    
                    if m < 1:
                        continue
                    
                    f_gt_motif = typed_ngrams.get('__f_gt_motif__', {})
                    f2 = typed_ngrams.get('__f2__', {})
                    f3 = typed_ngrams.get('__f3__', {})
                    f4 = typed_ngrams.get('__f4__', {})

                    use_motif_path = rng.random() < 0.75
                    
                    candidates = []
                    span_weights = []
                    
                    tools_current = [inst["tool"] for inst in instances]
                    
                    for a in range(len(instances)):
                        b = a + m
                        if b > len(instances):
                            continue
                        if any(inst["inst_id"] in node_pos_ids for inst in instances[a:b]):
                            continue
                        
                        u = tools_current[a-1] if a > 0 else None
                        v = tools_current[b] if b < len(instances) else None
                        
                        if v is None:
                            continue

                        if u is not None and not io_compat(u, v):
                            continue

                        if not use_motif_path:
                            weight = 1.0
                        elif u is None:
                            if m == 1 and a + 1 < len(tools_current):
                                freq = f2.get((tools_current[a], tools_current[a + 1]), 0)
                                weight = np.log1p(freq) + 0.1
                            elif m == 2 and a + 2 < len(tools_current):
                                freq = f3.get((tools_current[a], tools_current[a + 1], tools_current[a + 2]), 0)
                                weight = np.log1p(freq) + 0.1
                            elif m == 3 and a + 3 < len(tools_current):
                                freq = f4.get((tools_current[a], tools_current[a + 1], tools_current[a + 2], tools_current[a + 3]), 0)
                                weight = np.log1p(freq) + 0.1
                            else:
                                max_f4 = 0
                                for i in range(a, min(b - 3, len(tools_current) - 3)):
                                    key = (tools_current[i], tools_current[i + 1], tools_current[i + 2], tools_current[i + 3])
                                    max_f4 = max(max_f4, f4.get(key, 0))
                                weight = np.log1p(max_f4) + 0.1
                        else:
                            if m == 1 and a + 1 < len(tools_current) and u is not None and v is not None:
                                key3 = (u, tools_current[a], v)
                                weight = np.log1p(f3.get(key3, 0)) + 0.1
                            elif m == 2 and a + 2 < len(tools_current) and u is not None and v is not None:
                                key4 = (u, tools_current[a], tools_current[a + 1], v)
                                weight = np.log1p(f4.get(key4, 0)) + 0.1
                            else:
                                max_f4 = 0
                                if a + 2 < len(tools_current) and u is not None:
                                    key4_l = (u, tools_current[a], tools_current[a + 1], tools_current[a + 2])
                                    max_f4 = max(max_f4, f4.get(key4_l, 0))
                                if b - 3 >= 0 and v is not None:
                                    key4_r = (tools_current[b - 3], tools_current[b - 2], tools_current[b - 1], v)
                                    max_f4 = max(max_f4, f4.get(key4_r, 0))
                                freq_motif = f_gt_motif.get((u, v), 0)
                                weight = np.log1p(max(max_f4, freq_motif)) + 0.1
                        
                        candidates.append((a, b, m))
                        span_weights.append(weight)
                    
                    if candidates and sum(span_weights) > 0:
                        span_weights_arr = np.array(span_weights) / sum(span_weights)
                        a_sel, b_sel, m_sel = rng.choices(candidates, weights=span_weights_arr.tolist(), k=1)[0]
                        
                        deleted_insts = instances[a_sel:b_sel]

                        u_inst_id = None if a_sel == 0 else instances[a_sel-1]["inst_id"]

                        instances = instances[:a_sel] + instances[b_sel:]

                        v_inst_id = instances[a_sel]["inst_id"] if a_sel < len(instances) else None

                        if v_inst_id is not None:
                            gap_pos_pairs.append((u_inst_id, v_inst_id))
                        for del_inst in deleted_insts:
                            if del_inst["inst_id"] in node_pos_ids:
                                node_pos_ids.discard(del_inst["inst_id"])

                        cost_len = ETA_DROP_LEN * m_sel
                        cost_rel = 0.0

                        if user_request:
                            try:
                                req_emb = embedding_cache.encode_texts([user_request], prefix="query")[0]
                                for del_inst in deleted_insts:
                                    del_tool = del_inst["tool"]
                                    if del_tool in tool_embs:
                                        t_emb = tool_embs[del_tool]
                                        cos_sim = float(np.dot(t_emb, req_emb))
                                        tilde_r = (cos_sim + 1.0) / 2.0
                                        cost_rel += max(0.0, tilde_r)
                            except Exception:
                                pass
                        
                        cost_rel *= ETA_DROP_REL
                        cost = cost_len + cost_rel
                        total_cost += cost
                        
                        applied_ops.append({
                            "type": "MISSING",
                            "subtype": "DROP",
                            "deleted": [inst["tool"] for inst in deleted_insts],
                            "boundary": (u_inst_id, v_inst_id),
                            "cost": float(cost)
                        })
                        return True
            
                elif missing_subtype == "COMPRESS" and len(instances) >= 3:
                    candidates = []
                    span_weights = []
                    
                    tools_current = [inst["tool"] for inst in instances]
                    
                    for span_len in range(2, min(5, len(instances))):
                        for a in range(len(instances) - span_len):
                            b_idx = a + span_len
                            if any(inst["inst_id"] in node_pos_ids for inst in instances[a:b_idx]):
                                continue
                            u = tools_current[a-1] if a > 0 else None
                            v = tools_current[b_idx] if b_idx < len(instances) else None
                            
                            if v is None:
                                continue
                            
                            if u is None:
                                v_in = meta_map.get(v, (set(), set()))[0]
                                cand_out = set()
                                for t in v_in:
                                    cand_out.update(out_index.get(t, set()))
                                shortcut_candidates = list(cand_out)
                            else:
                                u_out = meta_map.get(u, (set(), set()))[1]
                                v_in = meta_map.get(v, (set(), set()))[0]
                                cand_in = set()
                                for t in u_out:
                                    cand_in.update(in_index.get(t, set()))
                                cand_out = set()
                                for t in v_in:
                                    cand_out.update(out_index.get(t, set()))
                                shortcut_candidates = list(cand_in & cand_out)
                            
                            if shortcut_candidates:
                                shortcut_candidates = [t_star for t_star in shortcut_candidates if can_connect(u, t_star, v)]
                            
                            if shortcut_candidates:
                                candidates.append((a, b_idx, span_len, shortcut_candidates))
                                span_weights.append(1.0)
                    
                    if candidates and sum(span_weights) > 0:
                        span_weights_arr = np.array(span_weights) / sum(span_weights)
                        a, b_idx, span_len, shortcut_cands = rng.choices(candidates, weights=span_weights_arr.tolist(), k=1)[0]
                        
                        deleted_insts = instances[a:b_idx]
                        span_steps = clean_step_texts([inst["step"] for inst in deleted_insts])
                        s_span = " ".join([s for s in span_steps if s])
                        
                        s_span_emb = None
                        if len(shortcut_cands) > 1:
                            try:
                                s_span_emb = embedding_cache.encode_texts([s_span], prefix="query")[0]
                                
                                lambda_cmp = 2.0
                                t_star_weights = []
                                for t_cand in shortcut_cands:
                                    t_emb = tool_embs.get(t_cand)
                                    if t_emb is None:
                                        t_desc = tool_descs.get(t_cand, t_cand)
                                        t_emb = embedding_cache.encode_texts([t_desc], prefix="passage")[0]
                                        tool_embs[t_cand] = t_emb
                                    q_proxy = float(np.dot(t_emb, s_span_emb))
                                    t_star_weights.append(np.exp(lambda_cmp * q_proxy))
                                
                                t_star_weights = np.array(t_star_weights) / sum(t_star_weights)
                                t_star = rng.choices(shortcut_cands, weights=t_star_weights.tolist(), k=1)[0]
                            except Exception:
                                t_star = rng.choice(shortcut_cands)
                        else:
                            t_star = rng.choice(shortcut_cands)
                        
                        u_inst_id = None if a == 0 else instances[a-1]["inst_id"]
                        v_boundary_inst_id = instances[b_idx]["inst_id"] if b_idx < len(instances) else None
                        
                        new_inst = {
                            "inst_id": next_inst_id,
                            "tool": t_star,
                            "step": s_span,
                            "is_compress_inserted": True
                        }
                        next_inst_id += 1
                        
                        instances = instances[:a] + [new_inst] + instances[b_idx:]
                        
                        node_pos_ids.add(new_inst["inst_id"])
                        if u_inst_id is None:
                            gap_pos_pairs.append((None, new_inst["inst_id"]))
                        else:
                            gap_pos_pairs.append((u_inst_id, new_inst["inst_id"]))
                        if v_boundary_inst_id is not None:
                            gap_pos_pairs.append((new_inst["inst_id"], v_boundary_inst_id))
                        
                        cost_len = ETA_CMP_LEN * (span_len - 1)
                        cost_cover = 0.0
                        
                        try:
                            if s_span_emb is None:
                                s_span_emb = embedding_cache.encode_texts([s_span], prefix="query")[0]
                            
                            t_emb = tool_embs.get(t_star)
                            if t_emb is None:
                                t_desc = tool_descs.get(t_star, t_star)
                                t_emb = embedding_cache.encode_texts([t_desc], prefix="passage")[0]
                                tool_embs[t_star] = t_emb
                            cos_sim = float(np.dot(t_emb, s_span_emb))
                            tilde_q = (cos_sim + 1.0) / 2.0
                            cost_cover = ETA_CMP_COVER * (1.0 - tilde_q)
                        except Exception:
                            cost_cover = ETA_CMP_COVER * 0.5
                        
                        cost = cost_len + cost_cover
                        total_cost += cost
                        
                        applied_ops.append({
                            "type": "MISSING",
                            "subtype": "COMPRESS",
                            "deleted": [inst["tool"] for inst in deleted_insts],
                            "inserted": t_star,
                            "insert_inst_id": new_inst["inst_id"],
                            "boundary": (u_inst_id, v_boundary_inst_id),
                            "cost": float(cost)
                        })
                        return True
            
            return False
        
        for op_type in desired_op_types:
            op_success = False
            if op_type == "CONFUSION":
                op_success = try_confusion()
                if not op_success:
                    op_success = try_missing() or try_missing()
            else:
                op_success = try_missing() or try_missing()
                if not op_success:
                    op_success = try_confusion()
            if not op_success:
                break
        
        if not applied_ops or len(instances) == 0:
            continue
        if len(applied_ops) != B:
            continue
        
        tools = [inst["tool"] for inst in instances]
        step_texts = clean_step_texts([inst["step"] for inst in instances])
        
        tools_tuple = tuple(tools)
        if tools_tuple in generated_tools_set:
            continue
        generated_tools_set.add(tools_tuple)
        
        edges = [(i, i+1) for i in range(len(tools) - 1)] if len(tools) > 1 else []
        tools_canon = tools
        edges_canon = edges
        instances_canon = instances
        
        inst_id_to_pos = {}
        for pos, inst in enumerate(instances_canon):
            inst_id_to_pos[inst["inst_id"]] = pos
        
        valid_ops = []
        total_cost = 0.0
        for op in applied_ops:
            if op.get("type") == "CONFUSION":
                inst_id = op.get("inst_id")
                if inst_id not in inst_id_to_pos:
                    continue
            valid_ops.append(op)
            total_cost += float(op.get("cost", 0.0))
        applied_ops = valid_ops

        if not applied_ops:
            continue

        node_pos_ids = {i for i in node_pos_ids if i in inst_id_to_pos}

        node_risk = [0] * len(tools_canon)
        for pos, inst in enumerate(instances_canon):
            if inst["inst_id"] in node_pos_ids:
                node_risk[pos] = 1
        
        gap_risk_edges_set = set()
        for (u_inst_id, v_inst_id) in gap_pos_pairs:
            if u_inst_id is None:
                u_gnn = 0
            elif u_inst_id in inst_id_to_pos:
                u_gnn = inst_id_to_pos[u_inst_id] + 1
            else:
                continue
            
            if v_inst_id is not None and v_inst_id in inst_id_to_pos:
                v_gnn = inst_id_to_pos[v_inst_id] + 1
            else:
                continue
            
            gap_risk_edges_set.add((u_gnn, v_gnn))
        
        gap_risk_edges = sorted(gap_risk_edges_set)
        
        cost = total_cost
        y_cons = np.exp(-total_cost / TAU_COST)
        if total_cost < 0.01:
            continue
        
        step_texts_canon = clean_step_texts([inst["step"] for inst in instances_canon])
        
        edges_gnn = [(u+1, v+1) for (u, v) in edges_canon]
        
        in_degree_canon = [0] * len(tools_canon)
        for (u, v) in edges_canon:
            in_degree_canon[v] += 1
        
        root_idx = None
        for i in range(len(tools_canon)):
            if in_degree_canon[i] == 0:
                root_idx = i
                break
        if root_idx is not None:
            edges_gnn.insert(0, (0, root_idx + 1))
        
        edges_gnn = sorted(list(set(edges_gnn)))
        
        items.append({
            "id": gt_ex["id"],
            "tools": tools_canon,
            "edges": edges_gnn,
            "y_cons": y_cons,
            "cost": cost, 
            "node_risk": node_risk,
            "gap_risk_edges": gap_risk_edges,
            "step_texts": step_texts_canon,
            "user_request": user_request,
            "is_gt": False,
            "label_type": "bad",
            "perturb_ops": applied_ops
        })
        perturb_count += 1

    return items
