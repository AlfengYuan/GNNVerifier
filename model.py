import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import random
import re
import sys
sys.path.append("../")
from gnn import GNNEncoder
from utils import init_random_state


class GraphConsistencyModel(nn.Module):
    def __init__(self, lm_dim, num_io_types, hidden_dim, n_layers, dropout=0.1,
                 align_dim=None, align_tau=0.07):
        super().__init__()

        self.lm_dim = lm_dim
        self.hidden_dim = hidden_dim
        self.num_io_types = num_io_types
        self.align_dim = int(align_dim) if align_dim is not None else lm_dim
        self.align_tau = float(align_tau)

        self.step_proj = nn.Linear(lm_dim, self.align_dim, bias=False)
        self.tool_proj = nn.Linear(lm_dim, self.align_dim, bias=False)

        proj_input_dim = lm_dim + num_io_types * 2 + lm_dim + 1
        self.proj_mlp = nn.Sequential(
            nn.Linear(proj_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.start_node_embedding = nn.Parameter(torch.randn(hidden_dim))

        edge_feat_input_dim = 3
        edge_feat_output_dim = hidden_dim // 4
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feat_input_dim, edge_feat_output_dim),
            nn.ReLU()
        )

        self.start_edge_embedding_fwd = nn.Parameter(torch.randn(edge_feat_output_dim))
        self.start_edge_embedding_bwd = nn.Parameter(torch.randn(edge_feat_output_dim))

        self.gnn = GNNEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            edge_attr_dim=edge_feat_output_dim,
            req_dim=lm_dim
        )

        readout_dim = hidden_dim * 2
        cons_input_dim = readout_dim + lm_dim
        self.cons_head = nn.Sequential(
            nn.Linear(cons_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        node_risk_input_dim = hidden_dim + lm_dim
        self.node_risk_head = nn.Sequential(
            nn.Linear(node_risk_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        gap_risk_input_dim = hidden_dim * 2 + edge_feat_output_dim + lm_dim
        self.gap_risk_head = nn.Sequential(
            nn.Linear(gap_risk_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def build_instance_node_features(self, tool_emb, step_emb, x_in_IO_list, x_out_IO_list,
                                     align_margin=None):
        num_nodes = tool_emb.size(0)
        device = tool_emb.device

        x_in_IO_tensor = torch.tensor(x_in_IO_list, dtype=torch.float32, device=device)
        x_out_IO_tensor = torch.tensor(x_out_IO_list, dtype=torch.float32, device=device)

        if align_margin is None:
            align_margin = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        align_margin = align_margin.view(-1, 1)

        proj_input = torch.cat([
            tool_emb,
            x_in_IO_tensor,
            x_out_IO_tensor,
            step_emb,
            align_margin
        ], dim=-1)

        node_feats = self.proj_mlp(proj_input)

        return node_feats

    def forward(self, node_feats, edge_index, req_emb, edge_feats=None, gaps=None, gap_edge_feats=None):
        num_nodes = node_feats.size(0)
        device = node_feats.device

        if num_nodes < 1:
            return {
                "S": torch.tensor([0.0], device=device),
                "S_logit": torch.tensor([0.0], device=device),
                "node_risks": torch.zeros(0, device=device),
                "node_logits": torch.zeros(0, device=device),
                "gap_risks": torch.zeros(0, device=device),
                "gap_logits": torch.zeros(0, device=device),
                "node_h": node_feats,
                "h_graph": torch.zeros(1, self.hidden_dim * 2, device=device),
            }

        if edge_feats is not None and edge_index.size(1) > 0:
            assert edge_feats.size(0) == edge_index.size(1),\
                f"edge_feats size {edge_feats.size(0)} != edge_index size {edge_index.size(1)}"

        node_h = self.gnn(node_feats, edge_index, edge_feats, req_emb)

        num_tools = num_nodes - 1
        if num_tools > 0:
            tool_node_h = node_h[1:]
            h_mean_tools = tool_node_h.mean(dim=0, keepdim=True)
            h_max_tools = tool_node_h.max(dim=0, keepdim=True)[0]
        else:
            h_mean_tools = torch.zeros(1, self.hidden_dim, device=device)
            h_max_tools = torch.zeros(1, self.hidden_dim, device=device)

        h_G_tools = torch.cat([h_mean_tools, h_max_tools], dim=-1)


        if req_emb.dim() == 1:
            req_emb_expanded = req_emb.unsqueeze(0)
        else:
            req_emb_expanded = req_emb

        cons_input = torch.cat([h_G_tools, req_emb_expanded], dim=-1)
        S_logit = self.cons_head(cons_input).view(-1)

        S = torch.sigmoid(S_logit)

        if num_tools > 0:
            tool_node_h = node_h[1:]
            req_emb_for_nodes = req_emb_expanded.expand(num_tools, -1)
            node_input = torch.cat([tool_node_h, req_emb_for_nodes], dim=-1)
            node_logits_raw = self.node_risk_head(node_input).view(-1)
            node_risks = torch.sigmoid(node_logits_raw)
        else:
            node_logits_raw = torch.zeros(0, device=device)
            node_risks = torch.zeros(0, device=device)

        if not gaps:
            gap_logits_raw = torch.zeros(0, device=device)
            gap_risks = torch.zeros(0, device=device)
        else:
            if gap_edge_feats is None:
                raise ValueError("gap_edge_feats must be provided when gaps is non-empty")
            if len(gaps) != gap_edge_feats.size(0):
                raise ValueError(f"len(gaps)={len(gaps)} != gap_edge_feats.size(0)={gap_edge_feats.size(0)}")

            u_idx = torch.tensor([u for u, _ in gaps], dtype=torch.long, device=device)
            v_idx = torch.tensor([v for _, v in gaps], dtype=torch.long, device=device)
            valid_mask = (u_idx < num_nodes) & (v_idx < num_nodes)

            gap_logits_raw = torch.zeros(len(gaps), device=device)
            if valid_mask.any():
                h_u = node_h[u_idx[valid_mask]]
                h_v = node_h[v_idx[valid_mask]]
                req = req_emb_expanded.expand(h_u.size(0), -1)
                e_uv = gap_edge_feats[valid_mask]
                gap_input = torch.cat([h_u, h_v, e_uv, req], dim=-1)
                gap_logits_raw[valid_mask] = self.gap_risk_head(gap_input).view(-1)

            gap_risks = torch.sigmoid(gap_logits_raw)

        return {
            "S": S,
            "S_logit": S_logit,
            "node_risks": node_risks,
            "node_logits": node_logits_raw,
            "gap_risks": gap_risks,
            "gap_logits": gap_logits_raw,
            "node_h": node_h,
            "h_graph": h_G_tools,
        }

class ModelTrainer:
    def __init__(self, args, device, tool_meta, confusion=None, typed_ngrams=None,
                 IO_TYPE2IDX=None, num_io_types=0):
        self.seed = args.seed
        self.device = device
        self.tool_meta = tool_meta
        self.confusion = confusion or {}
        self.typed_ngrams = typed_ngrams or {}
        self.IO_TYPE2IDX = IO_TYPE2IDX or {}
        self.num_io_types = num_io_types

        init_random_state(args.seed)

        from embedding_cache import get_embedding_cache
        self._embedding_cache = get_embedding_cache(
            cache_dir=f"./outputs/{args.dataset}/embedding_cache",
            device=str(device),
            lm_name=args.lm_name
        )
        self._embedding_cache.ensure_model_loaded()
        lm_dim = int(self._embedding_cache.lm_dim)
        self.req_dim = lm_dim
        self._req_emb_cache = {}

        if not self.IO_TYPE2IDX:
            io_types = set()
            for n in tool_meta["nodes"]:
                in_t = n.get("input-type", [])
                out_t = n.get("output-type", [])
                if not isinstance(in_t, list):
                    in_t = [in_t] if in_t else []
                if not isinstance(out_t, list):
                    out_t = [out_t] if out_t else []
                io_types.update(in_t)
                io_types.update(out_t)
            io_type_list = sorted(list(io_types))
            self.IO_TYPE2IDX = {t: i for i, t in enumerate(io_type_list)}
            self.num_io_types = len(io_types)

        self.tool_io_multihot = {}
        if self.IO_TYPE2IDX:
            for node in tool_meta["nodes"]:
                tool_id = node["id"]
                x_in = [0] * self.num_io_types
                x_out = [0] * self.num_io_types
                in_t = node.get("input-type", [])
                out_t = node.get("output-type", [])
                if not isinstance(in_t, list):
                    in_t = [in_t] if in_t else []
                if not isinstance(out_t, list):
                    out_t = [out_t] if out_t else []
                for t in in_t:
                    if isinstance(t, str) and t in self.IO_TYPE2IDX:
                        x_in[self.IO_TYPE2IDX[t]] = 1
                for t in out_t:
                    if isinstance(t, str) and t in self.IO_TYPE2IDX:
                        x_out[self.IO_TYPE2IDX[t]] = 1
                self.tool_io_multihot[tool_id] = (x_in, x_out)

        self.tool_map = {}
        for node in tool_meta["nodes"]:
            tool_id = node["id"]
            in_t = node.get("input-type", [])
            out_t = node.get("output-type", [])
            if not isinstance(in_t, list):
                in_t = [in_t] if in_t else []
            if not isinstance(out_t, list):
                out_t = [out_t] if out_t else []
            self.tool_map[tool_id] = {
                "desc": node.get("desc", tool_id),
                "in_types": set(in_t),
                "out_types": set(out_t)
            }

        self._tool_embeddings = {}
        self._precompute_tool_embeddings()

        model = GraphConsistencyModel(
            lm_dim=lm_dim,
            num_io_types=self.num_io_types,
            hidden_dim=args.gnn_hidden_dim,
            n_layers=args.gnn_layer,
            dropout=0.1,
            align_dim=getattr(args, "align_dim", lm_dim),
            align_tau=getattr(args, "align_tau", 0.07)
        )
        self.model = model.to(device)

        self.margin_rank = args.margin_rank
        self.lambda_rank = args.lambda_rank
        self.lambda_graph = args.lambda_graph
        self.lambda_gap = args.lambda_gap
        self.node_pos_weight = 3.0
        self.gap_pos_weight = 8.0
        self.cost_tau = float(getattr(args, "cost_tau", 0.8))
        self.align_tau = float(getattr(args, "align_tau", 0.07))
        self.align_hard_k = int(getattr(args, "align_hard_k", 5))
        self.align_rand_k = int(getattr(args, "align_rand_k", 10))
        self.align_lr = float(getattr(args, "align_lr", 1e-4))
        self.align_batch = int(getattr(args, "align_batch", 64))
        self.align_patience = int(getattr(args, "align_patience", 5))
        self.align_min_delta = float(getattr(args, "align_min_delta", 0.002))
        self.base_lr = float(args.lr)
        self.align_pretrained = False
        self.align_pretrain_epochs_default = 5

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0.0)

    def _precompute_tool_embeddings(self):
        tool_descs = []
        tool_ids = []
        for tool_id, info in self.tool_map.items():
            tool_ids.append(tool_id)
            tool_descs.append(info["desc"])

        if tool_descs:
            embeddings = self._encode_text_to_emb_cached(tool_descs, prefix="passage")
            for i, tool_id in enumerate(tool_ids):
                self._tool_embeddings[tool_id] = embeddings[i]

    def _get_tool_embedding(self, tool_id):
        if not tool_id:
            return torch.zeros(self.req_dim, device=self.device)
        if tool_id in self._tool_embeddings:
            return self._tool_embeddings[tool_id]

        if tool_id in self.tool_map:
            desc = self.tool_map[tool_id]["desc"]
        else:
            desc = tool_id
        if not desc:
            desc = tool_id if tool_id else "unknown"
        emb = self._encode_text_to_emb_cached([desc])[0]
        self._tool_embeddings[tool_id] = emb
        return emb

    def encode_request(self, text: str) -> torch.Tensor:
        if not text:
            return torch.zeros(self.req_dim, device=self.device)
        try:
            from embedding_cache import normalize_text
            key = normalize_text(text)
            if key in self._req_emb_cache:
                return self._req_emb_cache[key]
            emb = self._encode_text_to_emb_cached([text], prefix="query")
            self._req_emb_cache[key] = emb[0]
            return emb[0]
        except Exception:
            return torch.zeros(self.req_dim, device=self.device)

    def _encode_text_to_emb_cached(self, texts, prefix="passage"):
        if isinstance(texts, str):
            texts = [texts]
        return self._embedding_cache.encode_texts_tensor(texts, device=self.device, prefix=prefix)

    def _project_step(self, step_emb):
        q = self.model.step_proj(step_emb)
        return F.normalize(q, p=2, dim=-1)

    def _project_tool(self, tool_emb):
        k = self.model.tool_proj(tool_emb)
        return F.normalize(k, p=2, dim=-1)

    def _compute_align_margin(self, tool_ids, step_emb):
        if not tool_ids:
            return torch.zeros(0, device=self.device)
        if step_emb is None or step_emb.numel() == 0:
            return torch.zeros(len(tool_ids), device=self.device)

        q = self._project_step(step_emb)
        align_margin = torch.zeros(len(tool_ids), device=self.device)
        conf_topk = self.confusion.get("top_k", {}) if self.confusion else {}
        tau = float(self.model.align_tau)

        for i, tool_id in enumerate(tool_ids):
            t_emb = self._get_tool_embedding(tool_id)
            k_pos = self._project_tool(t_emb.unsqueeze(0)).squeeze(0)
            a_i = torch.dot(q[i], k_pos) / max(tau, 1e-6)

            candidates = [t for t, _ in conf_topk.get(tool_id, []) if t != tool_id]
            if not candidates:
                align_margin[i] = 0.0
                continue

            cand_embs = torch.stack([self._get_tool_embedding(t) for t in candidates], dim=0)
            k_cand = self._project_tool(cand_embs)
            scores = torch.matmul(k_cand, q[i]) / max(tau, 1e-6)
            a_star = torch.max(scores) if scores.numel() > 0 else torch.tensor(0.0, device=self.device)
            align_margin[i] = a_i - a_star

        return align_margin

    def build_graph_inputs(self, chain, step_texts=None, user_request=None, edges=None,
                           IO_TYPE2IDX=None):

        req_emb = self.encode_request(user_request) if user_request else torch.zeros(self.req_dim, device=self.device)

        if not chain or len(chain) == 0:
            start_feat = self.model.start_node_embedding.unsqueeze(0)
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            edge_dim = self.model.start_edge_embedding_fwd.shape[0]
            edge_feats = torch.empty((0, edge_dim), device=self.device)
            gap_edge_feats = torch.empty((0, edge_dim), device=self.device)
            return start_feat, edge_index, req_emb, edge_feats, [], gap_edge_feats

        num_tools = len(chain)

        tool_descs = []
        x_in_IO_list = []
        x_out_IO_list = []

        if IO_TYPE2IDX is None:
            IO_TYPE2IDX = self.IO_TYPE2IDX
        num_io_types = len(IO_TYPE2IDX) if IO_TYPE2IDX else self.num_io_types

        for tool_id in chain:
            if tool_id in self.tool_map:
                tool_info = self.tool_map[tool_id]
                desc = tool_info["desc"]
                tool_descs.append(desc)

                in_t = list(tool_info["in_types"])
                out_t = list(tool_info["out_types"])

                if IO_TYPE2IDX == self.IO_TYPE2IDX and tool_id in self.tool_io_multihot:
                    x_in_IO, x_out_IO = self.tool_io_multihot[tool_id]
                    x_in_IO_list.append(x_in_IO)
                    x_out_IO_list.append(x_out_IO)
                else:

                    from utils_preproc import build_io_multihot
                    x_in_IO, x_out_IO = build_io_multihot(tool_id, self.tool_meta, IO_TYPE2IDX, num_io_types)
                    x_in_IO_list.append(x_in_IO)
                    x_out_IO_list.append(x_out_IO)
            else:
                tool_descs.append(tool_id)
                x_in_IO_list.append([0] * num_io_types)
                x_out_IO_list.append([0] * num_io_types)

        if step_texts is None or len(step_texts) != num_tools:
            step_texts_list = tool_descs
        else:
            step_texts_list = step_texts

        cleaned = []
        for s in step_texts_list:
            if not s:
                cleaned.append("")
                continue
            t = str(s).strip()
            t = re.sub(r"^\s*Step\s*\d+\s*:\s*", "", t, flags=re.IGNORECASE)
            t = re.sub(r"^\s*\d+\s*:\s*", "", t)
            t = " ".join(t.split())
            cleaned.append(t)
        step_texts_list = cleaned

        tool_emb_list = [self._get_tool_embedding(tid) for tid in chain]
        tool_emb = torch.stack(tool_emb_list, dim=0)
        step_emb = self._encode_text_to_emb_cached(step_texts_list, prefix="query")
        align_margin = None
        if self.align_pretrained:
            align_margin = self._compute_align_margin(chain, step_emb)

        if edges is None:
            edges_gnn = [(i+1, i+2) for i in range(len(chain)-1)] if len(chain) > 1 else []
        else:
            edges_gnn = list(edges)

        num_nodes = num_tools + 1
        edge_list_forward = []
        for u, v in edges_gnn:
            if not isinstance(u, int) or not isinstance(v, int):
                continue
            if 0 <= u < num_nodes and 0 <= v < num_nodes and u != v:
                edge_list_forward.append((u, v))

        if not any(u == 0 for u, _ in edge_list_forward):
            in_degree = [0] * num_tools
            for u, v in edge_list_forward:
                if u > 0 and v > 0:
                    in_degree[v - 1] += 1
            root_idx = None
            for i in range(num_tools):
                if in_degree[i] == 0:
                    root_idx = i
                    break
            if root_idx is not None:
                edge_list_forward.append((0, root_idx + 1))

        edge_list_forward = sorted(set(edge_list_forward))

        tool_node_feats = self.model.build_instance_node_features(
            tool_emb, step_emb,
            x_in_IO_list, x_out_IO_list,
            align_margin=align_margin
        )

        start_node_feat = self.model.start_node_embedding.unsqueeze(0)
        node_feats = torch.cat([start_node_feat, tool_node_feats], dim=0)

        gaps = edge_list_forward
        edge_list_all = [[u, v] for u, v in edge_list_forward] + [[v, u] for u, v in edge_list_forward]
        edge_index = torch.tensor(edge_list_all, dtype=torch.long).t().contiguous().to(self.device)

        f2 = self.typed_ngrams.get('__f2__', {})
        f_gt_motif = self.typed_ngrams.get('__f_gt_motif__', {})

        edge_feat_list = []
        is_start_edge = []

        for edge_idx, (u_idx, v_idx) in enumerate(edge_list_forward):
            is_start = (u_idx == 0)
            is_start_edge.append(is_start)

            if u_idx == 0:
                compat_score = 0.0
                log_path_freq = 0.0
                f_motif = 0.0
            else:
                u_tool = chain[u_idx - 1]
                v_tool = chain[v_idx - 1]

                x_out_u = x_out_IO_list[u_idx - 1]
                x_in_v = x_in_IO_list[v_idx - 1]
                inter = sum([a * b for a, b in zip(x_out_u, x_in_v)])
                denom = max(sum(x_in_v), 1)
                compat_score = inter / denom

                path_freq = f2.get((u_tool, v_tool), 0)
                log_path_freq = np.log1p(path_freq)

                f_motif = f_gt_motif.get((u_tool, v_tool), 0)

            edge_feat = [compat_score, log_path_freq, f_motif]
            edge_feat_list.append(edge_feat)

        for edge_idx in range(len(edge_list_forward)):
            edge_feat_list.append(edge_feat_list[edge_idx])
            is_start_edge.append(is_start_edge[edge_idx])

        edge_feats_raw = torch.tensor(edge_feat_list, dtype=torch.float32).to(self.device)
        edge_feats_mlp = self.model.edge_mlp(edge_feats_raw)

        num_forward = len(edge_list_forward)
        edge_feats = edge_feats_mlp.clone()
        for i, is_start in enumerate(is_start_edge):
            if is_start:
                if i < num_forward:
                    edge_feats[i] = self.model.start_edge_embedding_fwd
                else:
                    edge_feats[i] = self.model.start_edge_embedding_bwd

        gap_edge_feats_list = []
        for (u_idx, v_idx) in gaps:
            if u_idx == 0:
                gap_edge_feats_list.append(self.model.start_edge_embedding_fwd)
            elif u_idx >= len(chain) + 1 or v_idx >= len(chain) + 1:

                edge_dim = self.model.start_edge_embedding_fwd.shape[0]
                gap_edge_feats_list.append(torch.zeros(edge_dim, device=self.device))
            else:
                u_tool = chain[u_idx - 1]
                v_tool = chain[v_idx - 1]

                x_out_u = x_out_IO_list[u_idx - 1]
                x_in_v = x_in_IO_list[v_idx - 1]
                inter = sum([a * b for a, b in zip(x_out_u, x_in_v)])
                denom = max(sum(x_in_v), 1)
                compat_score = inter / denom

                path_freq = f2.get((u_tool, v_tool), 0)
                log_path_freq = np.log1p(path_freq)

                f_motif_score = f_gt_motif.get((u_tool, v_tool), 0)

                gap_edge_raw = torch.tensor([compat_score, log_path_freq, f_motif_score],
                                           dtype=torch.float32, device=self.device)
                gap_edge_feat = self.model.edge_mlp(gap_edge_raw)
                gap_edge_feats_list.append(gap_edge_feat)

        if gap_edge_feats_list:
            gap_edge_feats = torch.stack(gap_edge_feats_list, dim=0)
        else:
            edge_dim = self.model.start_edge_embedding_fwd.shape[0]
            gap_edge_feats = torch.zeros(0, edge_dim, device=self.device)

        return node_feats, edge_index, req_emb, edge_feats, gaps, gap_edge_feats

    def train(self, train_data, num_epochs=10, stage1_epochs=None, val_items=None, patience=5, min_delta=0.002):

        if stage1_epochs is None:
            stage1_epochs = max(1, int(num_epochs * 0.7))

        if not self.align_pretrained:
            self.train_alignment_from_raw(train_data, num_epochs=self.align_pretrain_epochs_default)

        _, best_loss_stage1 = self._train_stage(
            train_data, stage1_epochs, stage=1, stage1_epochs=stage1_epochs,
            val_items=val_items, patience=patience, min_delta=min_delta
        )

        for p in self.model.parameters():
            p.requires_grad = False

        encoder_params = []
        if self.model.gnn.layers:
            for p in self.model.gnn.layers[-1].parameters():
                p.requires_grad = True
                encoder_params.append(p)
        for p in self.model.gnn.lin_out.parameters():
            p.requires_grad = True
            encoder_params.append(p)

        diag_heads_params = []
        for p in self.model.node_risk_head.parameters():
            p.requires_grad = True
            diag_heads_params.append(p)
        for p in self.model.gap_risk_head.parameters():
            p.requires_grad = True
            diag_heads_params.append(p)

        original_lr = self.optimizer.param_groups[0]['lr']
        lr_head = original_lr * 0.1
        lr_enc = original_lr * 0.05
        self.optimizer = torch.optim.AdamW(
            [
                {"params": diag_heads_params, "lr": lr_head},
                {"params": encoder_params, "lr": lr_enc},
            ],
            weight_decay=0.0
        )

        self.model.train()

        _, best_loss_stage2 = self._train_stage(
            train_data, num_epochs - stage1_epochs, stage=2, stage1_epochs=stage1_epochs,
            val_items=val_items, patience=patience, min_delta=min_delta
        )

        return self.model, best_loss_stage2

    def train_alignment_from_raw(self, raw_train_data, num_epochs=5):
        if num_epochs <= 0:
            return

        rng = random.Random(self.seed)

        pairs = self._collect_align_pairs(raw_train_data)

        if not pairs:
            return

        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.step_proj.parameters():
            p.requires_grad = True
        for p in self.model.tool_proj.parameters():
            p.requires_grad = True

        optimizer = torch.optim.Adam(
            list(self.model.step_proj.parameters()) + list(self.model.tool_proj.parameters()),
            lr=self.align_lr
        )

        tool_ids = list(self.tool_map.keys())
        conf_topk = self.confusion.get("top_k", {}) if self.confusion else {}
        tau = float(self.model.align_tau)

        best_loss = float("inf")
        patience_cnt = 0
        best_state = None

        for epoch in range(num_epochs):
            rng.shuffle(pairs)
            total_loss = 0.0
            step_count = 0
            batch_loss = 0.0

            for step_text, tool_id in pairs:
                if not step_text or not tool_id:
                    continue

                hard = [t for t, _ in conf_topk.get(tool_id, []) if t != tool_id]
                hard = hard[: self.align_hard_k]
                neg_pool = [t for t in tool_ids if t != tool_id and t not in hard]
                rand = rng.sample(neg_pool, k=min(self.align_rand_k, len(neg_pool)))
                candidates = [tool_id] + hard + rand

                step_emb = self._encode_text_to_emb_cached([step_text], prefix="query")
                tool_embs = torch.stack([self._get_tool_embedding(t) for t in candidates], dim=0)

                q = self._project_step(step_emb)
                k = self._project_tool(tool_embs)
                logits = torch.matmul(q, k.t()).view(-1) / max(tau, 1e-6)
                target = torch.tensor([0], dtype=torch.long, device=self.device)

                loss = F.cross_entropy(logits.view(1, -1), target)
                batch_loss = batch_loss + loss
                step_count += 1

                if step_count % self.align_batch == 0:
                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()
                    total_loss += batch_loss.item()
                    batch_loss = 0.0

            if step_count % self.align_batch != 0 and batch_loss is not None:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                total_loss += batch_loss.item()

            avg_loss = total_loss / max(1, step_count)

            if avg_loss + self.align_min_delta < best_loss:
                best_loss = avg_loss
                patience_cnt = 0
                best_state = {
                    "step_proj": {k: v.detach().cpu().clone() for k, v in self.model.step_proj.state_dict().items()},
                    "tool_proj": {k: v.detach().cpu().clone() for k, v in self.model.tool_proj.state_dict().items()},
                }
            else:
                patience_cnt += 1
                if patience_cnt >= self.align_patience:
                    break

        if best_state is not None:
            self.model.step_proj.load_state_dict(best_state["step_proj"])
            self.model.tool_proj.load_state_dict(best_state["tool_proj"])

        for p in self.model.parameters():
            p.requires_grad = True
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.base_lr, weight_decay=0.0)
        self.align_pretrained = True

    def _collect_align_pairs(self, raw_train_data):
        from utils_preproc import order_chain_with_steps_and_edges, clean_step_texts
        pairs = []
        for ex in raw_train_data:
            ex_id = ex.get("id") or ex.get("example_id")
            tools_raw = [n.get("task") if isinstance(n, dict) else n for n in ex.get("task_nodes", [])]
            steps = ex.get("task_steps", [])
            links = ex.get("task_links", [])
            if not tools_raw and ex.get("tools"):
                tools_raw = ex.get("tools", [])
                steps = ex.get("step_texts", []) or []
                edges = ex.get("edges", [])
                links = [{"source": tools_raw[u], "target": tools_raw[v]} for (u, v) in (edges or [])
                    if isinstance(u, int) and isinstance(v, int) and 0 <= u < len(tools_raw) and 0 <= v < len(tools_raw)]
            if not tools_raw or not steps:
                continue
            tools, steps, _ = order_chain_with_steps_and_edges(tools_raw, steps, links)
            steps = clean_step_texts(steps) if steps is not None else steps
            if steps is None or len(steps) != len(tools):
                continue
            for i in range(len(tools)):
                pairs.append((steps[i], tools[i]))
        return pairs

    def _auc_from_scores(self, y_true, y_score):
        if not y_true:
            return None
        pos = [s for y, s in zip(y_true, y_score) if y == 1]
        neg = [s for y, s in zip(y_true, y_score) if y == 0]
        if len(pos) == 0 or len(neg) == 0:
            return None
        scores = list(y_score)

        sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i])
        ranks = [0.0] * len(scores)
        i = 0
        while i < len(sorted_idx):
            j = i
            while j + 1 < len(sorted_idx) and scores[sorted_idx[j + 1]] == scores[sorted_idx[i]]:
                j += 1
            avg_rank = (i + j + 2) / 2.0
            for k in range(i, j + 1):
                ranks[sorted_idx[k]] = avg_rank
            i = j + 1
        rank_sum_pos = sum(r for r, y in zip(ranks, y_true) if y == 1)
        n_pos = len(pos)
        n_neg = len(neg)
        auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        return float(auc)

    @torch.no_grad()
    def _evaluate_val(self, val_items, stage=1):
        if not val_items:
            return None, None, []

        example_groups = defaultdict(list)
        for item in val_items:
            ex_id = item.get("id") or item.get("example_id")
            example_groups[ex_id].append(item)

        y_graph = []
        s_graph = []
        ranking_total = 0
        ranking_correct = 0

        y_node = []
        s_node = []
        y_gap = []
        s_gap = []

        details = []

        for ex_id, items in example_groups.items():
            scores = []
            costs = []
            for item in items:
                chain = item.get("tools", [])
                if not chain:
                    continue
                step_texts = item.get("step_texts", None)
                user_request = item.get("user_request", "")
                edges = item.get("edges", None)

                feats, edge_idx, req_emb, edge_feats, gaps, gap_edge_feats = self.build_graph_inputs(
                    chain, step_texts=step_texts, user_request=user_request, edges=edges,
                    IO_TYPE2IDX=self.IO_TYPE2IDX
                )
                out = self.model(feats, edge_idx, req_emb, edge_feats=edge_feats, gaps=gaps, gap_edge_feats=gap_edge_feats)
                S_logit = float(out["S_logit"].view(-1)[0].item())
                S_val = float(out["S"].view(-1)[0].item())
                scores.append(S_logit)
                costs.append(float(item.get("cost", 0.0)))

                y_graph.append(1 if item.get("is_gt", False) else 0)
                s_graph.append(S_logit)

                node_labels = item.get("node_risk", [])
                node_preds = out["node_risks"].detach().cpu().numpy().tolist()
                if node_labels and len(node_labels) == len(node_preds):
                    y_node.extend([1 if x > 0 else 0 for x in node_labels])
                    s_node.extend([float(x) for x in node_preds])

                gap_labels = []
                gap_risk_edges = set(tuple(e) for e in item.get("gap_risk_edges", []))
                for (u_idx, v_idx) in gaps:
                    gap_labels.append(1 if (u_idx, v_idx) in gap_risk_edges else 0)
                gap_preds = out["gap_risks"].detach().cpu().numpy().tolist()
                if gap_labels and len(gap_labels) == len(gap_preds):
                    y_gap.extend(gap_labels)
                    s_gap.extend([float(x) for x in gap_preds])

                details.append({
                    "id": item.get("id"),
                    "tools": item.get("tools", []),
                    "edges": item.get("edges", []),
                    "step_texts": item.get("step_texts", []),
                    "user_request": item.get("user_request", ""),
                    "is_gt": item.get("is_gt", False),
                    "label_type": item.get("label_type", "unknown"),
                    "perturb_ops": item.get("perturb_ops", []),
                    "cost": item.get("cost", 0.0),
                    "y_cons": item.get("y_cons", 0.0),
                    "node_risk": item.get("node_risk", []),
                    "gap_risk_edges": item.get("gap_risk_edges", []),
                    "S": S_val,
                    "S_logit": S_logit,
                    "node_risks": node_preds,
                    "gap_risks": gap_preds,
                    "gaps": gaps
                })

            for i in range(len(scores)):
                for j in range(len(scores)):
                    if i == j:
                        continue
                    if costs[i] < costs[j]:
                        ranking_total += 1
                        if scores[i] > scores[j]:
                            ranking_correct += 1

        ranking_acc = (ranking_correct / ranking_total) if ranking_total > 0 else 0.0
        auc_graph = self._auc_from_scores(y_graph, s_graph)
        score_graph = None
        if auc_graph is not None:
            score_graph = 0.4 * ranking_acc + 0.6 * auc_graph

        auc_node = self._auc_from_scores(y_node, s_node)
        auc_gap = self._auc_from_scores(y_gap, s_gap)
        score_diag = None
        if auc_node is not None and auc_gap is not None:
            score_diag = 0.5 * auc_node + 0.5 * auc_gap

        return score_graph, score_diag, details


    def _train_stage(self, train_data, num_epochs, stage=1, stage1_epochs=0,
                     val_items=None, patience=5, min_delta=0.002):

        example_groups = defaultdict(list)
        for item in train_data:
            ex_id = item.get("id") or item.get("example_id")
            example_groups[ex_id].append(item)


        if stage == 2:
            total_node_pos, total_node_neg = 0, 0
            total_gap_pos, total_gap_neg = 0, 0

            for ex_id, items in example_groups.items():
                for item in items:
                    node_risk = item.get("node_risk", [])
                    total_node_pos += sum(node_risk)
                    total_node_neg += len(node_risk) - sum(node_risk)

                    gap_risk_edges = set(tuple(e) for e in item.get("gap_risk_edges", []))

                    chain = item.get("tools", [])
                    step_texts = item.get("step_texts", None)
                    user_request = item.get("user_request", "")
                    edges = item.get("edges", None)
                    try:
                        _, _, _, _, gaps, _ = self.build_graph_inputs(
                            chain, step_texts=step_texts, user_request=user_request, edges=edges,
                            IO_TYPE2IDX=self.IO_TYPE2IDX
                        )
                        real_gaps_count = len(gaps)
                    except Exception:
                        num_edges = len(item.get("edges", []))
                        real_gaps_count = num_edges + min(3, len(chain))

                    total_gap_pos += len(gap_risk_edges)
                    total_gap_neg += real_gaps_count - len(gap_risk_edges)


            if total_node_pos > 0:
                computed_node_pos_weight = max(1.0, total_node_neg / total_node_pos)
            else:
                computed_node_pos_weight = self.node_pos_weight

            if total_gap_pos > 0:
                computed_gap_pos_weight = max(1.0, total_gap_neg / total_gap_pos)
            else:
                computed_gap_pos_weight = self.gap_pos_weight

            self.node_pos_weight = computed_node_pos_weight
            self.gap_pos_weight = computed_gap_pos_weight

        best_loss = float('inf')
        best_score = None
        patience_cnt = 0
        best_model_state = None

        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            n_items = 0

            for _, (ex_id, items) in enumerate(example_groups.items(), start=1):
                try:
                    if stage == 2:
                        loss_node_total = torch.tensor(0.0, device=self.device)
                        loss_gap_total = torch.tensor(0.0, device=self.device)
                        num_graphs_stage2 = 0

                    gt_item = None
                    perturb_items = []

                    for item in items:
                        if item.get("is_gt", False) or item.get("label_type") == "gt":
                            gt_item = item
                        else:
                            perturb_items.append(item)

                    if not gt_item:
                        continue

                    user_request = gt_item.get("user_request", "")
                    gt_chain = gt_item["tools"]
                    if not gt_chain:
                        continue

                    gt_edges = gt_item.get("edges", None)
                    gt_step_texts = gt_item.get("step_texts", None)
                    gt_feats, gt_edge_idx, gt_req_emb, gt_edge_feats, gt_gaps, gt_gap_edge_feats = self.build_graph_inputs(
                        gt_chain, step_texts=gt_step_texts, user_request=user_request, edges=gt_edges,
                        IO_TYPE2IDX=self.IO_TYPE2IDX
                    )
                    gt_out = self.model(gt_feats, gt_edge_idx, gt_req_emb, edge_feats=gt_edge_feats, gaps=gt_gaps, gap_edge_feats=gt_gap_edge_feats)
                    S_gt_logit = gt_out["S_logit"].view(-1)
                    S_gt_prob = gt_out["S"].view(-1)

                    if S_gt_logit.dim() == 0:
                        S_gt_logit = S_gt_logit.unsqueeze(0)
                        S_gt_prob = S_gt_prob.unsqueeze(0)

                    loss_node_gt = torch.tensor(0.0, device=self.device)
                    loss_gap_gt = torch.tensor(0.0, device=self.device)

                    if stage == 2:
                        gt_node_labels = torch.tensor(gt_item.get("node_risk", [0]*len(gt_chain)), dtype=torch.float32, device=self.device)
                        gt_node_logits = gt_out["node_logits"].view(-1)

                        if gt_node_labels.numel() > 0 and gt_node_logits.numel() > 0 and gt_node_labels.size(0) == gt_node_logits.size(0):
                            pos_weight = torch.tensor(self.node_pos_weight, device=self.device)
                            loss_node_gt = F.binary_cross_entropy_with_logits(gt_node_logits, gt_node_labels, pos_weight=pos_weight)

                        gt_gap_logits = gt_out["gap_logits"].view(-1)
                        if gt_gap_logits.numel() > 0:
                            gt_gap_labels = torch.zeros_like(gt_gap_logits)
                            pos_weight = torch.tensor(self.gap_pos_weight, device=self.device)
                            loss_gap_gt = F.binary_cross_entropy_with_logits(gt_gap_logits, gt_gap_labels, pos_weight=pos_weight)

                        loss_node_total = loss_node_total + loss_node_gt
                        loss_gap_total = loss_gap_total + loss_gap_gt
                        num_graphs_stage2 += 1

                    perturb_S_logits = []
                    perturb_costs = []

                    for perturb_item in perturb_items:
                        perturb_chain = perturb_item.get("tools", [])
                        if not perturb_chain:
                            continue

                        perturb_edges = perturb_item.get("edges", None)
                        perturb_step_texts = perturb_item.get("step_texts", None)
                        perturb_feats, perturb_edge_idx, perturb_req_emb, perturb_edge_feats, perturb_gaps, perturb_gap_edge_feats = self.build_graph_inputs(
                            perturb_chain, step_texts=perturb_step_texts, user_request=user_request, edges=perturb_edges,
                            IO_TYPE2IDX=self.IO_TYPE2IDX
                        )
                        perturb_out = self.model(perturb_feats, perturb_edge_idx, perturb_req_emb, edge_feats=perturb_edge_feats, gaps=perturb_gaps, gap_edge_feats=perturb_gap_edge_feats)

                        S_perturb_logit = perturb_out["S_logit"].view(-1)
                        S_perturb_prob = perturb_out["S"].view(-1)

                        if S_perturb_logit.dim() == 0:
                            S_perturb_logit = S_perturb_logit.unsqueeze(0)

                        cost = perturb_item.get("cost", 0.0)
                        perturb_S_logits.append(S_perturb_logit)
                        perturb_costs.append(cost)

                        if stage == 2:
                            loss_node_p = torch.tensor(0.0, device=self.device)
                            loss_gap_p = torch.tensor(0.0, device=self.device)
                            perturb_node_labels = torch.tensor(perturb_item.get("node_risk", []), dtype=torch.float32, device=self.device)
                            perturb_node_logits = perturb_out["node_logits"].view(-1)

                            if perturb_node_labels.numel() > 0 and perturb_node_logits.numel() > 0 and perturb_node_labels.size(0) == perturb_node_logits.size(0):
                                pos_weight = torch.tensor(self.node_pos_weight, device=self.device)
                                loss_node_p = F.binary_cross_entropy_with_logits(perturb_node_logits, perturb_node_labels, pos_weight=pos_weight)

                            gap_risk_edges = set(tuple(e) for e in perturb_item.get("gap_risk_edges", []))
                            perturb_gap_logits = perturb_out["gap_logits"].view(-1)

                            if perturb_gap_logits.numel() > 0:
                                gap_labels = []
                                for (u_idx, v_idx) in perturb_gaps:
                                    is_pos = (u_idx, v_idx) in gap_risk_edges
                                    gap_labels.append(1.0 if is_pos else 0.0)

                                perturb_gap_labels = torch.tensor(gap_labels, dtype=torch.float32, device=self.device)

                                if perturb_gap_labels.size(0) == perturb_gap_logits.size(0):
                                    pos_weight = torch.tensor(self.gap_pos_weight, device=self.device)
                                    loss_gap_p = F.binary_cross_entropy_with_logits(perturb_gap_logits, perturb_gap_labels, pos_weight=pos_weight)

                            loss_node_total = loss_node_total + loss_node_p
                            loss_gap_total = loss_gap_total + loss_gap_p
                            num_graphs_stage2 += 1

                    if stage == 1:
                        loss_graph = torch.tensor(0.0, device=self.device)
                        loss_rank = torch.tensor(0.0, device=self.device)

                        y_gt = torch.tensor(1.0, device=self.device)
                        loss_graph += F.binary_cross_entropy_with_logits(S_gt_logit.view(-1), y_gt.view(-1))

                        for i, (S_logit, cost) in enumerate(zip(perturb_S_logits, perturb_costs)):
                            y_cons = np.exp(-cost / self.cost_tau)
                            y_cons_tensor = torch.tensor(y_cons, dtype=torch.float32, device=self.device)
                            loss_graph += F.binary_cross_entropy_with_logits(S_logit.view(-1), y_cons_tensor.view(-1))

                        n_graphs = 1 + len(perturb_S_logits)
                        if n_graphs > 0:
                            loss_graph = loss_graph / n_graphs

                        if perturb_S_logits:
                            all_S_logits = [S_gt_logit] + perturb_S_logits
                            all_costs = [0.0] + perturb_costs

                            rank_losses = []
                            tau = max(self.margin_rank, 1e-6)
                            for i in range(len(all_S_logits)):
                                for j in range(i + 1, len(all_S_logits)):
                                    S_i = all_S_logits[i][0] if all_S_logits[i].numel() > 0 else torch.tensor(0.0, device=self.device)
                                    S_j = all_S_logits[j][0] if all_S_logits[j].numel() > 0 else torch.tensor(0.0, device=self.device)
                                    cost_diff = all_costs[j] - all_costs[i]

                                    target = torch.sigmoid(torch.tensor(cost_diff / tau, device=self.device))
                                    rank_logit = S_i - S_j
                                    rank_losses.append(F.binary_cross_entropy_with_logits(rank_logit.view(-1), target.view(-1)))

                            if rank_losses:
                                loss_rank = torch.stack(rank_losses).mean()

                        total_loss = self.lambda_graph * loss_graph + self.lambda_rank * loss_rank

                    else:
                        denom = max(1, num_graphs_stage2)
                        total_loss = (loss_node_total + self.lambda_gap * loss_gap_total) / denom

                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()

                    epoch_loss += total_loss.item()
                    n_items += 1

                except Exception:
                    continue

            avg_loss = epoch_loss / max(1, n_items)

            if val_items:
                was_training = self.model.training
                self.model.eval()
                score_graph, score_diag, details = self._evaluate_val(val_items, stage=stage)
                if was_training:
                    self.model.train()
                if stage == 1:
                    score_now = score_graph
                else:
                    score_now = score_diag

                if score_now is not None:
                    if best_score is None or score_now > best_score + min_delta:
                        best_score = score_now
                        patience_cnt = 0
                        best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                        best_loss = avg_loss
                    else:
                        patience_cnt += 1
                else:

                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        patience_cnt = 0
                        best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    else:
                        patience_cnt += 1
            else:
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_cnt = 0
                    best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_cnt += 1

            if patience_cnt >= patience:
                break

        if best_model_state:
            self.model.load_state_dict(best_model_state)
            self.model = self.model.to(self.device)

        return self.model, best_loss

    @torch.no_grad()
    def score_chain(self, chain, request_text="", edges=None, step_texts=None):
        self.model.eval()
        if not chain or len(chain) == 0:
            return {"S": 0.0, "node_risks": [], "gap_risks": [], "gaps": []}

        if edges is None:
            edges = []
            if len(chain) > 0:
                edges.append((0, 1))
            for i in range(len(chain)-1):
                edges.append((i+1, i+2))

        node_feats, edge_index, req_emb, edge_feats, gaps, gap_edge_feats = self.build_graph_inputs(
            chain, step_texts=step_texts, user_request=request_text, edges=edges,
            IO_TYPE2IDX=self.IO_TYPE2IDX
        )
        result = self.model(node_feats, edge_index, req_emb, edge_feats=edge_feats, gaps=gaps, gap_edge_feats=gap_edge_feats)

        S_val = result["S"].item()
        return {
            "S": S_val,
            "node_risks": result["node_risks"].cpu().numpy().tolist() if result["node_risks"].numel() > 0 else [],
            "gap_risks": result["gap_risks"].cpu().numpy().tolist() if result["gap_risks"].numel() > 0 else [],
            "gaps": gaps
        }