import argparse 
import torch 
import prettytable as pt
import json 
import os
import numpy as np
import hashlib
import time
import math
import asyncio
from openai import OpenAI
import sys
sys.path.append("./")
from model import ModelTrainer
from utils_preproc import order_chain_with_steps_and_edges, clean_step_texts, edges_to_links
from utils import init_random_state, load_test_data, get_cur_time, prepare_training_ids
# from evaluate import f1_score

# from sklearn.metrics import f1_score
# import evaluate
# f1 = evaluate.load("f1")

# results = f1.compute(
#     predictions=[0, 1, 1],
#     references=[0, 1, 0]
# )
# 原始训练集 = split_ids中test_ids之外的所有数据
#     │
#     ├── 在版本2中: 进一步划分为
#     │   ├── 90%: 实际用于后续GNN训练的数据
#     │   └── 10%: 作为"验证集"进行LLM推理 (direct_val.json)
#     │
#     └── 在版本3中:
#         ├── 90% (train_data_raw):
#         │   ├── 构建N-gram统计
#         │   ├── 构建混淆矩阵
#         │   ├── 生成扰动训练样本
#         │   ├── 训练GNN模型
#         │   └── 预训练对齐模型
#         │
#         └── 10% (val_data_raw):
#             ├── 用于GNN训练的早停验证
#             └── 用于搜索风险阈值和接受阈值


# ┌─────────────────────────────────────────────────────────────────────────┐
# │                           主入口 (main)                                  │
# │                    ┌─────────────────────────────────┐                  │
# │                    │  阶段0: 初始化与数据加载           │                  │
# │                    │  • 解析命令行参数                  │                  │
# │                    │  • 加载数据集划分 (test_ids)       │                  │
# │                    │  • 划分训练集/验证集 (90%/10%)      │                  │
# │                    └─────────────────────────────────┘                  │
# │                                    │                                    │
# │                                    ▼                                    │
# │  ┌─────────────────────────────────────────────────────────────────┐   │
# │  │                     阶段1: 特征工程与缓存构建                      │   │
# │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │   │
# │  │  │ build_io_   │  │ build_typed │  │ build_confusion_prior   │  │   │
# │  │  │ types_vocab │  │ _ngrams     │  │ (工具混淆矩阵)            │  │   │
# │  │  │ (IO类型词典) │  │ (N-gram统计) │  │                         │  │   │
# │  │  └─────────────┘  └─────────────┘  └─────────────────────────┘  │   │
# │  │                                    │                            │   │
# │  │                                    ▼                            │   │
# │  │              ┌─────────────────────────────────┐                │   │
# │  │              │  init_embedding_cache (嵌入缓存) │                │   │
# │  │              │  • 预计算所有工具的向量表示        │                │   │
# │  │              │  • 预计算用户请求的向量表示        │                │   │
# │  │              └─────────────────────────────────┘                │   │
# │  └─────────────────────────────────────────────────────────────────┘   │
# │                                    │                                    │
# │                                    ▼                                    │
# │  ┌─────────────────────────────────────────────────────────────────┐   │
# │  │                     阶段2: 模型训练与预训练                        │   │
# │  │  ┌─────────────────────────────────────────────────────────┐    │   │
# │  │  │  generate_perturbations_with_labels (生成训练样本)       │    │   │
# │  │  │  • 对GT计划添加扰动（替换/删除/打乱）生成负样本           │    │   │
# │  │  │  • 计算每个样本的代价和质量分数                          │    │   │
# │  │  └─────────────────────────────────────────────────────────┘    │   │
# │  │                                    │                            │   │
# │  │                                    ▼                            │   │
# │  │              ┌─────────────────────────────────┐                │   │
# │  │              │  ModelTrainer.train (GNN训练)     │                │   │
# │  │              │  • 对比学习：GT vs 扰动样本        │                │   │
# │  │              │  • 联合优化：图分数 + 节点风险 + 间隙风险 │                │   │
# │  │              └─────────────────────────────────┘                │   │
# │  │                                    │                            │   │
# │  │                    [可选] train_alignment_from_raw              │   │
# │  │                    (步骤-工具对齐预训练)                         │   │
# │  └─────────────────────────────────────────────────────────────────┘   │
# │                                    │                                    │
# │                                    ▼                                    │
# │  ┌─────────────────────────────────────────────────────────────────┐   │
# │  │                     阶段3: 验证集阈值搜索 (Hyperparameter Tuning) │   │
# │  │  ┌─────────────────────────────────────────────────────────┐    │   │
# │  │  │  Stage 3.1: search_risk_thresholds                     │    │   │
# │  │  │  • 输入: 验证集上的直接预测结果                         │    │   │
# │  │  │  • 输出: 最优 θ_node, θ_gap (风险阈值)                  │    │   │
# │  │  │  • 方法: 网格搜索最大化F1                              │    │   │
# │  │  └─────────────────────────────────────────────────────────┘    │   │
# │  │                                    │                            │   │
# │  │                                    ▼                            │   │
# │  │  ┌─────────────────────────────────────────────────────────┐    │   │
# │  │  │  Stage 3.2: stage1_candidate_thresholds                │    │   │
# │  │  │  • 基于验证集分数分布生成T_accept候选列表                 │    │   │
# │  │  └─────────────────────────────────────────────────────────┘    │   │
# │  │                                    │                            │   │
# │  │                                    ▼                            │   │
# │  │  ┌─────────────────────────────────────────────────────────┐    │   │
# │  │  │  Stage 3.3: stage2_search_thresholds_with_llm          │    │   │
# │  │  │  • 对每个验证样本预计算LLM修正结果                      │    │   │
# │  │  │  • 网格搜索最优T_accept (接受阈值)                        │    │   │
# │  │  │  • 选择使Node-F1+Link-F1最大的阈值                      │    │   │
# │  │  └─────────────────────────────────────────────────────────┘    │   │
# │  └─────────────────────────────────────────────────────────────────┘   │
# │                                    │                                    │
# │                                    ▼                                    │
# │  ┌─────────────────────────────────────────────────────────────────┐   │
# │  │                     阶段4: 测试集推理 (三阶段流水线)              │   │
# │  │                                                                 │   │
# │  │  ┌─────────────────────────────────────────────────────────┐    │   │
# │  │  │  Phase 1: score_one_sample (GNN评分)                    │    │   │
# │  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │    │   │
# │  │  │  │normalize_  │  │order_chain_ │  │  add_start_edge │ │    │   │
# │  │  │  │tools_list   │  │with_steps_  │  │  (添加START节点) │ │    │   │
# │  │  │  │(工具名规范化)│  │and_edges    │  │                 │ │    │   │
# │  │  │  └─────────────┘  └─────────────┘  └─────────────────┘ │    │   │
# │  │  │                          │                            │    │   │
# │  │  │                          ▼                            │    │   │
# │  │  │              ┌─────────────────────────┐              │    │   │
# │  │  │              │ controller.score_chain  │              │    │   │
# │  │  │              │ (GNN前向传播计算分数)    │              │    │   │
# │  │  │              │ • S: 整体规划质量分数    │              │    │   │
# │  │  │              │ • node_risks: 节点风险   │              │    │   │
# │  │  │              │ • gap_risks: 间隙风险    │              │    │   │
# │  │  │              └─────────────────────────┘              │    │   │
# │  │  └─────────────────────────────────────────────────────────┘    │   │
# │  │                                    │                            │   │
# │  │                                    ▼                            │   │
# │  │  ┌─────────────────────────────────────────────────────────┐    │   │
# │  │  │  Phase 2: refine_one_sample (LLM精细化修正)            │    │   │
# │  │  │                                                         │    │   │
# │  │  │  ┌─────────────────────────────────────────────────┐   │    │   │
# │  │  │  │  iterative_refine_with_llm (迭代优化核心)        │   │    │   │
# │  │  │  │                                                 │   │    │   │
# │  │  │  │  Step 1: 检查 S ≥ T_accept?                      │   │    │   │
# │  │  │  │       Yes ──► 直接返回 (接受)                     │   │    │   │
# │  │  │  │       No  ──► 继续优化                           │   │    │   │
# │  │  │  │                                                 │   │    │   │
# │  │  │  │  Step 2: 调用 call_llm_patch (LLM修正)           │   │    │   │
# │  │  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────┐  │   │    │   │
# │  │  │  │  │suggest_     │  │suggest_     │  │build_   │  │   │    │   │
# │  │  │  │  │replacement_ │  │insertion_   │  │full_gnn │  │   │    │   │
# │  │  │  │  │tools        │  │tools        │  │_report  │  │   │    │   │
# │  │  │  │  │(替换候选)    │  │(插入候选)    │  │(完整报告)│  │   │    │   │
# │  │  │  │  └─────────────┘  └─────────────┘  └─────────┘  │   │    │   │
# │  │  │  │                          │                     │   │    │   │
# │  │  │  │                          ▼                     │   │    │   │
# │  │  │  │              ┌─────────────────────────┐       │   │    │   │
# │  │  │  │              │ 构建Prompt + 调用LLM   │       │   │    │   │
# │  │  │  │              │ robust_json_extract    │       │   │    │   │
# │  │  │  │              │ (解析LLM返回的JSON)     │       │   │    │   │
# │  │  │  │              └─────────────────────────┘       │   │    │   │
# │  │  │  │                          │                     │   │    │   │
# │  │  │  │                          ▼                     │   │    │   │
# │  │  │  │  Step 3: parse_edit_ops (解析编辑操作)         │   │    │   │
# │  │  │  │  Step 4: validate_edit_ops (验证合法性)        │   │    │   │
# │  │  │  │       不合法 ──► call_llm_patch_fix (修正)     │   │    │   │
# │  │  │  │                                                 │   │    │   │
# │  │  │  │  Step 5: apply_edit_ops (应用编辑)              │   │    │   │
# │  │  │  │  • 替换节点工具                                │   │    │   │
# │  │  │  │  • 插入新工具到间隙                            │   │    │   │
# │  │  │  │  • 重新生成步骤描述                            │   │    │   │
# │  │  │  │                                                 │   │    │   │
# │  │  │  │  Step 6: order_chain_with_steps_and_edges      │   │    │   │
# │  │  │  │          (重新排序和连接)                       │   │    │   │
# │  │  │  │                                                 │   │    │   │
# │  │  │  │  Step 7: controller.score_chain (重新评分)     │   │    │   │
# │  │  │  │                                                 │   │    │   │
# │  │  │  │  Step 8: 稳定性检查 (对比GT质量不下降)          │   │    │   │
# │  │  │  │       不稳定 ──► 回滚到初始计划                  │   │    │   │
# │  │  │  │                                                 │   │    │   │
# │  │  │  │  Step 9: 改进检查 (S' > S + DELTA_IMPROVE?)    │   │    │   │
# │  │  │  │       未改进 ──► 回滚到初始计划                  │   │    │   │
# │  │  │  │       已改进 ──► 接受新计划                     │   │    │   │
# │  │  │  │                                                 │   │    │   │
# │  │  │  └─────────────────────────────────────────────────┘   │    │   │
# │  │  │                                                         │    │   │
# │  │  │  [并发控制: asyncio.Semaphore(args.multiworker)]        │    │   │
# │  │  └─────────────────────────────────────────────────────────┘    │   │
# │  │                                    │                            │   │
# │  │                                    ▼                            │   │
# │  │  ┌─────────────────────────────────────────────────────────┐    │   │
# │  │  │  Phase 3: 评估 (compute_f1_set_based)                    │    │   │
# │  │  │  • Direct vs Refined 对比                               │    │   │
# │  │  │  • 输出: Node-F1, Link-F1, Accuracy                     │    │   │
# │  │  └─────────────────────────────────────────────────────────┘    │   │
# │  │                                                                 │   │
# │  └─────────────────────────────────────────────────────────────────┘   │
# │                                                                         │
# └─────────────────────────────────────────────────────────────────────────┘

# 本项目是一个基于 GNN（图神经网络）评估与 LLM（大语言模型）迭代修正的任务规划优化系统。
# 下面以具体案例为例，详细拆解各大模块的输入、输出及数据形态。
#
# 【案例背景】
# ID: "10819379"
# User Request: "I have an image 'example.jpg' and want to change its background color to blue. Then, I'd like to estimate the depth of the objects in the edited image and classify the depth map as a table. Based on the classification, I want a text description to be generated and finally, create a video from this text."
#
# =================================================================================================
# 模块一：初始化与特征工程 (Initialization & Feature Engineering)
# =================================================================================================
# 作用：构建工具知识库、预计算嵌入向量、统计工具共现规律，为GNN和LLM提供底层支持。
#
# 1. build_io_types_vocab (IO类型词典构建)
#    - 输入: tool_meta (工具元数据，包含每个工具的 input-type/output-type)
#    - 输出: IO_TYPES (所有可能的IO类型列表), IO_TYPE2IDX (类型到索引的映射)
#    - 示例: {"image": 0, "text": 1, "depth_map": 2, ...}
#
# 2. build_typed_ngrams (N-gram 统计构建)
#    - 输入: train_data_raw (训练集原始数据), tool_meta
#    - 输出: typed_ngrams (字典，记录工具对或工具三元组的共现频率)
#    - 示例: {"__f2__": {("load_image", "change_background"): 150, ("change_background", "estimate_depth"): 80}}
#
# 3. build_confusion_prior (工具混淆矩阵/候选生成器)
#    - 输入: tool_meta, embedding_cache
#    - 输出: confusion (字典，记录每个工具易混淆的其他工具及其相似度分数)
#    - 示例: {"load_image": [("load_video", 0.85), ("read_text_file", 0.60)], ...}
#
# 4. init_embedding_cache (嵌入缓存初始化)
#    - 输入: tool_meta, lm_name (如 'intfloat/e5-large')
#    - 输出: embedding_cache (对象，存储所有工具描述和用户请求的向量表示)
#    - 示例: cache._tool_embeddings["change_background"] -> [0.12, -0.45, ..., 0.88] (768维向量)
#
# =================================================================================================
# 模块二：模型训练 (Model Training) - 离线阶段
# =================================================================================================
# 作用：训练GNN模型，使其能够评估任务规划的质量（整体分数S、节点风险、间隙风险）。
#
# 1. generate_perturbations_with_labels (生成对比学习样本)
#    - 输入: 单个训练样本 ex (含GT计划), confusion, typed_ngrams
#    - 输出: train_items (列表，包含正样本GT和负样本Perturbed)
#    - 示例 Item:
#      {
#        "tools": ["load_image", "change_background", "estimate_depth"],
#        "edges": [(0,1), (1,2)],
#        "y_cons": 1.0, # GT标签
#        "node_risk": [0.1, 0.1, 0.1],
#        "is_gt": True
#      }
#      {
#        "tools": ["load_image", "read_text_file", "estimate_depth"], # 错误替换
#        "edges": [(0,1), (1,2)],
#        "y_cons": 0.2, # 低分标签
#        "node_risk": [0.1, 0.9, 0.1], # 中间节点高风险
#        "is_gt": False
#      }
#
# 2. ModelTrainer.train (GNN训练)
#    - 输入: train_items, val_items
#    - 输出: best_model (训练好的GNN权重)
#    - 内部机制: 最小化对比损失，使GT计划的得分S高于扰动计划，同时预测准确的节点/间隙风险。
#
# =================================================================================================
# 模块三：阈值搜索 (Threshold Search) - 验证集阶段
# =================================================================================================
# 作用：确定两个关键阈值：风险阈值 (theta_node/gap) 和 接受阈值 (T_accept)。
#
# 1. search_risk_thresholds
#    - 输入: val_ids, val_content (验证集直接预测结果), controller (GNN模型)
#    - 输出: theta_node (节点风险阈值, e.g., 0.6), theta_gap (间隙风险阈值, e.g., 0.5)
#    - 逻辑: 网格搜索，寻找使“高风险节点/间隙”与“真实错误节点/间隙”F1分数最大的阈值。
#
# 2. stage2_search_thresholds_with_llm
#    - 输入: val_ids, thresholds候选列表, controller, tool_meta等
#    - 输出: T_accept (接受阈值, e.g., 0.85)
#    - 逻辑: 模拟推理过程。如果GNN评分 S >= T_accept，则直接接受；否则调用LLM修正。
#           选择使最终 Node-F1 + Link-F1 最大的 T_accept。
#
# =================================================================================================
# 模块四：测试集推理 (Inference Pipeline) - 在线阶段
# =================================================================================================
# 针对案例 ID "10819379" 的详细流转：
#
# --- Phase 1: GNN 评分 (Score One Sample) ---
# 1. 数据准备:
#    - 用户请求: "I have an image 'example.jpg'..."
#    - 直接预测计划 (Direct Plan, 来自LLM基线):
#      nodes: ["load_image", "change_background", "estimate_depth", "classify_table", "generate_text", "create_video"]
#      steps: ["Load example.jpg", "Change bg to blue", "Estimate depth", "Classify depth map", "Generate description", "Create video"]
#      edges: [(0,1), (1,2), (2,3), (3,4), (4,5)]
#
# 2. 预处理:
#    - normalize_tools_list: 规范化工具名 (处理别名)
#    - order_chain_with_steps_and_edges: 确保节点顺序与边一致
#    - add_start_edge: 添加虚拟起点 (0, 1) -> (START, load_image)
#
# 3. GNN 前向传播 (controller.score_chain):
#    - 输入: nodes, user_request, edges, step_texts
#    - 输出:
#      {
#        "S": 0.72,              # 整体质量分数 (假设低于 T_accept=0.85，需修正)
#        "node_risks": [0.1, 0.1, 0.8, 0.2, 0.1, 0.1], # estimate_depth 风险高? 或者 classify_table 风险高?
#        "gap_risks": [0.1, 0.1, 0.6, 0.1, 0.1],       # gap between depth and classify 风险高
#        "gaps": [(2,3), ...]    # 高风险间隙索引
#      }
#
# --- Phase 2: LLM 精细化修正 (Refine One Sample) ---
# 核心函数: iterative_refine_with_llm
#
# 1. 判断是否接受:
#    - S (0.72) < T_accept (0.85) -> 进入修正流程
#
# 2. 候选生成 (Candidate Generation):
#    A. 高风险节点分析 (Node Risk > theta_node):
#       - 假设 Node 2 ("estimate_depth") 风险高。
#       - suggest_replacement_tools:
#         - 输入: current_tool="estimate_depth", context=request+steps
#         - 逻辑: 查混淆矩阵 + IO兼容检查 + 语义相似度
#         - 输出: [{"tool": "compute_depth_map", "score": 0.9}, {"tool": "monocular_depth_estimation", "score": 0.85}]
#
#    B. 高风险间隙分析 (Gap Risk > theta_gap):
#       - 假设 Gap (2->3) ("estimate_depth" -> "classify_table") 风险高。
#       - suggest_insertion_tools:
#         - 输入: u="estimate_depth", v="classify_table"
#         - 逻辑: 查N-gram共现 + IO桥接 (depth_map output -> table input?)
#         - 输出: [{"tool": "convert_to_grayscale", "score": 0.7}, {"tool": "normalize_depth", "score": 0.8}]
#
# 3. 构建 Prompt 并调用 LLM (call_llm_patch):
#    - Prompt 内容:
#      - User Request
#      - Current Plan (JSON)
#      - GNN Diagnosis: "Node 2 has high risk (0.8). Gap 2->3 has high risk (0.6)."
#      - Candidates: "Node 2 candidates: [compute_depth_map, ...]; Gap 2->3 candidates: [normalize_depth, ...]"
#      - Instruction: "Output JSON edits only."
#
#    - LLM 输出 (LLM Output):
#      {
#        "edits": [
#          {"op": "replace_node", "node_id": 2, "candidate_id": 0, "step": "Step 3: Compute precise depth map using monocular estimation"},
#          {"op": "insert_on_gap", "gap_id": 0, "candidate_id": 1, "step": "Step 4: Normalize the depth values for classification"}
#        ]
#      }
#
# 4. 解析与验证 (Parse & Validate):
#    - parse_edit_ops: 解析JSON为操作列表
#    - validate_edit_ops: 检查ID合法性、无重复工具、步骤非空
#
# 5. 应用编辑 (Apply Edits):
#    - apply_edit_ops:
#      - 替换 Node 2: "estimate_depth" -> "compute_depth_map"
#      - 插入新节点: 在 Node 2 和 3 之间插入 "normalize_depth"
#      - 重新生成 Steps 和 Edges
#    - 新计划 (New Plan):
#      nodes: ["load_image", "change_background", "compute_depth_map", "normalize_depth", "classify_table", "generate_text", "create_video"]
#
# 6. 重新评分与稳定性检查 (Re-score & Check):
#    - controller.score_chain(New Plan) -> New_S
#    - 假设 New_S = 0.88
#    - 改进检查: New_S (0.88) > Old_S (0.72) + DELTA_IMPROVE (0.02) ? Yes.
#    - 稳定性检查: 如果有GT，检查F1是否下降。若无GT或F1未降，则接受。
#
# 7. 返回结果:
#    - Refined Plan: 上述新计划
#    - Strategy: "patch"
#
# --- Phase 3: 评估 (Evaluation) ---
# 1. 计算指标:
#    - Node-F1: 比较 Refined Nodes 与 GT Nodes (集合交集/并集)
#    - Link-F1: 比较 Refined Edges 与 GT Edges
#    - Accuracy: Node-F1 > 0.99 AND Link-F1 > 0.99
#
# 2. 输出示例:
#    | Dataset | Method          | N-F1   | L-F1   | Accuracy |
#    |---------|-----------------|--------|--------|----------|
#    | huggingface | Direct      | 0.7500 | 0.6800 | 0.45     |
#    | huggingface | Direct+LLM_Refined | 0.8200 | 0.7800 | 0.62     |
#
# =================================================================================================


# 核心函数流转详细说明
# 第一层：初始化与特征工程
# 函数	输入	输出	主要作用
# prepare_training_ids	dataset, train_num, test_ids	train_ids	从非测试数据中采样训练ID
# build_io_types_vocab	tool_meta	IO_TYPES, IO_TYPE2IDX	构建IO类型词典
# build_typed_ngrams	train_data, tool_meta	typed_ngrams	统计工具共现频率
# build_confusion_prior	tool_meta, embedding_cache	confusion	构建工具混淆矩阵
# init_embedding_cache	tool_meta, lm_name	embedding_cache	预计算所有嵌入向量

# 第二层：模型训练
# 函数	输入	输出	主要作用
# generate_perturbations_with_labels	train_data, confusion, typed_ngrams	train_items	生成对比学习样本（GT + 扰动）
# ModelTrainer.train	train_items, val_items	best_model	GNN对比学习训练
# train_alignment_from_raw	train_data	align_proj	步骤-工具语义对齐预训练

# 第三层：阈值搜索（验证集）
# 函数	输入	输出	主要作用
# load_test_data	dataset, method='direct_val'	val_content	加载验证集预测结果
# search_risk_thresholds	val_ids, val_content, controller	θ_node, θ_gap	搜索最优风险阈值
# stage1_candidate_thresholds	val_stats	candidate_T_accept	生成接受阈值候选
# stage2_search_thresholds_with_llm	val_ids, val_content, thresholds	best_T_accept	搜索最优接受阈值
# 第四层：测试推理（核心流水线）
# Phase 1: GNN评分
# 函数	输入	输出	主要作用
# normalize_tools_list	pred_nodes, alias_map	normalized_nodes	规范化工具名称
# order_chain_with_steps_and_edges	nodes, steps, links	ordered_nodes, ordered_steps, edges	排序节点和边
# add_start_edge	edges, num_nodes	edges_with_start	添加虚拟START节点
# controller.score_chain	nodes, user_request, edges, steps	{S, node_risks, gap_risks, gaps}	GNN前向传播
# Phase 2: LLM精细化修正
# 函数	输入	输出	主要作用
# iterative_refine_with_llm	init_plan, init_gnn, thresholds	refined_plan, strategy, history	迭代优化主控
# suggest_replacement_tools	controller, node_idx, current_plan	candidates	为高风险节点找替换候选
# suggest_insertion_tools	controller, gap_pos, typed_ngrams	candidates	为高风险间隙找插入候选
# build_full_gnn_report	gnn_result, current_plan	report	构建完整GNN诊断报告
# call_llm_patch	user_request, tool_meta, candidates, gnn_report	llm_output	调用LLM生成编辑操作
# parse_edit_ops	llm_output	ops	解析编辑操作
# validate_edit_ops	ops, node_candidates, gap_candidates	is_valid, errors	验证操作合法性
# call_llm_patch_fix	error_msg, candidates	fixed_output	LLM修正错误操作
# apply_edit_ops	plan, ops, candidates	new_plan	应用编辑生成新计划
# compute_f1_set_based	pred_list, gt_list	f1_score	计算F1评估指标

# 关键数据流转
# 用户请求 ──┬──► embedding_cache ──► 请求向量
#           │
#           ├──► 直接预测结果 (nodes, steps, links)
#           │         │
#           │         ▼
#           │    GNN评分 (score_chain)
#           │    ├──► S (整体质量分数)
#           │    ├──► node_risks[] (每个节点的风险)
#           │    └──► gap_risks[] (每个间隙的风险)
#           │         │
#           │    S < T_accept? ──Yes──► LLM修正流程
#           │         │
#           │    No (直接接受)
#           │         │
#           ▼         ▼
#     ┌─────────────────────────┐
#     │  高风险节点 ──► suggest_replacement_tools ──► 替换候选列表
#     │  高风险间隙 ──► suggest_insertion_tools ──► 插入候选列表
#     └─────────────────────────┘
#                 │
#                 ▼
#         ┌─────────────────┐
#         │  构建Prompt (包含GNN诊断 + 候选工具) │
#         │  call_llm_patch   │
#         └─────────────────┘
#                 │
#                 ▼
#         LLM生成编辑操作 (replace_node / insert_on_gap)
#                 │
#                 ▼
#         ┌─────────────────┐
#         │  验证 ──► 应用 ──► 重新评分 ──► 检查改进  │
#         │  validate ──► apply ──► score ──► check   │
#         └─────────────────┘
#                 │
#         改进足够? ──Yes──► 接受新计划
#                 │
#                No ──► 回滚到初始计划
#
# 并发控制机制
# ┌─────────────────────────────────────────┐
# │  asyncio.Semaphore(args.multiworker)    │
# │  限制并发数（默认4个并发）                 │
# │                                         │
# │  async def score_one_async(data_id):    │
# │      async with sem:                    │
# │          loop.run_in_executor(None, score_one_sample, data_id) │
# │                                         │
# │  tasks = [score_one_async(id) for id in test_ids] │
# │  asyncio.gather(*tasks)  # 并发执行      │
# └─────────────────────────────────────────┘
#
# 缓存机制
# 缓存类型	用途	实现
# _IO_CACHE	IO类型映射缓存	全局字典
# embedding_cache	工具/请求向量缓存	磁盘 + 内存
# llm_cache	LLM调用结果缓存	字典 (prompt_hash → result)
# typed_ngrams	N-gram统计缓存	JSON文件
# confusion	混淆矩阵缓存	JSON文件




# 在文件顶部添加这个自定义 F1 计算函数
def compute_f1_set_based(pred_list, gt_list):
    """
    基于集合的 F1 计算（忽略顺序）
    适用于工具节点和链接的评估
    """
    if not isinstance(pred_list, list):
        pred_list = list(pred_list) if pred_list else []
    if not isinstance(gt_list, list):
        gt_list = list(gt_list) if gt_list else []

    # 转换为集合
    pred_set = set(pred_list)
    gt_set = set(gt_list)

    # 计算 TP, FP, FN
    tp = len(pred_set & gt_set)  # 交集
    fp = len(pred_set - gt_set)  # 预测有但真实没有
    fn = len(gt_set - pred_set)  # 真实有但预测没有

    # 计算精确率、召回率、F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1

# 基于集合的F1计算
# 计算预测结果与真实结果的F1分数，不考虑元素顺序
# 将列表转换为集合
# 计算TP（真正例：交集）、FP（假正例：预测有但真实没有）、FN（假负例：真实有但预测没有）
# 计算Precision = TP/(TP+FP)，Recall = TP/(TP+FN)
# F1 = 2PrecisionRecall/(Precision+Recall)
def compute_f1_sequence_based(pred_list, gt_list):
    """
    基于序列的 F1 计算（考虑顺序，使用 LCS-like 方法）
    如果需要严格顺序匹配，可以使用此方法
    """
    if not isinstance(pred_list, list):
        pred_list = list(pred_list) if pred_list else []
    if not isinstance(gt_list, list):
        gt_list = list(gt_list) if gt_list else []

    # 使用集合方法计算（如果你需要顺序敏感，需要更复杂的算法）
    # 这里使用简单的精确匹配计数
    pred_set = set(pred_list)
    gt_set = set(gt_list)

    tp = sum(1 for p in pred_list if p in gt_set)
    fp = sum(1 for p in pred_list if p not in gt_set)
    fn = sum(1 for g in gt_list if g not in pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1

_api_key = os.environ.get("OPENAI_API_KEY")
_api_base_url = os.environ.get("OPENAI_API_BASE")
client = OpenAI(api_key=_api_key, base_url=_api_base_url)


# 核心阈值参数
TAU_ACCEPT = 0.9           # 接受阈值：GNN分数≥此值直接接受
DELTA_IMPROVE = 0.02       # 改进阈值：新计划必须比原计划好至少此值
THETA_NODE = 0.5           # 节点风险阈值（默认）
THETA_GAP = 0.5            # 间隙风险阈值（默认）

_IO_CACHE = {}             # IO类型缓存
LLM_MAX_RETRIES = 5        # LLM最大重试次数
LLM_RETRY_BASE_SEC = 1.0   # LLM重试基础等待时间
LLM_RETRY_MAX_SEC = 10.0   # LLM重试最大等待时间
LLM_MIN_INTERVAL_SEC = 0.5 # LLM调用最小间隔
_LLM_LAST_CALL_TS = 0.0    # 上次LLM调用时间戳

# LLM调用限流
def _throttle_llm_calls():
    global _LLM_LAST_CALL_TS
    now = time.time()
    wait = LLM_MIN_INTERVAL_SEC - (now - _LLM_LAST_CALL_TS)
    if wait > 0:
        time.sleep(wait)
    _LLM_LAST_CALL_TS = time.time()


def _tool_desc_map(tool_meta):
    return {n["id"]: n.get("desc", n.get("description", "")) for n in tool_meta["nodes"]}

def _format_steps_for_prompt(steps):
    cleaned = clean_step_texts(steps) if steps is not None else []
    formatted = []
    for i, s in enumerate(cleaned):
        if s:
            formatted.append(f"Step {i+1}: {s}")
        else:
            formatted.append(f"Step {i+1}:")
    return formatted

def _safe_int(v, default=-1):
    if isinstance(v, int):
        return v
    if isinstance(v, (float, str)):
        try:
            return int(float(v)) if isinstance(v, str) and "." in v else int(v)
        except (ValueError, TypeError):
            return default
    return default

# 解析编辑操作
# no_change / revert / keep_plan：不做修改
# replace_node(node_id, candidate_id)：替换节点工具
# insert_on_gap(gap_id, candidate_id)：在间隙插入新工具
def parse_edit_ops(llm_output):
    if not llm_output:
        return []
    edits = llm_output.get("edits") if isinstance(llm_output, dict) else None
    if not edits:
        return []
    ops = []
    for item in edits:
        if isinstance(item, dict):
            op = item.get("op")
            if op in ("revert", "no_change", "keep_plan"):
                ops.append({"op": "no_change", "raw": item})
            elif op == "replace_node":
                ops.append({"op": "replace_node", "node_id": _safe_int(item.get("node_id"), -1),
                            "candidate_id": _safe_int(item.get("candidate_id"), -1),
                            "step": item.get("step", "")})
            elif op == "insert_on_gap":
                ops.append({"op": "insert_on_gap", "gap_id": _safe_int(item.get("gap_id"), -1),
                            "candidate_id": _safe_int(item.get("candidate_id"), -1),
                            "step": item.get("step", "")})
            continue
        if not isinstance(item, str):
            continue
        s = item.strip()
        if s.startswith("revert") or s.startswith("no_change") or s.startswith("keep_plan"):
            ops.append({"op": "no_change", "raw": s})
            continue
        if s.startswith("replace_node"):
            try:
                inner = s[s.find("(") + 1:s.rfind(")")]
                node_id, cand_id = [int(x.strip()) for x in inner.split(",")]
                ops.append({"op": "replace_node", "node_id": node_id, "candidate_id": cand_id, "step": ""})
            except Exception:
                continue
        if s.startswith("insert_on_gap"):
            try:
                inner = s[s.find("(") + 1:s.rfind(")")]
                gap_id, cand_id = [int(x.strip()) for x in inner.split(",")]
                ops.append({"op": "insert_on_gap", "gap_id": gap_id, "candidate_id": cand_id, "step": ""})
            except Exception:
                continue
    return ops

# 验证编辑操作合法性
# 最多3个操作
# 节点ID必须在候选列表中
# 候选ID必须在有效范围内
# 不能重复使用同一个工具
# 每个操作必须有对应的步骤描述
def validate_edit_ops(ops, node_candidates, gap_candidates):
    errors = []
    if not ops:
        return True, []
    if any(op.get("op") == "no_change" for op in ops):
        return True, []
    if len(ops) > 3:
        errors.append("too_many_ops")
    node_map = {n["idx"]: n for n in node_candidates}
    gap_map = {g["gap_id"]: g for g in gap_candidates}
    chosen_tools = set()
    for op in ops:
        if op.get("op") == "replace_node":
            node_id = op.get("node_id")
            cand_id = op.get("candidate_id")
            if node_id not in node_map:
                errors.append(f"invalid_node_id:{node_id}")
                continue
            step_text = op.get("step", "")
            if not isinstance(step_text, str) or not step_text.strip():
                errors.append(f"missing_step_replace:{node_id}")
            cand_list = node_map[node_id].get("candidate_tools", [])
            if cand_id is None or cand_id < 0 or cand_id >= len(cand_list):
                errors.append(f"invalid_candidate_id:{cand_id}")
            else:
                tool = cand_list[cand_id]
                if tool in chosen_tools:
                    errors.append(f"duplicate_tool:{tool}")
                chosen_tools.add(tool)
        elif op.get("op") == "insert_on_gap":
            gap_id = op.get("gap_id")
            cand_id = op.get("candidate_id")
            if gap_id not in gap_map:
                errors.append(f"invalid_gap_id:{gap_id}")
                continue
            step_text = op.get("step", "")
            if not isinstance(step_text, str) or not step_text.strip():
                errors.append(f"missing_step_insert:{gap_id}")
            cand_list = gap_map[gap_id].get("candidate_tools", [])
            if cand_id is None or cand_id < 0 or cand_id >= len(cand_list):
                errors.append(f"invalid_candidate_id:{cand_id}")
            else:
                tool = cand_list[cand_id]
                if tool in chosen_tools:
                    errors.append(f"duplicate_tool:{tool}")
                chosen_tools.add(tool)
        else:
            errors.append(f"unknown_op:{op.get('op')}")
    return len(errors) == 0, errors

# 应用编辑操作
# 分离替换操作和插入操作
# 按顺序执行替换操作（更新节点工具和步骤）
# 按位置排序执行插入操作（在指定间隙插入新节点）
# 重新生成步骤描述和边连接
def apply_edit_ops(plan, ops, node_candidates, gap_candidates, tool_meta):
    if not ops or any(op.get("op") == "no_change" for op in ops):
        return plan
    nodes = list(plan.get("nodes", []))
    steps = list(plan.get("steps", []))
    node_map = {n["idx"]: n for n in node_candidates}
    gap_map = {g["gap_id"]: g for g in gap_candidates}
    desc_map = _tool_desc_map(tool_meta)

    replace_ops = [op for op in ops if op.get("op") == "replace_node"]
    insert_ops = [op for op in ops if op.get("op") == "insert_on_gap"]

    for op in sorted(replace_ops, key=lambda x: x.get("node_id", 0)):
        if op.get("op") == "replace_node":
            node_id = op["node_id"]
            cand_id = op["candidate_id"]
            cand_list = node_map[node_id].get("candidate_tools", [])
            new_tool = cand_list[cand_id]
            nodes[node_id] = new_tool
            step_text = op.get("step", "")
            if node_id < len(steps):
                steps[node_id] = step_text if step_text else steps[node_id]

    offset = 0
    def _gap_vpos(op):
        return gap_map.get(op.get("gap_id"), {}).get("v_pos", 0)

    for op in sorted(insert_ops, key=_gap_vpos):
        gap_id = op["gap_id"]
        cand_id = op["candidate_id"]
        gap = gap_map[gap_id]
        cand_list = gap.get("candidate_tools", [])
        new_tool = cand_list[cand_id]
        insert_pos = gap.get("v_pos", len(nodes)) + offset
        nodes.insert(insert_pos, new_tool)
        step_text = op.get("step", "")
        steps.insert(insert_pos, step_text if step_text else f"Step {insert_pos+1}: Call {new_tool}")
        offset += 1

    fixed_steps = []
    for i, tool in enumerate(nodes):
        if i < len(steps) and steps[i]:
            fixed_steps.append(steps[i])
        else:
            desc = desc_map.get(tool, "")
            fixed_steps.append(f"Step {i+1}: {desc[:100]}" if desc else f"Step {i+1}: Call {tool}")

    edges = [(i, i+1) for i in range(len(nodes) - 1)] if len(nodes) > 1 else []
    return {"nodes": nodes, "edges": edges, "steps": fixed_steps}


def build_tool_string(tool_meta):
    s = "# TASK LIST #:\n"
    for tool in tool_meta["nodes"]:
        s += json.dumps(tool, ensure_ascii=False) + "\n"
    return s


def _normalize_tool_key(name):
    if not name:
        return ""
    return "".join(c for c in name.lower() if c.isalnum())


def _build_tool_alias_map_from_nodes(nodes):
    alias = {}
    for node in nodes:
        tid = node.get("id")
        if tid:
            alias[_normalize_tool_key(tid)] = tid
    return alias


def normalize_tools_list(tools, alias_map):
    out = []
    for t in tools or []:
        norm = _normalize_tool_key(t)
        out.append(alias_map.get(norm, t))
    return out


def get_io_sets(tool_meta):
    cache_key = id(tool_meta)
    if cache_key in _IO_CACHE:
        return _IO_CACHE[cache_key]
    in_map = {}
    out_map = {}
    for node in tool_meta["nodes"]:
        tid = node["id"]
        in_types = node.get("input-type", [])
        out_types = node.get("output-type", [])
        if isinstance(in_types, str):
            in_types = [in_types] if in_types else []
        if isinstance(out_types, str):
            out_types = [out_types] if out_types else []
        in_map[tid] = set(in_types)
        out_map[tid] = set(out_types)
    _IO_CACHE[cache_key] = (in_map, out_map)
    return in_map, out_map


def io_compat(out_set, in_set):
    if not out_set or not in_set:
        return True 
    return len(out_set & in_set) > 0


def cosine_sim(a, b):
    if a is None or b is None:
        return 0.0
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def encode_text(cache, text, prefix=None):
    if not text:
        return None
    if prefix:
        embs = cache.encode_texts([text], prefix=prefix)
    else:
        embs = cache.encode_texts([text])
    return embs[0] if len(embs) > 0 else None


def links_to_edges(nodes, links):
    if not nodes:
        return []
    if not links:
        return [(i, i+1) for i in range(len(nodes)-1)] if len(nodes) > 1 else []

    if isinstance(links, list) and links and isinstance(links[0], dict):
        if "source" in links[0] and "target" in links[0]:
            links = [f'{x.get("source", "")}, {x.get("target", "")}' for x in links]

    if isinstance(links, list) and links and isinstance(links[0], (list, tuple)):
        edges = []
        for edge in links:
            if len(edge) == 2:
                u, v = edge[0], edge[1]
                if isinstance(u, int) and isinstance(v, int) and 0 <= u < len(nodes) and 0 <= v < len(nodes):
                    edges.append((u, v))
        return edges if edges else [(i, i+1) for i in range(len(nodes)-1)] if len(nodes) > 1 else []

    pos_map = {}
    for i, t in enumerate(nodes):
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
        best_score = float('inf')
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
    
    return edges if edges else [(i, i+1) for i in range(len(nodes)-1)] if len(nodes) > 1 else []


def add_start_edge(edges_tool_idx, num_nodes):
    if num_nodes <= 0:
        return []
    in_degree = [0] * num_nodes
    for u, v in edges_tool_idx:
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            in_degree[v] += 1
    root_idx = None
    for i in range(num_nodes):
        if in_degree[i] == 0:
            root_idx = i
            break
    edges_gnn = [(u + 1, v + 1) for (u, v) in edges_tool_idx]
    if root_idx is not None:
        edges_gnn.insert(0, (0, root_idx + 1))
    return edges_gnn

# 建议替换工具
# 从混淆矩阵中获取与当前工具易混淆的候选
# IO兼容性检查：候选工具的输入类型必须与前驱工具的输出类型兼容
# 计算多维度相似度分数：
#
#     对齐分数（如果有预训练对齐模型）
#     步骤-工具相似度（余弦相似度）
#     请求-工具相似度
#
# 返回Top-N候选及其理由
def suggest_replacement_tools(controller, tool_meta, confusion, current_nodes, current_steps,
                               node_idx, user_request="", topn=3):
    if node_idx < 0 or node_idx >= len(current_nodes):
        return []
    
    current_tool = current_nodes[node_idx]
    step_text = current_steps[node_idx] if node_idx < len(current_steps) else ""
    if step_text:
        step_text = clean_step_texts([step_text])[0]

    top_k = confusion.get("top_k", {}).get(current_tool, [])
    if not top_k:
        return []
    used_tools = set(current_nodes)
    in_map, out_map = get_io_sets(tool_meta)
    pred_out = set()
    succ_in = set()
    if node_idx > 0:
        pred_tool = current_nodes[node_idx - 1]
        pred_out = out_map.get(pred_tool, set())
    if node_idx < len(current_nodes) - 1:
        succ_tool = current_nodes[node_idx + 1]
        succ_in = in_map.get(succ_tool, set())

    cache = controller._embedding_cache
    step_emb = encode_text(cache, step_text, prefix="query") if step_text else None
    req_emb = encode_text(cache, user_request, prefix="query") if user_request else None
    align_q = None
    if controller.align_pretrained and step_text:
        with torch.no_grad():
            step_emb_t = controller._encode_text_to_emb_cached([step_text], prefix="query")
            align_q = controller._project_step(step_emb_t)[0]

    filtered = []
    for item in top_k:
        cand_tid = item[0] if isinstance(item, (list, tuple)) else item
        conf_score = item[1] if isinstance(item, (list, tuple)) and len(item) > 1 else 0.0

        if cand_tid == current_tool or cand_tid in used_tools:
            continue

        cand_in = in_map.get(cand_tid, set())
        cand_out = out_map.get(cand_tid, set())

        if pred_out and not io_compat(pred_out, cand_in):
            continue
        if succ_in and not io_compat(cand_out, succ_in):
            continue

        filtered.append((cand_tid, conf_score))

    if not filtered:
        return []

    filtered = sorted(filtered, key=lambda x: x[1], reverse=True)[:10]

    candidates = []
    for cand_tid, conf_score in filtered:
        q_proxy = 0.0
        align_score = None
        align_sig = None
        sim_req = 0.0
        if controller.align_pretrained and align_q is not None:
            with torch.no_grad():
                cand_emb_t = controller._get_tool_embedding(cand_tid)
                k = controller._project_tool(cand_emb_t.unsqueeze(0))[0]
                g = torch.dot(align_q, k) / float(controller.align_tau)
                align_score = float(g.item())
                align_sig = float(torch.sigmoid(g).item())
        else:
            if step_emb is not None:
                cand_emb = cache._tool_embeddings.get(cand_tid)
                if cand_emb is not None:
                    q_proxy = cosine_sim(step_emb, cand_emb)
        if req_emb is not None:
            cand_emb = cache._tool_embeddings.get(cand_tid)
            if cand_emb is not None:
                sim_req = cosine_sim(req_emb, cand_emb)

        step_tool_score = align_sig if align_sig is not None else q_proxy
        score = 0.5 * step_tool_score + 0.5 * sim_req
        if align_score is not None:
            reason = (
                f"IO compatible, align={align_score:.2f}, align_sig={step_tool_score:.2f}, "
                f"confusion={conf_score:.2f}, req_align={sim_req:.2f}"
            )
        else:
            reason = f"IO compatible, confusion={conf_score:.2f}, step_match={step_tool_score:.2f}, req_align={sim_req:.2f}"
        candidates.append({"tool": cand_tid, "reason": reason, "score": score})

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:topn]

# 建议插入工具
# IO桥接检查：候选工具的输入与前驱输出兼容，输出与后继输入兼容
# N-gram统计：利用训练数据中的工具共现频率
# 计算综合分数：60%请求相似度 + 20%前驱N-gram + 20%后继N-gram
# 根据间隙风险调整分数
def suggest_insertion_tools(controller, tool_meta, typed_ngrams, current_nodes, 
                            gap_u_pos, gap_v_pos, user_request, gap_risk=0.0, topn=3):
    if gap_v_pos < 0 or gap_v_pos >= len(current_nodes):
        return []
    
    in_map, out_map = get_io_sets(tool_meta)

    if gap_u_pos >= 0 and gap_u_pos < len(current_nodes):
        u_tool = current_nodes[gap_u_pos]
        u_out = out_map.get(u_tool, set())
    else:
        u_tool = "START"
        u_out = set() 
    
    v_tool = current_nodes[gap_v_pos]
    v_in = in_map.get(v_tool, set())
    cache = controller._embedding_cache
    req_emb = encode_text(cache, user_request, prefix="query") if user_request else None
    f2 = typed_ngrams.get("__f2__", {})
    candidates = []
    used_tools = set(current_nodes)
    all_tools = [n["id"] for n in tool_meta["nodes"] if n["id"] not in used_tools]
    
    for cand_tid in all_tools:
        if cand_tid in current_nodes:
            continue

        cand_in = in_map.get(cand_tid, set())
        cand_out = out_map.get(cand_tid, set())

        if u_out and not io_compat(u_out, cand_in):
            continue
        if v_in and not io_compat(cand_out, v_in):
            continue

        score_ng = 0.0
        if u_tool != "START":
            score_ng += math.log1p(f2.get((u_tool, cand_tid), 0))
        score_ng += math.log1p(f2.get((cand_tid, v_tool), 0))

        score_req = 0.0
        if req_emb is not None:
            cand_emb = cache._tool_embeddings.get(cand_tid)
            if cand_emb is not None:
                score_req = cosine_sim(req_emb, cand_emb)

        score = 0.6 * score_req + 0.2 * math.log1p(f2.get((u_tool, cand_tid), 0)) + 0.2 * math.log1p(f2.get((cand_tid, v_tool), 0))
        score = score * (1.0 + 0.5 * float(gap_risk))
        
        reason = f"IO bridge OK, ngram={score_ng:.2f}, req_align={score_req:.2f}"
        candidates.append({"tool": cand_tid, "reason": reason, "score": score})

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:topn]


def format_candidates_for_prompt(candidates):
    if not candidates:
        return "[]"
    formatted = []
    for c in candidates:
        formatted.append({"tool": c["tool"], "score": round(c["score"], 3)})
    return json.dumps(formatted, ensure_ascii=False)


def build_tool_gap_risks(gnn_result, num_tools, current_nodes=None):
    gaps = gnn_result.get("gaps", []) or []
    gap_risks = gnn_result.get("gap_risks", []) or []
    out = []
    
    for i, edge in enumerate(gaps):
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            continue
        u_idx, v_idx = edge
        if not isinstance(u_idx, int) or not isinstance(v_idx, int):
            continue

        risk = float(gap_risks[i]) if i < len(gap_risks) else 0.0

        if u_idx == 0:
            u_name = "START"
            u_pos = -1
        else:
            u_pos = u_idx - 1
            if 0 <= u_pos < num_tools and current_nodes:
                u_name = current_nodes[u_pos]
            else:
                continue
        
        v_pos = v_idx - 1
        if 0 <= v_pos < num_tools and current_nodes:
            v_name = current_nodes[v_pos]
        else:
            continue
        
        out.append((u_name, v_name, risk, u_pos, v_pos))

    out.sort(key=lambda x: x[2], reverse=True)
    return out


def build_full_gnn_report(gnn_result, current_plan):
    nodes = current_plan.get("nodes", [])
    edges = current_plan.get("edges", [])
    steps = current_plan.get("steps", [])
    num_tools = len(nodes)

    node_risks = gnn_result.get("node_risks", []) or []
    gap_risks = gnn_result.get("gap_risks", []) or []
    gaps = gnn_result.get("gaps", []) or []

    gap_named = []
    for i, edge in enumerate(gaps):
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            continue
        u_idx, v_idx = edge
        if not isinstance(u_idx, int) or not isinstance(v_idx, int):
            continue
        if u_idx == 0:
            u_name = "START"
        else:
            u_pos = u_idx - 1
            u_name = nodes[u_pos] if 0 <= u_pos < num_tools else f"IDX_{u_idx}"
        v_pos = v_idx - 1
        v_name = nodes[v_pos] if 0 <= v_pos < num_tools else f"IDX_{v_idx}"
        risk = float(gap_risks[i]) if i < len(gap_risks) else 0.0
        gap_named.append({"u": u_name, "v": v_name, "risk": risk, "u_idx": u_idx, "v_idx": v_idx})

    return {
        "S": float(gnn_result.get("S", 0.0)),
        "nodes": nodes,
        "edges": edges,
        "steps": steps,
        "node_risks": node_risks,
        "gap_risks": gap_risks,
        "gaps": gaps,
        "gaps_named": gap_named
    }


def robust_json_extract(text):
    if not text:
        return None
    
    text = text.strip()

    if "```json" in text:
        match = text.split("```json")[1].split("```")[0] if "```json" in text else text
        text = match.strip()
    elif "```" in text:
        match = text.split("```")[1].split("```")[0] if text.count("```") >= 2 else text
        text = match.strip()

    if "{" in text and "}" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        text = text[start:end]

    text = text.replace("\\_", "_")
    text = text.replace(",]", "]").replace(",}", "}")
    
    try:
        return json.loads(text)
    except:
        return None

# 调用LLM进行计划修正（核心函数）
# Prompt构建流程：
#
#     构建工具列表字符串
#     格式化当前计划（节点、步骤、链接）
#     提取高风险节点和间隙，生成候选工具
#     构建完整的GNN评估报告
#     组合成结构化Prompt发送给LLM
#
# 关键设计：
#
#     提供详细的GNN诊断信息（分数、风险值）
#     明确候选工具列表（限制LLM选择范围）
#     强制JSON输出格式
#     支持缓存机制避免重复调用
def call_llm_patch(user_request, tool_meta, confusion, typed_ngrams, controller,
                   current_plan, gnn_result,
                   theta_node=THETA_NODE, theta_gap=THETA_GAP,
                   temperature=0.2, max_tokens=1200, llm_cache=None,
                   cache_key_prefix=None, trace_payload=None):
    tool_list_str = build_tool_string(tool_meta)
    
    current_nodes = current_plan.get("nodes", [])
    current_steps = current_plan.get("steps", [])
    current_edges = current_plan.get("edges", [])

    current_plan_json = {
        "task_steps": _format_steps_for_prompt(current_steps),
        "task_nodes": current_nodes,
        "task_links": [[e[0], e[1]] for e in current_edges]
    }
    
    S = gnn_result.get("S", 0.0)
    node_risks = gnn_result.get("node_risks", [])

    indexed_risks = [(idx, risk) for idx, risk in enumerate(node_risks)]
    indexed_risks.sort(key=lambda x: x[1], reverse=True)
    high_nodes = [item for item in indexed_risks if item[1] >= theta_node]

    node_diag_lines = []
    node_candidates = []
    for idx, risk in high_nodes:
        if 0 <= idx < len(current_nodes):
            tool = current_nodes[idx]
            step = current_steps[idx] if idx < len(current_steps) else ""
            candidates = suggest_replacement_tools(
                controller, tool_meta, confusion, current_nodes, current_steps, idx,
                user_request=user_request, topn=3
            )
            cand_str = format_candidates_for_prompt(candidates)
            cand_list = [c["tool"] for c in candidates]
            node_diag_lines.append(
                f"  - Node {idx} ({tool}): risk={risk:.3f}, step=\"{step[:40]}...\"\n"
                f"    replacement_candidates: {cand_str}"
            )
            node_candidates.append({
                "idx": idx,
                "tool": tool,
                "risk": float(risk),
                "step": step,
                "candidates": candidates,
                "candidate_tools": cand_list
            })
    node_diag_str = "\n".join(node_diag_lines) if node_diag_lines else "  None"

    num_tools = len(current_nodes)
    tool_gap_risks = build_tool_gap_risks(gnn_result, num_tools, current_nodes)
    high_gaps = [item for item in tool_gap_risks if item[2] >= theta_gap]

    gap_diag_lines = []
    gap_candidates = []
    for item in high_gaps:
        u_name, v_name, risk, u_pos, v_pos = item
        candidates = suggest_insertion_tools(
            controller, tool_meta, typed_ngrams, current_nodes, u_pos, v_pos,
            user_request, gap_risk=risk, topn=3
        )
        cand_str = format_candidates_for_prompt(candidates)
        cand_list = [c["tool"] for c in candidates]
        gap_diag_lines.append(
            f"  - Gap ({u_name} -> {v_name}): risk={risk:.3f}\n"
            f"    insertion_candidates: {cand_str}"
        )
        gap_candidates.append({
            "gap_id": len(gap_candidates),
            "u_name": u_name,
            "v_name": v_name,
            "risk": float(risk),
            "u_pos": u_pos,
            "v_pos": v_pos,
            "candidates": candidates,
            "candidate_tools": cand_list
        })
    gap_diag_str = "\n".join(gap_diag_lines) if gap_diag_lines else "  None"

    if node_candidates:
        node_summary_str = " ".join(
            [
                f"Node {n['idx']} ({n['tool']}) has a high risk score ({n['risk']:.2f}), suggesting it might be irrelevant or incorrect for the user request."
                for n in node_candidates
            ]
        )
    else:
        node_summary_str = "No high-risk nodes detected."
    if gap_candidates:
        gap_summary_str = " ".join(
            [
                f"Gap {g['gap_id']} ({g['u_name']} -> {g['v_name']}) has a high risk score ({g['risk']:.2f}), suggesting a missing step between these nodes."
                for g in gap_candidates
            ]
        )
    else:
        gap_summary_str = "No high-risk gaps detected."
    
    full_gnn_report = build_full_gnn_report(gnn_result, current_plan)
    prompt = f"""{tool_list_str}

# USER REQUEST #
{user_request}

# CURRENT PLAN (JSON) #
{json.dumps(current_plan_json, ensure_ascii=False, indent=2)}

# GNN EVALUATION #
The GNN is a plan evaluator and reports:
- Graph score S ∈ [0,1]: higher suggests stronger alignment with the user request.
- Node risk ∈ [0,1]: higher indicates a node may use an incorrect tool.
- Gap risk ∈ [0,1]: higher indicates a gap (including START→first node and tool-to-tool edges) may be incomplete and need an inserted tool.

# TOP RISK NODES #
{node_diag_str}

# TOP RISK GAPS #
{gap_diag_str}

# RISK SUMMARY #
Node summary: {node_summary_str}
Gap summary: {gap_summary_str}

# FULL GNN ANALYSIS (JSON) #
{json.dumps(full_gnn_report, ensure_ascii=False)}

# GOAL #
You are a plan refinement assistant. Analyze the user request, the current plan, and the GNN diagnostics to decide whether and how to improve the plan so it better satisfies the request.

# Common errors are mainly two types #
1) Wrong tool choice: a tool is semantically similar but incorrect → replace the node.
2) Missing step: especially missing preprocessing → insert a tool on the risky gap.

# TASK #
Think step by step internally. Based on the user request, the current plan, and the GNN diagnostics, first decide whether any change is necessary.
If no change is needed, return empty edits. If changes are needed, select replacement/insertion tools only from the provided candidates and propose minimal edits.
If improvements are not clear with the provided candidates, do not modify.
Return EDIT OPERATIONS only (no analysis text). It is valid to return no edits.

# RULES #
1. Output JSON ONLY (no extra text).
2. Modify at most 3 places; 0 is allowed.
3. Do not use the same candidate tool in multiple edits.
4. Allowed ops:
   - replace_node(node_id, candidate_id, step)
   - insert_on_gap(gap_id, candidate_id, step)
   - no_change()
5. candidate_id must be an integer index from the candidate list.
6. Candidate order is arbitrary; read each candidate's tool description in tool_list_str and compare carefully.
7. insert_on_gap must use the gap_id from the list below and only inserts between (u_id, v_id).
8. replace_node must use node_id from the list below and only replaces the tool at that node.
9. Each node/gap may remain unchanged; prefer fewer edits and only change when necessary.
10. For every edit, provide a new step text aligned with the chosen tool and request.
11. The updated steps/tools should solve the request better than the current plan; otherwise do not modify.
12. For any edits, keep steps aligned 1-to-1 with nodes (same count, same order).

# CANDIDATES (node_id -> candidate_id -> tool) #
{json.dumps([{ "node_id": n["idx"], "candidates": n["candidate_tools"] } for n in node_candidates], ensure_ascii=False)}

# CANDIDATES (gap_id -> candidate_id -> tool) #
{json.dumps([{ "gap_id": g["gap_id"], "u_pos": g["u_pos"], "v_pos": g["v_pos"], "candidates": g["candidate_tools"] } for g in gap_candidates], ensure_ascii=False)}

# OUTPUT FORMAT (minimal edits) #
{{
  "edits": [
    {{"op":"replace_node","node_id":0,"candidate_id":1,"step":"Step 1: ..."}},
    {{"op":"insert_on_gap","gap_id":0,"candidate_id":2,"step":"Step 2: ..."}}
  ]
}}
If the current workflow is already optimal, return: {{"edits":[]}}
"""

    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if trace_payload is not None:
        trace_payload["prompt_hash"] = prompt_hash
        trace_payload["node_candidates"] = node_candidates
        trace_payload["gap_candidates"] = gap_candidates
    cache_key = None
    if llm_cache is not None and cache_key_prefix is not None:
        cache_key = (cache_key_prefix[0], cache_key_prefix[1], prompt_hash)
        if isinstance(llm_cache, dict):
            cached = llm_cache.get(cache_key)
        else:
            cached = llm_cache.get(cache_key) if hasattr(llm_cache, "get") else None
        if cached is not None:
            return cached

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            _throttle_llm_calls()
            response = client.chat.completions.create(
                model=LLM_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            content = response.choices[0].message.content.strip()
            result = robust_json_extract(content)

            if not result:
                retry_prompt = "只输出严格JSON，不要任何解释：\n" + prompt
                _throttle_llm_calls()
                response = client.chat.completions.create(
                    model=LLM_NAME,
                    messages=[{"role": "user", "content": retry_prompt}],
                    temperature=0.0,
                    max_tokens=max_tokens
                )
                result = robust_json_extract(response.choices[0].message.content)

            if result is not None:
                if llm_cache is not None and cache_key is not None:
                    if isinstance(llm_cache, dict):
                        llm_cache[cache_key] = result
                    elif hasattr(llm_cache, "set"):
                        llm_cache.set(cache_key, result, data_id=cache_key[0],
                                      strategy=cache_key[1], prompt_hash=prompt_hash)
                return result
        except Exception:
            if attempt == LLM_MAX_RETRIES:
                break
        time.sleep(min(LLM_RETRY_BASE_SEC * (2 ** (attempt - 1)), LLM_RETRY_MAX_SEC))
    return None

# 功能：在初次生成的编辑操作不合法时，用错误信息提示LLM重新生成
def call_llm_patch_fix(user_request, error_msg, node_candidates, gap_candidates,
                       temperature=0.2, max_tokens=600):
    prompt = f"""# USER REQUEST #
{user_request}

# ERROR #
{error_msg}

# CANDIDATES (node_id -> candidate_id -> tool) #
{json.dumps([{ "node_id": n["idx"], "candidates": n["candidate_tools"] } for n in node_candidates], ensure_ascii=False)}

# CANDIDATES (gap_id -> candidate_id -> tool) #
{json.dumps([{ "gap_id": g["gap_id"], "u_pos": g["u_pos"], "v_pos": g["v_pos"], "candidates": g["candidate_tools"] } for g in gap_candidates], ensure_ascii=False)}

# TASK #
Fix the edit operations to satisfy all constraints.
Only output:
{{"edits":[
  {{"op":"replace_node","node_id":0,"candidate_id":1,"step":"Step 1: ..."}},
  {{"op":"insert_on_gap","gap_id":0,"candidate_id":2,"step":"Step 2: ..."}}
]}}
Or {{\"edits\":[]}} / no_change().
"""
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            _throttle_llm_calls()
            response = client.chat.completions.create(
                model=LLM_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            content = response.choices[0].message.content.strip()
            result = robust_json_extract(content)
            if result:
                return result
        except Exception:
            if attempt == LLM_MAX_RETRIES:
                break
        time.sleep(min(LLM_RETRY_BASE_SEC * (2 ** (attempt - 1)), LLM_RETRY_MAX_SEC))
    return None

# 迭代优化（核心流程）
# ┌─────────────────────────────────────────┐
# │  初始计划 (init_plan)                   │
# │  GNN评分 S                              │
# │  S ≥ TAU_ACCEPT? ──Yes──► 直接接受      │
# │      │                                  │
# │     No                                  │
# │      ▼                                  │
# │  LLM分析高风险节点/间隙                   │
# │  生成编辑操作 (replace/insert)          │
# │  验证并应用编辑                         │
# │  重新排序和连接边                       │
# │  GNN重新评分 S'                         │
# │  稳定性检查 (对比GT质量不下降)            │
# │  S' > S + DELTA_IMPROVE? ──Yes──► 接受  │
# │      │                                  │
# │     No                                  │
# │      ▼                                  │
# │  回滚到初始计划 (rollback)               │
# └─────────────────────────────────────────┘
def iterative_refine_with_llm(controller, user_request, init_plan, init_gnn,
                                  tool_meta, confusion, typed_ngrams, allowed_tools,
                                  threshold_accept=TAU_ACCEPT,
                                  llm_temperature=0.2, llm_cache=None,
                                  data_id=None, gt_nodes=None, gt_links=None,
                                  alias_map=None):
    history = {"init": {}}
    gt_payload = None
    if gt_nodes is not None or gt_links is not None:
        gt_payload = {
            "gt_nodes": gt_nodes,
            "gt_links": gt_links
        }
    
    base_S = float(init_gnn.get("S", 0.0))
    candidates = [(base_S, init_plan, init_gnn, "init")]

    best_plan = init_plan
    best_gnn = init_gnn
    best_S = base_S
    
    history["init"] = {
        "strategy": "init",
        "S": base_S,
        "nodes": init_plan.get("nodes", []),
        "steps": init_plan.get("steps", []),
        "edges": init_plan.get("edges", [])
    }

    if float(best_gnn.get("S", 0.0)) >= threshold_accept:
        return best_plan, "accept", history

    strategy = "patch"
    trace_payload = {}
    llm_output = call_llm_patch(
        user_request, tool_meta, confusion, typed_ngrams, controller,
        best_plan, best_gnn,
        theta_node=THETA_NODE,
        theta_gap=THETA_GAP,
        temperature=llm_temperature,
        llm_cache=llm_cache,
        cache_key_prefix=(data_id, "patch") if data_id is not None else None,
        trace_payload=trace_payload
    )

    if not llm_output:
        return best_plan, "rollback", history

    ops = parse_edit_ops(llm_output)
    node_candidates = trace_payload.get("node_candidates", []) or []
    gap_candidates = trace_payload.get("gap_candidates", []) or []
    trace_payload["edit_ops"] = ops
    valid, errors = validate_edit_ops(ops, node_candidates, gap_candidates)
    if not valid:
        fix = call_llm_patch_fix(
            user_request,
            ";".join(errors),
            node_candidates,
            gap_candidates,
            temperature=llm_temperature
        )
        if fix:
            ops = parse_edit_ops(fix)
            valid, errors = validate_edit_ops(ops, node_candidates, gap_candidates)
    if not valid:
        return best_plan, "rollback", history
    new_plan = apply_edit_ops(best_plan, ops, node_candidates, gap_candidates, tool_meta)
    if new_plan is None:
        return best_plan, "rollback", history

    if not new_plan["nodes"]:
        return best_plan, "rollback", history

    ordered_tools, ordered_steps, ordered_edges = order_chain_with_steps_and_edges(
        new_plan["nodes"], new_plan.get("steps", []), new_plan.get("edges", [])
    )
    new_plan["nodes"] = ordered_tools
    new_plan["steps"] = ordered_steps
    new_plan["edges"] = ordered_edges

    new_edges_gnn = add_start_edge(new_plan["edges"], len(new_plan["nodes"])) if new_plan["nodes"] else []
    new_gnn = controller.score_chain(
        new_plan["nodes"], user_request,
        edges=new_edges_gnn,
        step_texts=new_plan["steps"]
    )
    new_S = float(new_gnn.get("S", 0.0))

    quality_before = quality_after = None
    if gt_nodes is not None and gt_links is not None:
        quality_before = compute_f1_set_based(best_plan.get("nodes", []), gt_nodes)
        quality_after = compute_f1_set_based(new_plan.get("nodes", []), gt_nodes)

    stability_ok = True
    if quality_before is not None and quality_after is not None:
        stability_ok = (quality_after - quality_before) >= 0.0

    if not stability_ok:
        return best_plan, "rollback", history

    candidates.append((new_S, new_plan, new_gnn, strategy))
    history["patch"] = {
        "strategy": strategy,
        "S": new_S,
        "nodes": new_plan["nodes"],
        "steps": new_plan["steps"],
        "edges": new_plan["edges"]
    }

    candidates.sort(key=lambda x: x[0], reverse=True)
    final_S, final_plan, final_gnn, final_strategy = candidates[0]

    if final_S < base_S + DELTA_IMPROVE:
        final_plan = init_plan
        final_strategy = "rollback"
    
    return final_plan, final_strategy, history


def stage1_candidate_thresholds(val_stats, base_percentile=0.7,
                                grid_min=0.5, grid_max=1.0, grid_step=0.02):
    if not val_stats:
        return TAU_ACCEPT, [TAU_ACCEPT]

    scores = sorted([float(x["S"]) for x in val_stats])
    if len(scores) < 3:
        return TAU_ACCEPT, [TAU_ACCEPT]

    idx = int(len(scores) * base_percentile)
    idx = min(max(idx, 0), len(scores) - 1)
    base_t = scores[idx]

    steps = int(round((grid_max - grid_min) / grid_step)) + 1
    thresholds = [round(grid_min + i * grid_step, 4) for i in range(steps)]
    thresholds = [t for t in thresholds if grid_min <= t <= grid_max]
    return base_t, thresholds


def _f1_binary(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

# 搜索最优风险阈值
# 方法：网格搜索（0.1到0.9，步长0.05），最大化F1分数
def search_risk_thresholds(val_ids, val_content, controller, t_accept,
                           alias_map=None):
    node_risks_all = []
    node_labels_all = []
    gap_risks_all = []
    gap_labels_all = []

    for data_id in val_ids:
        content = val_content[data_id]
        user_request = content.get("user_request", "")

        pred_nodes_raw = normalize_tools_list(content.get("pred_task_nodes", []), alias_map) if alias_map else content.get("pred_task_nodes", [])
        pred_links_raw = content.get("pred_task_links", [])
        pred_steps = content.get("steps", None) or content.get("pred_task_steps", [])
        pred_tools, pred_steps, pred_edges_tool = order_chain_with_steps_and_edges(
            pred_nodes_raw, pred_steps, pred_links_raw
        )
        pred_edges_gnn = add_start_edge(pred_edges_tool, len(pred_tools)) if pred_tools else []
        pred_gnn = controller.score_chain(
            pred_tools, user_request, edges=pred_edges_gnn, step_texts=pred_steps
        ) if pred_tools else {"node_risks": [], "gap_risks": [], "gaps": []}
        pred_S = float(pred_gnn.get("S", 0.0))
        if pred_S >= t_accept:
            continue

        gt_nodes_raw = normalize_tools_list(content.get("gt_task_nodes", []), alias_map) if alias_map else content.get("gt_task_nodes", [])
        gt_links_raw = content.get("gt_task_links", [])
        gt_steps = content.get("gt_task_steps", [])
        gt_tools, gt_steps, gt_edges_tool = order_chain_with_steps_and_edges(
            gt_nodes_raw, gt_steps, gt_links_raw
        )

        gt_tool_set = set(gt_tools)
        gt_next = {gt_tools[i]: gt_tools[i + 1] for i in range(len(gt_tools) - 1)}
        gt_prev = {gt_tools[i + 1]: gt_tools[i] for i in range(len(gt_tools) - 1)}
        if gt_tools:
            gt_prev[gt_tools[0]] = "START"

        for i, tool in enumerate(pred_tools):
            risk = float(pred_gnn.get("node_risks", [])[i]) if i < len(pred_gnn.get("node_risks", [])) else 0.0
            label = 0 if tool in gt_tool_set else 1
            node_risks_all.append(risk)
            node_labels_all.append(label)

        tool_gap_risks = build_tool_gap_risks(pred_gnn, len(pred_tools), pred_tools)
        for (u_name, v_name, risk, _u_pos, _v_pos) in tool_gap_risks:
            risk = float(risk)
            gap_bad = False
            if u_name == "START":
                if not gt_tools or gt_tools[0] != v_name:
                    gap_bad = True
            elif u_name in gt_tool_set and gt_next.get(u_name) != v_name:
                gap_bad = True
            if v_name in gt_tool_set and gt_prev.get(v_name) != u_name:
                gap_bad = True
            label = 1 if gap_bad else 0
            gap_risks_all.append(risk)
            gap_labels_all.append(label)

    if not node_risks_all:
        return THETA_NODE, THETA_GAP, {"node_grid": [], "gap_grid": [], "t_accept": t_accept}

    thresholds = [round(x, 2) for x in np.arange(0.1, 0.91, 0.05)]
    best_node_t, best_node_f1 = THETA_NODE, -1.0
    node_grid = []
    for t in thresholds:
        y_pred = [1 if r >= t else 0 for r in node_risks_all]
        f1 = _f1_binary(node_labels_all, y_pred)
        node_grid.append({"threshold": t, "f1": f1})
        if f1 > best_node_f1:
            best_node_f1 = f1
            best_node_t = t

    best_gap_t, best_gap_f1 = THETA_GAP, -1.0
    gap_grid = []
    for t in thresholds:
        y_pred = [1 if r >= t else 0 for r in gap_risks_all]
        f1 = _f1_binary(gap_labels_all, y_pred)
        gap_grid.append({"threshold": t, "f1": f1})
        if f1 > best_gap_f1:
            best_gap_f1 = f1
            best_gap_t = t

    return best_node_t, best_gap_t, {
        "node_grid": node_grid,
        "gap_grid": gap_grid,
        "t_accept": t_accept,
        "node_samples": len(node_risks_all),
        "gap_samples": len(gap_risks_all)
    }

# 搜索最优接受阈值
# 优化策略：
#     对每个验证样本，预计算LLM修正后的计划
#     网格搜索不同的T_accept阈值
#     选择使Node-F1和Link-F1平均值最大的阈值
def stage2_search_thresholds_with_llm(val_ids, val_content, controller, tool_meta, confusion,
                                      typed_ngrams, allowed_tools, thresholds,
                                      max_samples=None, alias_map=None, base_t=None,
                                      fallback_min=0.7):
    if not thresholds:
        return TAU_ACCEPT

    sample_ids = val_ids[:max_samples] if max_samples else val_ids
    best_score = -1.0
    best_t = thresholds[0]

    llm_cache = {}
    patch_cache = {}

    for data_id in sample_ids:
        content = val_content[data_id]
        user_request = content.get("user_request", "")

        pred_nodes_raw = normalize_tools_list(content.get("pred_task_nodes", []), alias_map) if alias_map else content.get("pred_task_nodes", [])
        pred_links_raw = content.get("pred_task_links", [])
        pred_steps = content.get("steps", None) or content.get("pred_task_steps", [])
        pred_tools, pred_steps, pred_edges_tool = order_chain_with_steps_and_edges(
            pred_nodes_raw, pred_steps, pred_links_raw
        )
        pred_edges_gnn = add_start_edge(pred_edges_tool, len(pred_tools)) if pred_tools else []

        base_plan = {"nodes": pred_tools, "edges": pred_edges_tool, "steps": pred_steps if isinstance(pred_steps, list) else []}
        base_gnn = controller.score_chain(
            pred_tools, user_request,
            edges=pred_edges_gnn,
            step_texts=pred_steps
        ) if pred_tools else {"S": 0.0, "node_risks": [], "gap_risks": [], "gaps": []}

        trace_payload = {}
        llm_output = call_llm_patch(
            user_request, tool_meta, confusion, typed_ngrams, controller,
            base_plan, base_gnn,
            temperature=LLM_REFINE_TEMPERATURE,
            llm_cache=llm_cache,
            cache_key_prefix=(data_id, "patch"),
            trace_payload=trace_payload
        )
        if not llm_output:
            patch_cache[data_id] = (base_plan, base_gnn)
            continue

        ops = parse_edit_ops(llm_output)
        trace_payload["edit_ops"] = ops
        node_candidates = trace_payload.get("node_candidates", []) or []
        gap_candidates = trace_payload.get("gap_candidates", []) or []
        valid, errors = validate_edit_ops(ops, node_candidates, gap_candidates)
        if not valid:
            fix = call_llm_patch_fix(
                user_request,
                ";".join(errors),
                node_candidates,
                gap_candidates,
                temperature=LLM_REFINE_TEMPERATURE
            )
            if fix:
                ops = parse_edit_ops(fix)
                valid, errors = validate_edit_ops(ops, node_candidates, gap_candidates)
        if not valid:
            patch_cache[data_id] = (base_plan, base_gnn)
            continue

        new_plan = apply_edit_ops(base_plan, ops, node_candidates, gap_candidates, tool_meta)
        if new_plan is None:
            patch_cache[data_id] = (base_plan, base_gnn)
            continue

        ordered_tools, ordered_steps, ordered_edges = order_chain_with_steps_and_edges(
            new_plan["nodes"], new_plan.get("steps", []), new_plan.get("edges", [])
        )
        new_plan["nodes"] = ordered_tools
        new_plan["steps"] = ordered_steps
        new_plan["edges"] = ordered_edges
        new_edges_gnn = add_start_edge(new_plan["edges"], len(new_plan["nodes"])) if new_plan["nodes"] else []
        new_gnn = controller.score_chain(
            new_plan["nodes"], user_request,
            edges=new_edges_gnn,
            step_texts=new_plan["steps"]
        )
        patch_cache[data_id] = (new_plan, new_gnn)

    for idx, t_accept in enumerate(thresholds, start=1):
        node_f1_list, link_f1_list = [], []
        for data_id in sample_ids:
            content = val_content[data_id]
            gt_nodes_raw = normalize_tools_list(content.get("gt_task_nodes", []), alias_map) if alias_map else content.get("gt_task_nodes", [])
            gt_links_raw = content.get("gt_task_links", [])
            gt_links = edges_to_links(gt_nodes_raw, links_to_edges(gt_nodes_raw, gt_links_raw))

            pred_nodes_raw = normalize_tools_list(content.get("pred_task_nodes", []), alias_map) if alias_map else content.get("pred_task_nodes", [])
            pred_links_raw = content.get("pred_task_links", [])
            pred_steps = content.get("steps", None) or content.get("pred_task_steps", [])
            pred_tools, pred_steps, pred_edges_tool = order_chain_with_steps_and_edges(
                pred_nodes_raw, pred_steps, pred_links_raw
            )
            pred_edges_gnn = add_start_edge(pred_edges_tool, len(pred_tools)) if pred_tools else []
            base_gnn = controller.score_chain(
                pred_tools, content.get("user_request", ""),
                edges=pred_edges_gnn,
                step_texts=pred_steps
            ) if pred_tools else {"S": 0.0}
            base_S = float(base_gnn.get("S", 0.0))

            if base_S >= t_accept:
                final_plan = {"nodes": pred_tools, "edges": pred_edges_tool}
            else:
                final_plan, _ = patch_cache.get(data_id, ({"nodes": pred_tools, "edges": pred_edges_tool}, None))

            pred_links = edges_to_links(final_plan.get("nodes", []), final_plan.get("edges", []))
            node_f1 = compute_f1_set_based(final_plan.get("nodes", []), gt_nodes_raw)
            link_f1 =compute_f1_set_based(pred_links, gt_links)
            node_f1_list.append(node_f1)
            link_f1_list.append(link_f1)

        avg_node_f1 = sum(node_f1_list) / len(node_f1_list) if node_f1_list else 0.0
        avg_link_f1 = sum(link_f1_list) / len(link_f1_list) if link_f1_list else 0.0
        score = 0.5 * avg_node_f1 + 0.5 * avg_link_f1

        if score > best_score:
            best_score = score
            best_t = t_accept

    if base_t is not None and best_t < fallback_min:
        return base_t
    return best_t

# | 阶段       | 功能                   | 关键操作                                                              |
# | -------- | -------------------- | ----------------------------------------------------------------- |
# | **数据准备** | 加载数据集，划分训练/验证集       | `prepare_training_ids()`, 10%验证集                                  |
# | **特征工程** | 构建N-gram统计、混淆矩阵、IO类型 | `build_typed_ngrams()`, `build_confusion_prior()`                 |
# | **模型训练** | GNN模型训练（对比学习）        | `ModelTrainer.train()`                                            |
# | **阈值搜索** | 在验证集上搜索最优阈值          | `search_risk_thresholds()`, `stage2_search_thresholds_with_llm()` |
# | **测试推理** | 三阶段推理流程              | 见下                                                                |

# Phase 1: GNN评分
#     └── 对所有直接预测结果用GNN打分
#     └── 记录节点风险、间隙风险、总分S
#
# Phase 2: LLM精细化修正
#     └── S < T_accept 的样本调用 iterative_refine_with_llm()
#     └── 生成修正后的计划
#
# Phase 3: 评估
#     └── 对比 Direct vs Refined 的 Node-F1, Link-F1, Accuracy

# ┌─────────────────────────────────────────────────────────────┐
# │                         输入层                              │
# │  用户请求 + 候选工具列表 + 直接预测结果（LLM生成）              │
# └─────────────────────────────────────────────────────────────┘
#                               ▼
# ┌─────────────────────────────────────────────────────────────┐
# │                      GNN评估层 (ModelTrainer)               │
# │  • 图神经网络评分 S ∈ [0,1]                                  │
# │  • 节点风险 node_risks（工具选择是否正确）                    │
# │  • 间隙风险 gap_risks（是否需要插入新工具）                   │
# └─────────────────────────────────────────────────────────────┘
#                               ▼
#                     ┌─────────────────┐
#               S ≥ TAU_ACCEPT?          No
#                     │                   │
#                    Yes                  ▼
#                     │           ┌─────────────────┐
#                     │           │   候选生成模块   │
#                     │           │ • 替换候选：IO兼容+相似度 │
#                     │           │ • 插入候选：N-gram+IO桥接 │
#                     │           └─────────────────┘
#                     │                   │
#                     ▼                   ▼
#             ┌─────────────┐     ┌─────────────────┐
#             │   直接接受    │     │   LLM修正模块    │
#             └─────────────┘     │ • 分析GNN诊断    │
#                                 │ • 选择候选工具    │
#                                 │ • 生成编辑操作    │
#                                 └─────────────────┘
#                                         │
#                                         ▼
#                                 ┌─────────────────┐
#                                 │   应用编辑操作   │
#                                 │ • 替换节点      │
#                                 │ • 插入间隙      │
#                                 └─────────────────┘
#                                         │
#                                         ▼
#                                 ┌─────────────────┐
#                                 │   重新GNN评分    │
#                                 │ 稳定性检查       │
#                                 │ 质量改进检查     │
#                                 └─────────────────┘

# | 组件         | 创新                       |
# | ---------- | ------------------------ |
# | **GNN评估器** | 学习规划的结构质量，识别错误工具和缺失步骤    |
# | **LLM修正器** | 利用LLM的推理能力，在有限候选空间内做决策   |
# | **IO兼容性**  | 利用工具的输入输出类型约束，减少搜索空间     |
# | **对齐预训练**  | 学习步骤文本与工具描述的语义对齐         |
# | **两阶段阈值**  | 分离风险阈值（F1优化）和接受阈值（端到端优化） |
if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    
    # Dataset and training
    parser.add_argument('--dataset', type=str, default='ultratool', choices=['huggingface', 'multimedia', 'dailylife', 'tmdb', 'ultratool'])
    parser.add_argument('--train_num', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--lambda_rank', type=float, default=1.0)
    parser.add_argument('--margin_rank', type=float, default=0.2)
    parser.add_argument('--lambda_graph', type=float, default=2.0)
    parser.add_argument('--lambda_gap', type=float, default=1.5)
    
    # LM related
    parser.add_argument('--lm_name', type=str, default='intfloat/e5-large')

    parser.add_argument('--gnn_hidden_dim', type=int, default=1024)
    parser.add_argument('--gnn_layer', type=int, default=3)

    parser.add_argument('--llm_temperature', type=float, default=0.2)
    parser.add_argument('--llm_name', type=str, default='deepseek-chat',
                        help="Prediction folder name and refinement LLM model name.")

    # Step-tool alignment pretrain
    parser.add_argument('--align_pretrain_epochs', type=int, default=0)
    parser.add_argument('--align_lr', type=float, default=1e-4)
    parser.add_argument('--align_dim', type=int, default=1024)
    parser.add_argument('--align_tau', type=float, default=0.07)
    parser.add_argument('--cost_tau', type=float, default=0.6)
    parser.add_argument('--align_hard_k', type=int, default=5)
    parser.add_argument('--align_rand_k', type=int, default=5)
    parser.add_argument('--align_batch', type=int, default=64)
    parser.add_argument('--align_patience', type=int, default=5)
    parser.add_argument('--align_min_delta', type=float, default=0.002)

    # Test related
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--save_model', type=int, default=1, choices=[0, 1])
    parser.add_argument('--load_model', type=int, default=1, choices=[0, 1])
    parser.add_argument('--multiworker', type=int, default=4,
                        help="Number of concurrent workers for inference.")
    args = parser.parse_args()

    print('= ' * 20, flush=True)
    print('## Starting Time:', get_cur_time(), flush=True)
    print(args, "\n")
    
    init_random_state(args.seed)
    LLM_REFINE_TEMPERATURE = float(args.llm_temperature)
    LLM_NAME = args.llm_name
    device = torch.device(args.device)
    
    split_info = json.load(open(f"./data/{args.dataset}/split_ids.json", 'r'))
    test_ids_set = set(split_info["test_ids"]["chain"])
    data_file = f"./data/{args.dataset}/data.json"
    train_ids = prepare_training_ids(
        args.dataset,
        train_num=args.train_num,
        alignment_ids=list(test_ids_set)
    )
    train_ids_set = set(train_ids)
    train_data = []
    with open(data_file, 'r') as f:
        for line in f:
            ex = json.loads(line)
            if ex["id"] in train_ids_set:
                train_data.append(ex)

    rng = np.random.RandomState(args.seed)
    indices = np.arange(len(train_data))
    rng.shuffle(indices)
    val_size = max(1, int(0.1 * len(train_data)))
    val_indices = set(indices[:val_size].tolist())
    train_data_raw = [ex for i, ex in enumerate(train_data) if i not in val_indices]
    val_data_raw = [ex for i, ex in enumerate(train_data) if i in val_indices]
    val_ids_set = set(ex["id"] for ex in val_data_raw)
    print(f"[Training Data] Train={len(train_data_raw)}, Val={len(val_data_raw)}")

    tool_meta = json.load(open(f"./data/{args.dataset}/tool_desc.json", 'r'))
    graph_meta = json.load(open(f"./data/{args.dataset}/graph_desc.json", 'r'))
    alias_map_graph = _build_tool_alias_map_from_nodes(graph_meta.get("nodes", []))  #这两个函数的作用是构建工具名称的规范化映射（别名映射），用于处理工具名称的大小写和格式不一致问题。
    all_tool_names = [node["id"] for node in tool_meta["nodes"]]
    allowed_tools_set = set(all_tool_names)
    print(f"[GCM] Tool list ({len(all_tool_names)})")

    from embedding_cache import init_embedding_cache
    print(f"[GCM] Initializing embedding cache...", flush=True)
    embedding_cache = init_embedding_cache(
        tool_meta, 
        cache_dir=f"./outputs/{args.dataset}/embedding_cache",
        device=args.device,
        lm_name=args.lm_name
    )#这个函数的作用是初始化并预计算工具描述的嵌入向量缓存，用于后续快速检索工具语义相似度。

    from utils_preproc import (
        build_io_types_vocab,
        build_typed_ngrams,
        build_confusion_prior,
        generate_perturbations_with_labels
    )
    
    cache_dir = f"./outputs/{args.dataset}/embedding_cache"
    print("Building global IO types vocabulary...")
    IO_TYPES, IO_TYPE2IDX = build_io_types_vocab(tool_meta)
    num_io_types = len(IO_TYPES)

    print("Building typed n-grams...")
    typed_ngrams = build_typed_ngrams(train_data_raw, tool_meta, n_range=(2, 4))
    ngrams_cache_file = f"outputs/{args.dataset}/typed_ngrams.json"
    os.makedirs(os.path.dirname(ngrams_cache_file), exist_ok=True)
    save_ngrams = {}
    if "__f2__" in typed_ngrams:
        save_ngrams["__f2__"] = {str(k): v for k, v in typed_ngrams["__f2__"].items()}
    if "__f3__" in typed_ngrams:
        save_ngrams["__f3__"] = {str(k): v for k, v in typed_ngrams["__f3__"].items()}
    if "__f4__" in typed_ngrams:
        save_ngrams["__f4__"] = {str(k): v for k, v in typed_ngrams["__f4__"].items()}
    if "__f_gt_motif__" in typed_ngrams:
        save_ngrams["__f_gt_motif__"] = {str(k): v for k, v in typed_ngrams["__f_gt_motif__"].items()}
    if "__tau_motif__" in typed_ngrams:
        save_ngrams["__tau_motif__"] = typed_ngrams["__tau_motif__"]
    with open(ngrams_cache_file, 'w') as f:
        json.dump(save_ngrams, f)
    print("Building confusion matrix...")
    lm_tag = args.lm_name.replace("/", "_")
    confusion_cache_file = f"outputs/{args.dataset}/confusion_topk10_{lm_tag}.json"
    confusion = build_confusion_prior(tool_meta, topk=10,
                                       cache_dir=cache_dir, device=args.device, lm_name=args.lm_name)
    os.makedirs(os.path.dirname(confusion_cache_file), exist_ok=True)
    with open(confusion_cache_file, "w") as f:
        json.dump(confusion, f, indent=2, ensure_ascii=False)


    print("[GCM] Initializing ModelTrainer...", flush=True)
    controller = ModelTrainer(
        args, device, tool_meta,
        confusion=confusion,
        typed_ngrams=typed_ngrams,
        IO_TYPE2IDX=IO_TYPE2IDX,
        num_io_types=num_io_types
    )
    controller.lambda_gap = float(args.lambda_gap)

    align_proj = None
    if args.align_pretrain_epochs > 0:
        align_ckpt = f"outputs/{args.dataset}/align_head.pt"
        controller.train_alignment_from_raw(train_data_raw, num_epochs=args.align_pretrain_epochs)
        controller.align_pretrained = True
        os.makedirs(os.path.dirname(align_ckpt), exist_ok=True)
        torch.save(
            {
                "step_proj": controller.model.step_proj.state_dict(),
                "tool_proj": controller.model.tool_proj.state_dict(),
                "align_dim": int(getattr(args, "align_dim", 1024)),
                "align_tau": float(getattr(args, "align_tau", 0.07))
            },
            align_ckpt
        )
        print(f"[Align] Saved alignment head to {align_ckpt}")
        ckpt = torch.load(align_ckpt, map_location="cpu")
        step_sd = ckpt.get("step_proj")
        tool_sd = ckpt.get("tool_proj")
        if step_sd and tool_sd:
            step_proj = torch.nn.Linear(step_sd["weight"].shape[1], step_sd["weight"].shape[0],
                                        bias="bias" in step_sd)
            tool_proj = torch.nn.Linear(tool_sd["weight"].shape[1], tool_sd["weight"].shape[0],
                                        bias="bias" in tool_sd)
            step_proj.load_state_dict(step_sd)
            tool_proj.load_state_dict(tool_sd)
            step_proj.eval()
            tool_proj.eval()
            align_proj = {"step_proj": step_proj, "tool_proj": tool_proj, "tau": args.align_tau}

    print("Generating training pairs (G_gt, G_cand)...")
    train_items = []
    perturbation_records = []
    perturbation_file = f"outputs/{args.dataset}/perturbation_graphs.json"
    os.makedirs(os.path.dirname(perturbation_file), exist_ok=True)
    for ex in train_data_raw:
        items = generate_perturbations_with_labels(
            ex, confusion, tool_meta, typed_ngrams,
            cache_dir=cache_dir, device=args.device, lm_name=args.lm_name,
            align_proj=align_proj, cost_tau=args.cost_tau
        )
        train_items.extend(items)
        for item in items:
            record = {
                "example_id": ex["id"],
                "user_request": ex.get("user_request", ""),
                "tools": item["tools"],
                "edges": item.get("edges", []),
                "y_cons": item["y_cons"],
                "cost": item.get("cost", 0.0),
                "node_risk": item["node_risk"],
                "gap_risk_edges": item.get("gap_risk_edges", []),
                "step_texts": item.get("step_texts", []),
                "is_gt": item.get("is_gt", False),
                "label_type": item.get("label_type", "unknown"),
                "perturb_ops": item.get("perturb_ops", [])
            }
            perturbation_records.append(record)
    with open(perturbation_file, 'w') as f:
        json.dump(perturbation_records, f, indent=2, ensure_ascii=False)

    val_items = []
    if val_data_raw:
        val_perturbation_file = f"outputs/{args.dataset}/perturbation_graphs_val.json"
        for ex in val_data_raw:
            items = generate_perturbations_with_labels(
                ex, confusion, tool_meta, typed_ngrams,
                cache_dir=cache_dir, device=args.device, lm_name=args.lm_name,
                align_proj=align_proj, cost_tau=args.cost_tau
            )
            val_items.extend(items)
        os.makedirs(os.path.dirname(val_perturbation_file), exist_ok=True)
        with open(val_perturbation_file, 'w') as f:
            json.dump(val_items, f, indent=2, ensure_ascii=False)
        print(f"[GCM] Generated {len(val_items)} validation items from {len(val_data_raw)} examples")

    save_path = (
        f"ckpts/{args.dataset}/"
        f"lgraph{args.lambda_graph}_lr{args.lr}_lgap{args.lambda_gap}_tau{args.cost_tau}.pt"
    )
    if args.load_model and os.path.exists(save_path):
        ckpt = torch.load(save_path, map_location=device)
        controller.model.load_state_dict(ckpt, strict=False)
    else:
        best_model, best_loss = controller.train(
            train_items,
            num_epochs=args.epoch,
            val_items=val_items,
            patience=args.patience,
            min_delta=0.002
        )
        controller.model = best_model

        if args.save_model:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(best_model.state_dict(), save_path)

    controller.model.eval()

    llm_short_names = {
        "gpt-3.5-turbo": "GPT-3.5",
        "gpt-4o": "GPT-4O",
        "Qwen/Qwen3-235B-A22B-Instruct-2507": "qwen3-235b-a22b"
    }
    base_llm = llm_short_names.get(LLM_NAME, LLM_NAME)
    val_ids_ordered = list(val_ids_set)
    val_direct_ids, val_direct_content = load_test_data(
        args.dataset, base_llm, val_ids_ordered, method='direct_val'
    )

    val_stats = []
    for data_id in val_direct_ids:
        pred_tools_raw = normalize_tools_list(val_direct_content[data_id]["pred_task_nodes"], alias_map_graph)
        gt_tools_raw = normalize_tools_list(val_direct_content[data_id]["gt_task_nodes"], alias_map_graph)
        user_request = val_direct_content[data_id].get("user_request", "")
        
        pred_links_raw = val_direct_content[data_id].get("pred_task_links", [])
        pred_step_texts = val_direct_content[data_id].get("steps", None) or val_direct_content[data_id].get("pred_task_steps", [])
        pred_tools, pred_step_texts, pred_edges_tool = order_chain_with_steps_and_edges(
            pred_tools_raw, pred_step_texts, pred_links_raw
        )
        pred_edges_gnn = add_start_edge(pred_edges_tool, len(pred_tools)) if pred_tools else []
        
        if pred_tools:
            pred_result = controller.score_chain(pred_tools, user_request, edges=pred_edges_gnn, step_texts=pred_step_texts)
            pred_S = float(pred_result.get("S", 0.0))
        else:
            pred_result = {"S": 0.0, "node_risks": [], "gap_risks": [], "gaps": []}
            pred_S = 0.0
        
        gt_links_raw = val_direct_content[data_id].get("gt_task_links", [])
        gt_step_texts = val_direct_content[data_id].get("gt_task_steps", [])
        gt_tools, gt_step_texts, gt_edges_tool = order_chain_with_steps_and_edges(
            gt_tools_raw, gt_step_texts, gt_links_raw
        )
        gt_links = edges_to_links(gt_tools, gt_edges_tool)
        pred_links = edges_to_links(pred_tools, pred_edges_tool)
        
        node_f1 = compute_f1_set_based(pred_tools, gt_tools)
        link_f1 = compute_f1_set_based(pred_links, gt_links)
        
        val_stats.append({
            "id": data_id,
            "S": pred_S,
            "node_f1": node_f1,
            "link_f1": link_f1
        })
    
    risk_thresh_path = f"outputs/{args.dataset}/risk_thresholds.json"
    t_accept_for_risk = 0.8
    theta_node, theta_gap, _ = search_risk_thresholds(
        val_direct_ids, val_direct_content, controller, t_accept_for_risk,
        alias_map_graph
    )
    THETA_NODE, THETA_GAP = theta_node, theta_gap
    os.makedirs(os.path.dirname(risk_thresh_path), exist_ok=True)
    with open(risk_thresh_path, "w") as f:
        json.dump({"theta_node": theta_node, "theta_gap": theta_gap, "t_accept": t_accept_for_risk}, f, indent=2, ensure_ascii=False)

    base_t, candidate_thresholds = stage1_candidate_thresholds(val_stats)
    print(f"[GCM] Grid search T_accept")
    T_accept = stage2_search_thresholds_with_llm(
        val_direct_ids, val_direct_content, controller, tool_meta, confusion,
        typed_ngrams, allowed_tools_set, candidate_thresholds,
        max_samples=None,
        alias_map=alias_map_graph,
        base_t=base_t
    )
    # Inference
    print("[GCM] Starting inference on test set...")
    table = pt.PrettyTable()
    table.field_names = ["Dataset", "Method", "N-F1", "L-F1", "Accuracy"]
    
    final_pred_dict = {}

    alignment_ids = list(test_ids_set)
    direct_ids, direct_content = load_test_data(args.dataset, base_llm, alignment_ids, method='direct')
    test_ids = direct_ids

    all_requests = set()
    for data_id in test_ids:
        if data_id in direct_content:
            req = direct_content[data_id].get("user_request", "")
            if req:
                all_requests.add(req)
    embedding_cache.precompute_requests(list(all_requests))
    
    # Phase 1: Score all direct predictions with GNN 
    print(f"\n[Phase 1] Scoring {len(test_ids)} direct predictions with GNN...")
    
    def score_one_sample(data_id):
        pred_tools_raw_eval = direct_content[data_id]["pred_task_nodes"]
        pred_links_raw_eval = direct_content[data_id].get("pred_task_links", [])
        gt_tools_raw_eval = direct_content[data_id]["gt_task_nodes"]
        gt_links_raw_eval = direct_content[data_id].get("gt_task_links", [])

        pred_tools_raw = normalize_tools_list(pred_tools_raw_eval, alias_map_graph)
        gt_tools_raw = normalize_tools_list(gt_tools_raw_eval, alias_map_graph)
        user_request = direct_content[data_id].get("user_request", "")

        gt_links_raw = gt_links_raw_eval
        gt_step_texts = direct_content[data_id].get("gt_task_steps", [])
        gt_tools, gt_step_texts, gt_edges_tool = order_chain_with_steps_and_edges(
            gt_tools_raw, gt_step_texts, gt_links_raw
        )
        gt_links = edges_to_links(gt_tools, gt_edges_tool)

        pred_links_raw = pred_links_raw_eval
        pred_step_texts = direct_content[data_id].get("steps", None) or direct_content[data_id].get("pred_task_steps", [])
        
        if not pred_tools_raw:
            pred_result = {"S": 0.0, "node_risks": [], "gap_risks": [], "gaps": []}
            pred_edges_tool = []
            pred_links = []
            pred_tools = []
            pred_step_texts = []
        else:
            pred_tools, pred_step_texts, pred_edges_tool = order_chain_with_steps_and_edges(
                pred_tools_raw, pred_step_texts, pred_links_raw
            )
            pred_edges_gnn = add_start_edge(pred_edges_tool, len(pred_tools)) if pred_tools else []
            pred_links = edges_to_links(pred_tools, pred_edges_tool)
            pred_result = controller.score_chain(pred_tools, user_request, edges=pred_edges_gnn, step_texts=pred_step_texts)
        
        return data_id, {
            "example_id": data_id,
            "user_request": user_request,
            "gt": {
                "nodes": gt_tools,
                "links": gt_links,
                "nodes_raw": gt_tools_raw_eval,
                "links_raw": gt_links_raw_eval
            },
            "direct": {
                "nodes": pred_tools,
                "links": pred_links,
                "edges": pred_edges_tool,
                "steps": pred_step_texts if isinstance(pred_step_texts, list) else [],
                "nodes_raw": pred_tools_raw_eval,
                "links_raw": pred_links_raw_eval,
                "S": float(pred_result.get("S", 0.0)),
                "node_risks": pred_result.get("node_risks", []),
                "gap_risks": pred_result.get("gap_risks", []),
                "gaps": pred_result.get("gaps", [])
            }
        }
    
    sem = asyncio.Semaphore(args.multiworker)

    async def score_one_async(data_id):
        async with sem:
            loop = asyncio.get_event_loop()
            data_id_result, result = await loop.run_in_executor(None, score_one_sample, data_id)
            final_pred_dict[data_id_result] = result

    loop = asyncio.get_event_loop()
    tasks = [score_one_async(data_id) for data_id in test_ids]
    loop.run_until_complete(asyncio.gather(*tasks))
    
    # Phase 2: Refine predictions with LLM based on GNN score
    print(f"\n[Phase 2] Refining predictions with LLM...")

    def refine_one_sample(data_id):
        user_request = final_pred_dict[data_id]["user_request"]
        direct_data = final_pred_dict[data_id]["direct"]

        init_plan = {
            "nodes": direct_data["nodes"],
            "edges": direct_data.get("edges", []),
            "steps": direct_data.get("steps", [])
        }

        if len(init_plan["steps"]) != len(init_plan["nodes"]):
            tool_desc_map = {n["id"]: n.get("desc", n.get("description", "")) for n in tool_meta["nodes"]}
            fixed_steps = []
            for i, node_id in enumerate(init_plan["nodes"]):
                if i < len(init_plan["steps"]) and init_plan["steps"][i]:
                    fixed_steps.append(init_plan["steps"][i])
                else:
                    desc = tool_desc_map.get(node_id, "")
                    fixed_steps.append(f"Step {i+1}: {desc[:100]}" if desc else f"Step {i+1}: Call {node_id}")
            init_plan["steps"] = fixed_steps

        if init_plan["nodes"]:
            init_plan["edges"] = [(i, i + 1) for i in range(len(init_plan["nodes"]) - 1)]
        else:
            init_plan["edges"] = []
        
        init_gnn = {
            "S": direct_data["S"],
            "node_risks": direct_data["node_risks"],
            "gap_risks": direct_data["gap_risks"],
            "gaps": direct_data["gaps"]
        }

        gt_nodes = final_pred_dict[data_id]["gt"]["nodes"]
        gt_links = final_pred_dict[data_id]["gt"]["links"]
        refined_plan, final_strategy, _ = iterative_refine_with_llm(
            controller=controller,
            user_request=user_request,
            init_plan=init_plan,
            init_gnn=init_gnn,
            tool_meta=tool_meta,
            confusion=confusion,
            typed_ngrams=typed_ngrams,
            allowed_tools=allowed_tools_set,
            threshold_accept=T_accept,
            llm_temperature=LLM_REFINE_TEMPERATURE,
            llm_cache=None,
            data_id=data_id,
            gt_nodes=gt_nodes,
            gt_links=gt_links,
            alias_map=alias_map_graph
        )
        
        refined_nodes = refined_plan.get("nodes", [])
        refined_edges_tool = refined_plan.get("edges", [])
        refined_steps = refined_plan.get("steps", [])
        refined_nodes, refined_steps, refined_edges_tool = order_chain_with_steps_and_edges(
            refined_nodes, refined_steps, refined_edges_tool
        )
        refined_links = edges_to_links(refined_nodes, refined_edges_tool)
        
        if not refined_nodes:
            refined_result = {"S": 0.0, "node_risks": [], "gap_risks": [], "gaps": []}
        else:
            refined_edges_gnn = add_start_edge(refined_edges_tool, len(refined_nodes))
            refined_result = controller.score_chain(refined_nodes, user_request, 
                                                     edges=refined_edges_gnn, 
                                                     step_texts=refined_steps)
        
        return data_id, final_strategy, {
            "nodes": refined_nodes,
            "links": refined_links
        }

    sem = asyncio.Semaphore(args.multiworker)

    async def refine_one_async(data_id):
        async with sem:
            loop = asyncio.get_event_loop()
            data_id_result, _, refined_data = await loop.run_in_executor(None, refine_one_sample, data_id)
            final_pred_dict[data_id_result]["refined"] = refined_data

    loop = asyncio.get_event_loop()
    tasks = [refine_one_async(data_id) for data_id in test_ids]
    loop.run_until_complete(asyncio.gather(*tasks))

    # Phase 3: Evaluate results
    print(f"\n[Phase 3] Evaluating results...")
    for method in ["direct", "refined"]:
        node_f1_list, link_f1_list, acc_list = [], [], []
        
        for data_id in test_ids:
            if data_id not in final_pred_dict or method not in final_pred_dict[data_id]:
                continue
            
            content = final_pred_dict[data_id]
            if method == "direct":
                pred_node = content["direct"].get("nodes_raw", content["direct"]["nodes"])
                pred_link = content["direct"].get("links_raw", content["direct"]["links"])
                gt_node = content["gt"].get("nodes_raw", content["gt"]["nodes"])
                gt_link = content["gt"].get("links_raw", content["gt"]["links"])
            else:
                pred_node = content[method]["nodes"]
                pred_link = content[method]["links"]
                gt_node = content["gt"]["nodes"]
                gt_link = content["gt"]["links"]
            
            node_f1 = compute_f1_set_based(pred_node, gt_node)
            link_f1 = compute_f1_set_based(pred_link, gt_link)
            acc = float(node_f1 > 0.99 and link_f1 > 0.99)
            
            node_f1_list.append(node_f1)
            link_f1_list.append(link_f1)
            acc_list.append(acc)
        
        avg_node_f1 = sum(node_f1_list) / len(node_f1_list) if node_f1_list else 0.0
        avg_link_f1 = sum(link_f1_list) / len(link_f1_list) if link_f1_list else 0.0
        avg_acc = sum(acc_list) / len(acc_list) if acc_list else 0.0

        method_name = "Direct" if method == "direct" else "Direct+LLM_Refined"
        
        table.add_row([args.dataset, method_name, f"{avg_node_f1:.4f}", f"{avg_link_f1:.4f}", f"{avg_acc:.2f}"])
    
    print(table)

    print('\n## Finishing Time:', get_cur_time(), flush=True)
    print('= ' * 20)

    # 在 print(table) 之前添加：
    import json

    output_result_file = f"./prediction/{args.dataset}/{base_llm}/gcm_refined.json"
    os.makedirs(os.path.dirname(output_result_file), exist_ok=True)

    # 新增：准备正确和错误预测的文件路径
    correct_pred_file = f"./prediction/{args.dataset}/{base_llm}/correct_predictions.json"
    incorrect_pred_file = f"./prediction/{args.dataset}/{base_llm}/incorrect_predictions.json"

    correct_predictions = []
    incorrect_predictions = []

    with open(output_result_file, 'w') as f:
        for data_id in test_ids:
            if data_id in final_pred_dict:
                content = final_pred_dict[data_id]

                # 获取预测结果和真实结果
                pred_nodes = content["refined"]["nodes"]
                pred_links = content["refined"]["links"]
                gt_nodes = content["gt"]["nodes"]
                gt_links = content["gt"]["links"]

                # 使用与源代码完全相同的计算方法
                node_f1 = compute_f1_set_based(pred_nodes, gt_nodes)
                link_f1 = compute_f1_set_based(pred_links, gt_links)
                # 与源代码一致：acc = float(node_f1 > 0.99 and link_f1 > 0.99)
                is_correct = float(node_f1 > 0.99 and link_f1 > 0.99)

                record = {
                    "id": data_id,
                    "user_request": content["user_request"],
                    "pred_task_nodes": pred_nodes,
                    "pred_task_links": pred_links,
                    "direct_S": content["direct"]["S"],
                    "strategy": content.get("strategy", "unknown")
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                # 根据预测是否正确，分别保存到不同的列表
                if is_correct:
                    correct_record = {
                        "id": data_id,
                        "user_request": content["user_request"],
                        "pred_task_nodes": pred_nodes,
                        "pred_task_links": pred_links,
                        "gt_task_nodes": gt_nodes,
                        "gt_task_links": gt_links,
                        "node_f1": node_f1,
                        "link_f1": link_f1,
                        "direct_S": content["direct"]["S"],
                        "strategy": content.get("strategy", "unknown")
                    }
                    correct_predictions.append(correct_record)
                else:
                    incorrect_record = {
                        "id": data_id,
                        "user_request": content["user_request"],
                        "pred_task_nodes": pred_nodes,
                        "pred_task_links": pred_links,
                        "gt_task_nodes": gt_nodes,
                        "gt_task_links": gt_links,
                        "node_f1": node_f1,
                        "link_f1": link_f1,
                        "direct_S": content["direct"]["S"],
                        "strategy": content.get("strategy", "unknown"),
                        "error_analysis": {
                            "missing_nodes": list(set(gt_nodes) - set(pred_nodes)),
                            "extra_nodes": list(set(pred_nodes) - set(gt_nodes)),
                            "missing_links": list(set(gt_links) - set(pred_links)),
                            "extra_links": list(set(pred_links) - set(gt_links))
                        }
                    }
                    incorrect_predictions.append(incorrect_record)

    # 保存正确预测的文件 - 使用与源代码相同的准确率计算
    total_samples = len(correct_predictions) + len(incorrect_predictions)
    avg_acc = sum([1.0 for _ in correct_predictions]) / total_samples if total_samples > 0 else 0.0

    with open(correct_pred_file, 'w', encoding='utf-8') as f:
        json.dump({
            "total_correct": len(correct_predictions),
            "total_samples": total_samples,
            "accuracy": avg_acc,
            "predictions": correct_predictions
        }, f, ensure_ascii=False, indent=2)

    # 保存错误预测的文件
    with open(incorrect_pred_file, 'w', encoding='utf-8') as f:
        json.dump({
            "total_incorrect": len(incorrect_predictions),
            "total_samples": total_samples,
            "error_rate": len(incorrect_predictions) / total_samples if total_samples > 0 else 0,
            "predictions": incorrect_predictions
        }, f, ensure_ascii=False, indent=2)

    print(f"Saved refined results to: {output_result_file}")
    print(f"Saved correct predictions ({len(correct_predictions)}) to: {correct_pred_file}")
    print(f"Saved incorrect predictions ({len(incorrect_predictions)}) to: {incorrect_pred_file}")

    print("Done!")

    print("Done!")
