import torch
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

_CACHES = {}


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    return " ".join(str(text).strip().split())


class EmbeddingCache:
    def __init__(self, cache_dir: str = "../data/embedding_cache", device: str = "cuda:0",
                 lm_name: str = "intfloat/e5-large"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.lm_name = lm_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._model = None
        self._tokenizer = None
        self.lm_dim = None 
        self._text_cache: Dict[str, np.ndarray] = {}
        self._tool_embeddings: Dict[str, np.ndarray] = {}
        
        self._load_cache()
    
    def _load_model(self):
        if self._model is None:
            from transformers import AutoTokenizer, AutoModel
            print("self.lm_name: ", self.lm_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self.lm_name)
            self._model = AutoModel.from_pretrained(self.lm_name)
            self._model = self._model.to(self.device)
            self._model.eval()
            self.lm_dim = int(self._model.config.hidden_size)

    def ensure_model_loaded(self):
        self._load_model()
    
    def _load_cache(self):
        cache_file = self.cache_dir / "text_embeddings.npz"
        tool_cache_file = self.cache_dir / "tool_embeddings.npz"
        
        if cache_file.exists():
            try:
                data = np.load(cache_file, allow_pickle=True)
                self._text_cache = dict(data['cache'].item())
            except Exception:
                pass

        if tool_cache_file.exists():
            try:
                data = np.load(tool_cache_file, allow_pickle=True)
                self._tool_embeddings = dict(data['embeddings'].item())
            except Exception:
                pass
    
    def save_cache(self):
        cache_file = self.cache_dir / "text_embeddings.npz"
        tool_cache_file = self.cache_dir / "tool_embeddings.npz"
        
        try:
            np.savez(cache_file, cache=self._text_cache)
        except Exception:
            pass

        if self._tool_embeddings:
            try:
                np.savez(tool_cache_file, embeddings=self._tool_embeddings)
            except Exception:
                pass
    
    def encode_texts(self, texts: List[str], prefix: str = "passage") -> np.ndarray:
        if not texts:
            dim = int(self.lm_dim or 0)
            return np.zeros((0, dim), dtype=np.float32)
        
        self._load_model()
        
        embeddings = []
        new_texts = []
        new_indices = []

        for i, text in enumerate(texts):
            norm_text = normalize_text(text)
            cache_key = f"{prefix}:{norm_text}"
            if cache_key in self._text_cache:
                embeddings.append(self._text_cache[cache_key])
            else:
                embeddings.append(None)
                new_texts.append(norm_text)
                new_indices.append(i)

        if new_texts:
            batch_size = 32 
            all_new_embs = []
            
            for batch_start in range(0, len(new_texts), batch_size):
                batch_texts_raw = new_texts[batch_start:batch_start + batch_size]
                batch_texts = [f"{prefix}: {t}" for t in batch_texts_raw]
                
                encoded = self._tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    output = self._model(**encoded)
    
                    attention_mask = encoded['attention_mask']  
                    last_hidden = output.last_hidden_state 
                    
                    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                    sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
                    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                    batch_embs = sum_embeddings / sum_mask

                    batch_embs = torch.nn.functional.normalize(batch_embs, p=2, dim=-1)
                    batch_embs = batch_embs.cpu().numpy()
                
                all_new_embs.append(batch_embs)
            
            new_embs = np.concatenate(all_new_embs, axis=0)

            for idx, emb in zip(new_indices, new_embs):
                cache_key = f"{prefix}:{normalize_text(texts[idx])}"
                self._text_cache[cache_key] = emb
                embeddings[idx] = emb
        
        return np.array(embeddings, dtype=np.float32)
    
    def encode_texts_tensor(self, texts: List[str], device: Optional[torch.device] = None, prefix: str = "passage") -> torch.Tensor:
        if device is None:
            device = self.device
        
        embs_np = self.encode_texts(texts, prefix=prefix)
        return torch.from_numpy(embs_np).to(device)
    
    def precompute_tool_embeddings(self, tool_meta: Dict, force: bool = False):
        nodes = tool_meta.get("nodes", [])
        if not nodes:
            return

        ordered_tool_ids = [n["id"] for n in nodes]
        existing_ids = set(self._tool_embeddings.keys())
        new_ids_set = set(ordered_tool_ids)
        
        if not force and new_ids_set.issubset(existing_ids):
            return
        
        tool_ids = []
        tool_descs = []
        for n in nodes:
            tid = n["id"]
            desc = n.get("desc", tid)
            tool_ids.append(tid)
            tool_descs.append(desc)

        embeddings = self.encode_texts(tool_descs, prefix="passage")

        for i, tid in enumerate(tool_ids):
            self._tool_embeddings[tid] = embeddings[i]
        
        self.save_cache()

    def get_all_tool_embeddings(self) -> Dict[str, np.ndarray]:
        return self._tool_embeddings.copy()

    def precompute_requests(self, requests: List[str]):
        if not requests:
            return

        new_requests = [r for r in requests if r and f"query:{normalize_text(r)}" not in self._text_cache]
        if new_requests:
            self.encode_texts(new_requests, prefix="query")
            self.save_cache()


def get_embedding_cache(cache_dir: str = "./outputs/embedding_cache", device: str = "cuda:0",
                        lm_name: str = "intfloat/e5-large") -> EmbeddingCache:
    key = (str(cache_dir), str(device), str(lm_name))
    if key not in _CACHES:
        _CACHES[key] = EmbeddingCache(cache_dir=cache_dir, device=device, lm_name=lm_name)
    return _CACHES[key]


def init_embedding_cache(tool_meta: Dict, cache_dir: str = "./outputs/embedding_cache", device: str = "cuda:0",
                         lm_name: str = "intfloat/e5-large"):
    cache = get_embedding_cache(cache_dir=cache_dir, device=device, lm_name=lm_name)
    cache.precompute_tool_embeddings(tool_meta)
    return cache
