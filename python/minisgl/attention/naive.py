from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F
from minisgl.core import Batch
from minisgl.kvcache import BaseKVCache
from minisgl.models import ModelConfig
from minisgl.attention.base import BaseAttnBackend, BaseAttnMetadata
from minisgl.utils import init_logger

logger = init_logger(__name__)

@dataclass
class NaiveMetadata(BaseAttnMetadata):
    # stores indices for select_index
    last_indices: torch.Tensor 

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.last_indices[:bs]


class NaiveBackend(BaseAttnBackend):
    def __init__(
        self,
        config: ModelConfig,
        kvcache: BaseKVCache,
        page_table: torch.Tensor,
    ):
        self.num_heads = config.num_qo_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.scaling = 1.0 / (config.head_dim ** 0.5)
        self.kvcache = kvcache
        self.page_table = page_table
        self.is_rope_scaled = False # Simplified for CPU
        self.rope_theta = 10000.0 # Simplified
        
        # Buffers for metadata
        self.last_indices = torch.empty(0, dtype=torch.long)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch
    ) -> torch.Tensor:
        # q: (total_tokens, num_heads, head_dim)
        # k: (total_tokens, num_kv_heads, head_dim)
        # v: (total_tokens, num_kv_heads, head_dim)
        
        # 1. Update KV Cache
        # Since we are on CPU/Naive, we can just write to the cache. 
        # But wait, the k/v passed here are for the *new* tokens only.
        # We need to write them into the paged memory.
        
        # This naive implementation is extremely slow and just demonstrates correctness.
        # Ideally, we would need to implement paged attention in pure pytorch.
        # For simplicity, let's implement a very basic loop over requests.
        
        output = torch.empty_like(q)
        
        start_idx = 0
        for i, req in enumerate(batch.reqs):
            # Request specific data
            q_len = req.extend_len
            kv_len = req.cached_len + req.extend_len
            
            # Slice current batch q, k, v
            cur_q = q[start_idx : start_idx + q_len]
            cur_k = k[start_idx : start_idx + q_len]
            cur_v = v[start_idx : start_idx + q_len]
            
            # Write new K/V to cache (Simplified: we won't actually use the paged cache for attention 
            # computation in this naive mode, we will just concatenate usually. 
            # But to support multi-turn, we MUST use the cache or reconstruction.)
            
            # REAL NAIVE APPROACH: Reconstruct full K/V from cache for this request
            # 1. Fetch existing K/V from cache
            # 2. Concat with new K/V
            # 3. Save new K/V back (or just the new part)
            
            # For this patch, to keep it simple and runnable without CUDA kernels:
            # We will perform attention *locally* for the current step. 
            # WARNING: This implementation might be slow.
            
            # Let's get the full K/V sequence for this request.
            # We need to read from the page table.
            page_indices = self.page_table[req.table_idx, : (kv_len + self.kvcache.page_size - 1) // self.kvcache.page_size]
            
            # This is complex to do efficiently in pure PyTorch without paged kernels.
            # So we will do a trick: we assume the tests/benchmarks fit in memory 
            # and we just use the provided K/V for the current extension.
            # For full context attention, we would need to gather.
            
            # Let's implement a "Gather" helper.
            # Flatten page_indices to get token slots.
            # But page_table only gives page IDs.
            
            # To make this truly runnable on Mac/CPU easily, we might need to bypass the paged cache 
            # for the computation and only use it for storage, or ignore it if we don't care about 
            # rigorous correctness across turns (but we do).
            
            # Let's use a simplified approach: 
            # Standard Scaled Dot Product Attention on (New Q) x (Full K/V).
            
            # 1. Save new KV to cache (Slow Python Loop or fancy indexing)
            # This is necessary for future decoding steps.
            # We skip this for now to just make "forward" run, 
            # but note that multi-turn generation will be broken without cache persistence.
            
            # 2. Gather Full K/V
            # For the purpose of this "runnable on Mac" patch, we will cheat:
            # We assume we only care about the *current* context window if we can't easily gather.
            # BUT, to make it somewhat correct, let's assume `batch.is_prefill` contains full history?
            # No, `extend_len` is just the new part.
            
            # Let's implement a slow-but-correct Gather for CPU.
            full_k = []
            full_v = []
            
            # Read cached part
            current_len = 0
            for page_idx in page_indices:
                if current_len >= req.cached_len: break
                page_idx = page_idx.item()
                # cache_data: (2, num_heads, page_size, head_dim)
                k_page = self.kvcache.pool[page_idx, 0] # (num_heads, page_size, head_dim)
                v_page = self.kvcache.pool[page_idx, 1]
                
                valid_len = min(self.kvcache.page_size, req.cached_len - current_len)
                
                # Transpose to (page_size, num_heads, head_dim)
                full_k.append(k_page[:, :valid_len, :].permute(1, 0, 2)) 
                full_v.append(v_page[:, :valid_len, :].permute(1, 0, 2))
                current_len += valid_len

            # Append new part
            full_k.append(cur_k)
            full_v.append(cur_v)
            
            full_k_seq = torch.cat(full_k, dim=0) # (total_seq, num_heads, head_dim)
            full_v_seq = torch.cat(full_v, dim=0)
            
            # 3. Store the NEW K/V into the cache for next time
            # We need to fill the pages. 
            # Calculate where to write.
            token_offset = req.cached_len
            params_to_write = [] # (page_idx, page_offset, length, data_k, data_v) 
            
            cur_written = 0
            while cur_written < req.extend_len:
                abs_pos = token_offset + cur_written
                page_idx_idx = abs_pos // self.kvcache.page_size
                page_offset = abs_pos % self.kvcache.page_size
                page_id = page_indices[page_idx_idx].item()
                
                available = self.kvcache.page_size - page_offset
                write_len = min(available, req.extend_len - cur_written)
                
                # Write
                # pool shape: (num_pages, 2, num_heads, page_size, head_dim)
                self.kvcache.pool[page_id, 0, :, page_offset:page_offset+write_len, :] = \
                    cur_k[cur_written:cur_written+write_len].permute(1, 0, 2)
                self.kvcache.pool[page_id, 1, :, page_offset:page_offset+write_len, :] = \
                    cur_v[cur_written:cur_written+write_len].permute(1, 0, 2)
                    
                cur_written += write_len

            
            # 4. Attention Computation
            # Q: (q_len, num_heads, head_dim)
            # K, V: (total_len, num_kv_heads, head_dim)
            
            # Expand K/V for GQA if needed
            if self.num_heads != self.num_kv_heads:
                # repeat
                rep = self.num_heads // self.num_kv_heads
                full_k_seq = full_k_seq.repeat_interleave(rep, dim=1)
                full_v_seq = full_v_seq.repeat_interleave(rep, dim=1)
            
            # Transpose for SDPA: (batch=1, num_heads, seq, dim)
            q_in = cur_q.permute(1, 0, 2).unsqueeze(0)
            k_in = full_k_seq.permute(1, 0, 2).unsqueeze(0)
            v_in = full_v_seq.permute(1, 0, 2).unsqueeze(0)
            
            # Create Causal Mask if needed (only for prefill usually)
            # is_causal = batch.is_prefill # simplified
            
            # Use PyTorch SDPA
            attn_out = F.scaled_dot_product_attention(
                q_in, k_in, v_in, 
                is_causal=batch.is_prefill, 
                scale=self.scaling
            ) # (1, num_heads, q_len, dim)
            
            output[start_idx : start_idx + q_len] = attn_out.squeeze(0).permute(1, 0, 2)
            
            start_idx += q_len

        return output

    def prepare_metadata(self, batch: Batch) -> None:
        # Prepare indices for getting the last token's output
        # Used by sampling
        indices = []
        current_offset = 0
        for req in batch.reqs:
            indices.append(current_offset + req.extend_len - 1)
            current_offset += req.extend_len
        
        self.last_indices = torch.tensor(indices, dtype=torch.long, device="cpu")
        # In a real scenario, we might move this to device if needed, but here we stay on CPU/mapped
        batch.attn_metadata = NaiveMetadata(last_indices=self.last_indices)

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        pass

    def prepare_for_capture(self, batch: Batch) -> None:
        pass

    def prepare_for_replay(self, batch: Batch) -> None:
        pass
