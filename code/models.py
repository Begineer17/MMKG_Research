from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Optional

import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from scipy.sparse import csgraph
from scipy.optimize import linear_sum_assignment
import pickle
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelArgs:
    dim: int 
    n_layers: int = 8
    n_heads: int = 5
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 2048
    head_dim: int = -1
    q_rank: int = 12
    rank: int = 2
    using_groupnorm: bool = False

class T6GroupNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True, memory_efficient=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class TPA(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads

        self.n_heads = args.n_heads
        self.head_dim = args.head_dim if args.head_dim > 0 else args.dim // args.n_heads
        # maybe different from args.dim // args.n_heads

        self.n_head = args.n_heads
        self.q_rank = args.q_rank
        self.rank = args.rank
        self.dim = args.dim
        
        self.using_groupnorm = args.using_groupnorm
        
        self.W_A_q = nn.Linear(args.dim, self.n_head * self.q_rank, bias=False)
        self.W_A_k = nn.Linear(args.dim, self.n_head * self.rank, bias=False)
        self.W_A_v = nn.Linear(args.dim, self.n_head * self.rank, bias=False)

        # Define B projection parameters for Q, K, V
        self.W_B_q = nn.Linear(args.dim, self.q_rank * self.head_dim, bias=False)
        self.W_B_k = nn.Linear(args.dim, self.rank * self.head_dim, bias=False)
        self.W_B_v = nn.Linear(args.dim, self.rank * self.head_dim, bias=False)
        
        self.cache_kA = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_heads, self.rank,)).cuda()
        self.cache_vA = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_heads, self.rank,)).cuda()
        self.cache_kB = torch.zeros((args.max_batch_size, args.max_seq_len, self.rank, self.head_dim,)).cuda()
        self.cache_vB = torch.zeros((args.max_batch_size, args.max_seq_len, self.rank, self.head_dim,)).cuda()
        
        self.reset_parameters()

        if self.using_groupnorm:
            self.subln = T6GroupNorm(self.head_dim, eps=1e-5, elementwise_affine=True)
    def reset_parameters(self, args):
        W_A_q_tensor = self.W_A_q.weight.view(self.dim, self.n_head, self.q_rank)
        W_A_k_tensor = self.W_A_k.weight.view(self.dim, self.n_head, self.rank)
        W_A_v_tensor = self.W_A_v.weight.view(self.dim, self.n_head, self.rank)
        nn.init.xavier_uniform_(W_A_q_tensor)
        nn.init.xavier_uniform_(W_A_k_tensor)
        nn.init.xavier_uniform_(W_A_v_tensor)
        self.W_A_q.weight.data = W_A_q_tensor.view_as(self.W_A_q.weight)
        self.W_A_k.weight.data = W_A_k_tensor.view_as(self.W_A_k.weight)
        self.W_A_v.weight.data = W_A_v_tensor.view_as(self.W_A_v.weight)

        W_B_q_tensor = self.W_B_q.weight.view(self.dim, self.q_rank, self.head_dim)
        W_B_k_tensor = self.W_B_k.weight.view(self.dim, self.rank, self.head_dim)
        W_B_v_tensor = self.W_B_v.weight.view(self.dim, self.rank, self.head_dim)
        nn.init.xavier_uniform_(W_B_q_tensor)
        nn.init.xavier_uniform_(W_B_k_tensor)
        nn.init.xavier_uniform_(W_B_v_tensor)
        self.W_B_q.weight.data = W_B_q_tensor.view_as(self.W_B_q.weight)
        self.W_B_k.weight.data = W_B_k_tensor.view_as(self.W_B_k.weight)
        self.W_B_v.weight.data = W_B_v_tensor.view_as(self.W_B_v.weight)
        
    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = Q.shape

        A_q = self.W_A_q(Q).view(bsz, seqlen, self.n_head, self.q_rank)
        A_k = self.W_A_k(K).view(bsz, seqlen, self.n_head, self.rank)
        A_v = self.W_A_v(V).view(bsz, seqlen, self.n_head, self.rank)

        # Compute intermediate variables B for Q, K, and V
        B_q = self.W_B_q(Q).view(bsz, seqlen, self.q_rank, self.head_dim)
        B_k = self.W_B_k(K).view(bsz, seqlen, self.rank, self.head_dim)
        B_v = self.W_B_v(V).view(bsz, seqlen, self.rank, self.head_dim)

        B_q, B_k = apply_rotary_emb(B_q, B_k, freqs_cis=freqs_cis)
        
        # Cache A_k, A_v
        self.cache_kA = self.cache_kA.to(A_k)
        self.cache_vA = self.cache_vA.to(A_v)
        
        self.cache_kA[:bsz, start_pos : start_pos + seqlen] = A_k
        self.cache_vA[:bsz, start_pos : start_pos + seqlen] = A_v
        
        A_k = self.cache_kA[:bsz, : start_pos + seqlen]
        A_v = self.cache_vA[:bsz, : start_pos + seqlen]
        
        # Cache B_k, B_v
        
        self.cache_kB = self.cache_kB.to(B_k)
        self.cache_vB = self.cache_vB.to(B_v)
        
        self.cache_kB[:bsz, start_pos : start_pos + seqlen] = B_k
        self.cache_vB[:bsz, start_pos : start_pos + seqlen] = B_v
        
        B_k = self.cache_kB[:bsz, : start_pos + seqlen]
        B_v = self.cache_vB[:bsz, : start_pos + seqlen]
        
        # Reshape A_q, A_k, A_v
        A_q = A_q.view(bsz * seqlen, self.n_head, self.q_rank)
        A_k = A_k.view(bsz * seqlen, self.n_head, self.rank)
        A_v = A_v.view(bsz * seqlen, self.n_head, self.rank)

        # Reshape B_k, B_v  
        B_q = B_q.view(bsz * seqlen, self.q_rank, self.head_dim)
        B_k = B_k.view(bsz * seqlen, self.rank, self.head_dim)
        B_v = B_v.view(bsz * seqlen, self.rank, self.head_dim)
        
        q = torch.bmm(A_q, B_q).div_(self.q_rank).view(bsz, seqlen, self.n_head, self.head_dim)
        k = torch.bmm(A_k, B_k).div_(self.rank).view(bsz, seqlen, self.n_head, self.head_dim)
        v = torch.bmm(A_v, B_v).div_(self.rank).view(bsz, seqlen, self.n_head, self.head_dim)
        
        k = k.transpose(1, 2) 
        scores = torch.matmul(q.transpose(1, 2), k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  
        scores = F.softmax(scores.float(), dim=-1).type_as(q)
        output = torch.matmul(scores, v.transpose(1, 2))  
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

class MEncoderLayer(nn.Module):
    """
    A single M-Encoder layer that:
      1) Does Prefix-Guided Interaction in the multi-head attention.
      2) Uses the modified FFN(xt) = ReLU( xt W1 + Agg(xv) W3 + b1 ) W2 + b2
         for Correlation-Aware Fusion.
    """
    def __init__(self, args):
        super(MEncoderLayer, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.config = args
        
        # # ---- Multi-head attention for text side ----
        # self.text_q_proj = nn.Linear(config['hidden_dim'], config['hidden_dim']).to(device)
        # self.text_k_proj = nn.Linear(config['hidden_dim'], config['hidden_dim']).to(device)
        # self.text_v_proj = nn.Linear(config['hidden_dim'], config['hidden_dim']).to(device)

        # # ---- Multi-head attention for visual side ----
        # self.visl_q_proj = nn.Linear(config['hidden_dim'], config['hidden_dim']).to(device)
        # self.visl_k_proj = nn.Linear(config['hidden_dim'], config['hidden_dim']).to(device)
        # self.visl_v_proj = nn.Linear(config['hidden_dim'], config['hidden_dim']).to(device)

        # ---- Feed-Forward weights for text side, using the "modified" formula
        # W1: (hidden_dim, 4*hidden_dim),  W2: (4*hidden_dim, hidden_dim)
        self.W1 = nn.Linear(args.dim, 4 * args.dim, bias=False).to(device)   # for xt
        self.W3 = nn.Linear(args.dim, 4 * args.dim, bias=False).to(device)   # for Agg(xv)
        self.b1 = nn.Parameter(torch.zeros(4 * args.dim)).to(device)
        self.W2 = nn.Linear(4 * args.dim, args.dim, bias=False).to(device)
        self.b2 = nn.Parameter(torch.zeros(args.dim)).to(device)

        # If you want to do the same on the visual side, define parallel layers for W1, W3, etc.
        # Or keep it one-sided, depending on the paper’s approach.

        # ---- LayerNorms
        self.text_attn_ln = nn.LayerNorm(args.dim).to(device)
        self.text_ffn_ln  = nn.LayerNorm(args.dim).to(device)
        self.visl_attn_ln = nn.LayerNorm(args.dim).to(device)
        self.visl_ffn_ln  = nn.LayerNorm(args.dim).to(device)

        # Scaling factor in attention
        self.scale_factor = (args.dim // args.n_heads) ** -0.5

    def forward(self, text_in, visl_in):
        """
        text_in: (batch_size, T, hidden_dim)
        visl_in: (batch_size, V, hidden_dim)
        returns: (text_out, visl_out)
        """
        # -----------------------------------------------------------
        # 1) Prefix-Guided Interaction for text side
        #    text queries from text, keys+values from [text; visual].
        # -----------------------------------------------------------
        # Q_text = self.text_q_proj(text_in)
        # K_text = self.text_k_proj(text_in)
        # V_text = self.text_v_proj(text_in)

        # K_visl = self.visl_k_proj(visl_in)
        # V_visl = self.visl_v_proj(visl_in)

        ## K_cat_text = torch.cat([K_text, K_visl], dim=1)  # (B, T+V, H)
        ## V_cat_text = torch.cat([V_text, V_visl], dim=1)  # (B, T+V, H)

        # K_text = K_text + K_visl
        # V_text = V_text + V_visl

        # attn_scores_text = torch.matmul(Q_text, K_text.transpose(-2, -1)) * self.scale_factor
        # attn_weights_text = F.softmax(attn_scores_text, dim=-1)
        # attn_out_text = torch.matmul(attn_weights_text, V_text)

        # attn_out_text = (TPA(self.config).to(self.device))(text_in, text_in, text_in)
        attn_out_text, _ = nn.MultiheadAttention(self.config.dim, num_heads=self.config.n_heads).to(self.device)(text_in, text_in, text_in)

        # attn_out_text, _ = (nn.MultiheadAttention(self.hidden_dim, num_heads=self.num_heads).to(self.device))(Q_text, K_text, V_text)

        # attn_out_text = self.text_out_proj(attn_out_text) 
        text_attn_res = self.text_attn_ln(text_in + attn_out_text)  # residual

        # -----------------------------------------------------------
        # 2) Prefix-Guided Interaction for visual side
        #    visual queries from visual, keys+values from [visual; text].
        # -----------------------------------------------------------
        # Q_visl = self.visl_q_proj(visl_in)
        # K_visl2 = self.visl_k_proj(visl_in)
        # V_visl2 = self.visl_v_proj(visl_in)

        # K_text2 = self.text_k_proj(text_in)
        # V_text2 = self.text_v_proj(text_in)

        # K_cat_visl = torch.cat([K_visl2, K_text2], dim=1)
        # V_cat_visl = torch.cat([V_visl2, V_text2], dim=1)

        # attn_scores_visl = torch.matmul(Q_visl, (K_visl2+K_text2).transpose(-2, -1)) * self.scale_factor
        # attn_weights_visl = F.softmax(attn_scores_visl, dim=-1)
        # attn_out_visl = torch.matmul(attn_weights_visl, (V_visl2+V_text2))

        # attn_out_visl = (TPA(self.config).to(self.device))(visl_in, visl_in + text_in, visl_in + text_in)
        attn_out_visl, _ = nn.MultiheadAttention(self.config.dim, num_heads=self.config.n_heads).to(self.device)(visl_in, visl_in + text_in, visl_in + text_in)

        # attn_out_visl, _ = (nn.MultiheadAttention(self.hidden_dim, num_heads=self.num_heads).to(self.device))(Q_visl, (K_visl2 + K_text2), (V_visl2 + V_text2))

        # attn_out_visl = self.visl_out_proj(attn_out_visl)
        visl_attn_res = self.visl_attn_ln(visl_in + attn_out_visl)

        # -----------------------------------------------------------
        # 3) Correlation-Aware Fusion (CAF) in the FFN of text side,
        #    using the modified feed-forward formula:
        #      FFN(x_t) = ReLU( x_t W1 + Agg(x_v) W3 + b1 ) W2 + b2
        # -----------------------------------------------------------
        # (A) Compute token-wise similarity => (B,T,V)
        S = torch.matmul(text_attn_res, visl_attn_res.transpose(-2, -1))
        weights_for_fusion = F.softmax(S, dim=-1)  # sum over V dimension
        # (B) Weighted sum of visual tokens => shape (B,T,H)
        agg_visl = torch.matmul(weights_for_fusion, visl_attn_res)

        # (C) Feed-Forward with your custom formula
        # x_t => x_t W1,  agg(x_v) => agg_visl W3, plus biases
        xtW1      = self.W1(text_attn_res)        # (B,T,4H)
        xvW3      = self.W3(agg_visl)             # (B,T,4H)
        ff_input  = xtW1 + xvW3 + self.b1         # (B,T,4H)
        ff_hidden = F.relu(ff_input)             # ReLU
        ff_output = self.W2(ff_hidden) + self.b2  # (B,T,H)

        text_ffn_res = self.text_ffn_ln(text_attn_res + ff_output)  # final text

        # If you wish, do a parallel feed-forward for the visual side as well
        # or simply keep visl_attn_res if not needed.

        return text_ffn_res, visl_attn_res

class MEncoder(nn.Module):
    """
    Stacks multiple MEncoderLayer layers to fuse text + visual embeddings.
    """
    def __init__(self, config):
        super(MEncoder, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config = config
        self.layers = nn.ModuleList([
            MEncoderLayer(config).to(device)
            for _ in range(config.n_layers)
        ])

    def forward(self, text_in, visl_in):
        # text_in: (batch_size, T, hidden_dim)
        # visl_in: (batch_size, V, hidden_dim)
        text_in = text_in.unsqueeze(1)
        visl_in = visl_in.unsqueeze(1)
        for layer in self.layers:
            text_in, visl_in = layer(text_in, visl_in)
        return text_in, visl_in

class KBCModel(nn.Module, ABC):
    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        ranks = torch.ones(len(queries))
        fb_ling_f=r'../pre_train/matrix_fb_ling.npy'
        fb_visual_f=r'../pre_train/matrix_fb_visual.npy'
        wn_ling_f=r"../pre_train/matrix_wn_ling.npy"
        wn_visual_f=r"../pre_train/matrix_wn_visual.npy"
        fb_ling,fb_visual,wn_ling,wn_visual=torch.tensor(np.load(fb_ling_f)),torch.tensor(np.load(fb_visual_f)),torch.tensor(np.load(wn_ling_f)),torch.tensor(np.load(wn_visual_f))        
        multimodal_embeddings=[wn_ling,wn_visual]
        multimodal_embeddings1=[fb_ling,fb_visual]
        
        with tqdm(total=queries.shape[0], unit='ex') as bar:
            bar.set_description(f'Evaluation')
            with torch.no_grad():
                b_begin = 0
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    target_idxs = these_queries[:, 2].cpu().tolist()
                    scores, _ = self.forward(these_queries,multimodal_embeddings)
                    targets = torch.stack([scores[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)

                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())]
                        filter_out += [queries[b_begin + i, 2].item()]  
                        scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()
                    b_begin += batch_size
                    bar.update(batch_size)
        return ranks

    def _calc(self, lhs, rel, rhs, forward=True):
        denominator = torch.sqrt(rel[0] ** 2 + rel[1] ** 2 + rel[2] ** 2 + rel[3] ** 2)
        #print(denominator)
        rel_r = rel[0] / denominator
        rel_i = rel[1] / denominator
        rel_j = rel[2] / denominator
        rel_k = rel[3] / denominator
        

        A = lhs[0] * rel_r - lhs[1] * rel_i - lhs[2] * rel_j - lhs[3] * rel_k
        B = lhs[0] * rel_i + rel_r * lhs[1] + lhs[2] * rel_k - rel_j * lhs[3]
        C = lhs[0] * rel_j + rel_r * lhs[2] + lhs[3] * rel_i - rel_k * lhs[1]
        D = lhs[0] * rel_k + rel_r * lhs[3] + lhs[1] * rel_j - rel_i * lhs[2]
         
        if forward:
            score_r = A @ rhs[0].transpose(0, 1) + B @ rhs[1].transpose(0, 1) + C @ rhs[2].transpose(0, 1) + D @ rhs[3].transpose(0, 1)
        else:
            score_r = A * rhs[0] + B * rhs[1] + C * rhs[2] + D * rhs[3]
        return score_r

        # return A, B, C, D
    
    def quat_affine(self, quat: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], rel_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply an affine transformation in quaternion space.
        """
        # Stack quaternion components to shape [batch, 4, rank]
        Q = torch.stack(quat, dim=1)  # [batch, 4, rank]
        batch_size = Q.shape[0]
        rel_idx = rel_idx.long()
        
        # Map relation indices to valid range
        rel_idx = torch.clamp(rel_idx, 0, self.affine_B.shape[0]-1)
        
        # Get transformation matrices for each relation in the batch
        # A will have shape [batch_size, 4, 4]
        A = torch.matrix_exp(self.affine_B[rel_idx])
        
        # Apply batch matrix multiplication for each example individually
        Q_trans_list = []
        for i in range(batch_size):
            # Extract single example matrices
            A_i = A[i]  # Shape [4, 4]
            Q_i = Q[i:i+1]  # Shape [1, 4, rank]
            
            # Apply transformation and add translation
            trans_i = torch.bmm(A_i.unsqueeze(0), Q_i) + self.affine_t[rel_idx[i]].unsqueeze(0)
            Q_trans_list.append(trans_i)
        
        # Concatenate results back into batch
        Q_trans = torch.cat(Q_trans_list, dim=0)  # Shape [batch_size, 4, rank]
        
        # Split back into four components
        a_new = Q_trans[:, 0, :]
        b_new = Q_trans[:, 1, :]
        c_new = Q_trans[:, 2, :]
        d_new = Q_trans[:, 3, :]
        return a_new, b_new, c_new, d_new
    
    def mobius_transform(self, q, rel_idx):
        """
        Applies a quaternionic Möbius transformation to q, using the transformation parameters
        corresponding to each relation in rel_idx.
        
        For each sample in the batch:
            T(q) = (A ⊗ q + B) ⊗ (C ⊗ q + D)^{-1}
        where A, B, C, and D are the relation-specific parameters.
        """
        batch = q.size(0)
        rank = self.rank

        # Split the quaternion q into its four components.
        q_r = q[:, :rank]
        q_i = q[:, rank:2*rank]
        q_j = q[:, 2*rank:3*rank]
        q_k = q[:, 3*rank:]

        # Gather relation-specific Möbius parameters.
        # Each has shape (num_relations, 4, rank); here we select the ones for each sample.
        mobius_a = self.mobius_a[rel_idx]  # shape: (batch, 4, rank)
        mobius_b = self.mobius_b[rel_idx]
        mobius_c = self.mobius_c[rel_idx]
        mobius_d = self.mobius_d[rel_idx]

        # print(mobius_a.shape)
        # import sys
        # sys.exit()

        # Split into components.
        A_r = mobius_a[:, 0, :]; A_i = mobius_a[:, 1, :]
        A_j = mobius_a[:, 2, :]; A_k = mobius_a[:, 3, :]

        B_r = mobius_b[:, 0, :]; B_i = mobius_b[:, 1, :]
        B_j = mobius_b[:, 2, :]; B_k = mobius_b[:, 3, :]

        C_r = mobius_c[:, 0, :]; C_i = mobius_c[:, 1, :]
        C_j = mobius_c[:, 2, :]; C_k = mobius_c[:, 3, :]

        D_r = mobius_d[:, 0, :]; D_i = mobius_d[:, 1, :]
        D_j = mobius_d[:, 2, :]; D_k = mobius_d[:, 3, :]

        # Compute numerator: (A ⊗ q + B)
        num_r = A_r * q_r - A_i * q_i - A_j * q_j - A_k * q_k + B_r
        num_i = A_r * q_i + A_i * q_r + A_j * q_k - A_k * q_j + B_i
        num_j = A_r * q_j - A_i * q_k + A_j * q_r + A_k * q_i + B_j
        num_k = A_r * q_k + A_i * q_j - A_j * q_i + A_k * q_r + B_k

        # Compute denominator: (C ⊗ q + D)
        den_r = C_r * q_r - C_i * q_i - C_j * q_j - C_k * q_k + D_r
        den_i = C_r * q_i + C_i * q_r + C_j * q_k - C_k * q_j + D_i
        den_j = C_r * q_j - C_i * q_k + C_j * q_r + C_k * q_i + D_j
        den_k = C_r * q_k + C_i * q_j - C_j * q_i + C_k * q_r + D_k

        # Compute the norm squared of the denominator (adding a small epsilon for stability).
        den_norm_sq = den_r**2 + den_i**2 + den_j**2 + den_k**2 + 1e-8

        # Compute the inverse of the denominator.
        inv_den_r = den_r / den_norm_sq
        inv_den_i = -den_i / den_norm_sq
        inv_den_j = -den_j / den_norm_sq
        inv_den_k = -den_k / den_norm_sq

        # Multiply numerator by the inverse denominator (quaternion multiplication).
        trans_r = num_r * inv_den_r - num_i * inv_den_i - num_j * inv_den_j - num_k * inv_den_k
        trans_i = num_r * inv_den_i + num_i * inv_den_r + num_j * inv_den_k - num_k * inv_den_j
        trans_j = num_r * inv_den_j - num_i * inv_den_k + num_j * inv_den_r + num_k * inv_den_i
        trans_k = num_r * inv_den_k + num_i * inv_den_j - num_j * inv_den_i + num_k * inv_den_r

        transformed = torch.cat([trans_r, trans_i, trans_j, trans_k], dim=1)
        return transformed

    def forward(self, x, multi_modal):
        device = x.device

        # print(self.img_vec.shape)
        # print(self.img_vec[x[:,0]].shape)
        # print(self.img_vec[x[:,1]].shape)
        # print(self.img_vec[x[:,2]].shape)
        # import sys
        # sys.exit()

        mats_img = nn.Parameter(torch.Tensor(self.img_dimension, self.rank), requires_grad=True).to(device)
        nn.init.xavier_uniform_(mats_img)
        mats_ling = nn.Parameter(torch.Tensor(self.ling_dimension, self.rank), requires_grad=True).to(device)
        nn.init.xavier_uniform_(mats_ling)

        img_embeddings = self.img_vec.to(device).mm(mats_img.to(device))
        ling_embeddings = self.ling_vec.to(device).mm(mats_ling.to(device))

        # lhs_ling = ling_embeddings[x[:, 0]]
        # lhs_visl = img_embeddings[x[:, 0]]
        # fused_text, fused_visl = self.m_encoder(lhs_ling, lhs_visl)

        args = ModelArgs()
        args.dim = self.rank

        # config = {
        #     'rank': 4, 'q_rank': 12, 'num_heads': 8, 'hidden_dim': self.rank, 
        #     'num_layers': 4
        # }

        m_encoder = MEncoder(args)
        fused_text, fused_visl = m_encoder(ling_embeddings, img_embeddings)
        
        fused_embeddings = fused_text.squeeze(1) # => final fused vector for the head
        # print(fused_embeddings.shape)
        # import sys
        # sys.exit()

        # mats_fused = nn.Parameter(torch.Tensor(self.rank, 4 * self.rank), requires_grad=True).to(device)
        # nn.init.xavier_uniform_(mats_fused)

        # fused_embeddings = fused_embeddings.to(device).mm(mats_fused.to(device))   

        embedding = (self.embeddings[0].weight.to(device) * self.alpha + fused_embeddings * self.gamma) * self.scale

        # embedding = fused_embeddings * self.scale

        lhs = embedding[x[:, 0]]
        rel = self.embeddings[1](x[:, 1])
        rhs = embedding[x[:, 2]]
        # rhs = embedding

        # lhs_transformed = self.mobius_transform(lhs, x[:, 1])

        # Project entity embeddings into quaternion space
        # lhs_quat = (lhs_transformed[:, :self.rank],
        #             lhs_transformed[:, self.rank:2*self.rank],
        #             lhs_transformed[:, 2*self.rank:3*self.rank],
        #             lhs_transformed[:, 3*self.rank:])
        # rhs_quat = (rhs[:, :self.rank],
        #             rhs[:, self.rank:2*self.rank],
        #             rhs[:, 2*self.rank:3*self.rank],
        #             rhs[:, 3*self.rank:])
        # rel_quat = (rel[:, :self.rank],
        #             rel[:, self.rank:2*self.rank],
        #             rel[:, 2*self.rank:3*self.rank],
        #             rel[:, 3*self.rank:])
        
        # --- AFFINE TRANSFORMATION IN QUATERNION SPACE ---
        # Apply the learnable affine transformation to the quaternion embeddings.
        # lhs_quat = self.quat_affine(lhs_quat, x[:,1])
        # rhs_quat = self.quat_affine(rhs_quat, x[:,1])
        # ----------------------------------------------------

        # Compute the score (using the Hamilton product and inner product) against all entity embeddings
        # to_score = self.embeddings[0].weight
        # to_score = (to_score[:, :self.rank],
        #             to_score[:, self.rank:2*self.rank],
        #             to_score[:, 2*self.rank:3*self.rank],
        #             to_score[:, 3*self.rank:])
        # score = self._calc(lhs_quat, rel_quat, to_score)

        # factors = (torch.sqrt(lhs_quat[0] ** 2 + lhs_quat[1] ** 2 + lhs_quat[2] ** 2 + lhs_quat[3] ** 2),
        #            torch.sqrt(rel_quat[0] ** 2 + rel_quat[1] ** 2 + rel_quat[2] ** 2 + rel_quat[3] ** 2),
        #            torch.sqrt(rhs_quat[0] ** 2 + rhs_quat[1] ** 2 + rhs_quat[2] ** 2 + rhs_quat[3] ** 2))
        score = lhs + rel 
        factors = (torch.sqrt(lhs ** 2), torch.sqrt(rel ** 2), torch.sqrt(rhs ** 2))
        return score, factors

class model_wn(KBCModel):
    def __init__(self, sizes: Tuple[int, int, int], rank: int, init_size: float = 1e-3):
        super(model_wn, self).__init__()    
        self.sizes = sizes
        self.rank = rank
        
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.gamma = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        # self.beta = nn.Parameter(torch.tensor(0.3), requires_grad=True)
        self.scale = 0.1

        # Pre-trained embeddings
        wn_ling_f = r"../pre_train/matrix_wn_ling.npy"
        wn_visual_f = r"../pre_train/matrix_wn_visual.npy"
        
        wn_ling, wn_visual = torch.tensor(np.load(wn_ling_f)), torch.tensor(np.load(wn_visual_f))
        self.img_vec = wn_visual.to(torch.float32)
        self.ling_vec = wn_ling.to(torch.float32)

        self.img_dimension = wn_visual.shape[-1]
        self.ling_dimension = wn_ling.shape[-1]
        
        # self.mats_img = nn.Parameter(torch.Tensor(self.img_dimension, 4 * rank), requires_grad=True)
        # nn.init.xavier_uniform_(self.mats_img)
        # self.mats_ling = nn.Parameter(torch.Tensor(self.ling_dimension, 4 * rank), requires_grad=True)
        # nn.init.xavier_uniform_(self.mats_ling)
        
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 4 * rank, sparse=True) for s in sizes[:2]
        ])

        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        # self.embeddings1[0].weight.data *= init_size
        # self.embeddings1[1].weight.data *= init_size

#FBIMG    

class model_fb(KBCModel):
    def __init__(self, sizes: Tuple[int, int, int], rank: int, init_size: float = 1e-3):
        super(model_fb, self).__init__()    
        self.sizes = sizes
        self.rank = rank
        
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.gamma = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.scale = 0.1

        # Pre-trained embeddings
        fb_ling_f = r"../pre_train/matrix_fb_ling.npy"
        fb_visual_f = r"../pre_train/matrix_fb_visual.npy"
        
        fb_ling, fb_visual = torch.tensor(np.load(fb_ling_f)), torch.tensor(np.load(fb_visual_f))
        self.img_vec = fb_visual.to(torch.float32)
        self.ling_vec = fb_ling.to(torch.float32)

        self.img_dimension = fb_visual.shape[-1]
        self.ling_dimension = fb_ling.shape[-1]
        
        # Projection matrices
        # self.mats_img = nn.Parameter(torch.Tensor(self.img_dimension, 4 * rank), requires_grad=True)
        # nn.init.xavier_uniform_(self.mats_img)
        # self.mats_ling = nn.Parameter(torch.Tensor(self.ling_dimension, 4 * rank), requires_grad=True)
        # nn.init.xavier_uniform_(self.mats_ling)
        
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 4 * rank, sparse=True) for s in sizes[:2]
        ])

        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        # self.embeddings1[0].weight.data *= init_size
        # self.embeddings1[1].weight.data *= init_size
    
#db15k
class model_db(KBCModel):
    def __init__(self, sizes: Tuple[int, int, int], rank: int, init_size: float = 1e-3, num_relations = 2000):
        super(model_db, self).__init__()    
        self.sizes = sizes
        self.rank = rank
        
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.gamma = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        # self.beta = nn.Parameter(torch.tensor(0.3), requires_grad=True)
        self.scale = 0.1

        db_ling_f = r"../pre_train/DB15K-textual.pth"
        db_visual_f = r"../pre_train/DB15K-visual.pth"
        # db_numeric_f = r"../pre_train/DB15K-visual.pth"

        # db_visual = torch.load(db_visual_f)
        # print(db_visual.shape)
        db_ling, db_visual = torch.load(db_ling_f, weights_only=True), torch.load(db_visual_f, weights_only=True) #, torch.load(db_numeric_f, weights_only=True)
        
        self.img_vec = db_visual.to(torch.float32).to('cuda')
        self.ling_vec = db_ling.to(torch.float32).to('cuda')

        self.img_dimension = db_visual.shape[-1]
        self.ling_dimension = db_ling.shape[-1]

        # print(self.img_vec.shape)
        # print(self.img_dimension)
        # print(self.ling_vec.shape)
        # print(self.ling_dimension)
         
        # self.mats_img = nn.Parameter(torch.Tensor(self.img_dimension, 4 * rank), requires_grad=True)
        # nn.init.xavier_uniform_(self.mats_img)
        # self.mats_ling = nn.Parameter(torch.Tensor(self.ling_dimension, 4 * rank), requires_grad=True)
        # nn.init.xavier_uniform_(self.mats_ling)
        
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, rank, sparse=True) for s in sizes[:2]
        ])

        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        # self.embeddings1[0].weight.data *= init_size
        # self.embeddings1[1].weight.data *= init_size

        # Instead of directly learning A, we learn B and set A = exp(B) to guarantee invertibility.
        # self.affine_B = nn.Parameter(torch.zeros(num_relations, 4, 4))  # Unconstrained parameter
        # # t is a learnable translation (of shape [4, rank]) applied to each quaternion component.
        # self.affine_t = nn.Parameter(torch.zeros(num_relations, 4, rank))

        # self.mobius_a = nn.Parameter(torch.randn(num_relations, 4, rank))
        # self.mobius_b = nn.Parameter(torch.randn(num_relations, 4, rank))
        # self.mobius_c = nn.Parameter(torch.randn(num_relations, 4, rank))
        # self.mobius_d = nn.Parameter(torch.randn(num_relations, 4, rank))

#mkgw

class model_mkgw(KBCModel):
    def __init__(self, sizes: Tuple[int, int, int], rank: int, init_size: float = 1e-3, num_relations = 2000):
        super(model_mkgw, self).__init__()    
        self.sizes = sizes
        self.rank = rank
        
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.gamma = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        # self.beta = nn.Parameter(torch.tensor(0.3), requires_grad=True)
        self.scale = 0.1

        db_ling_f = r"../pre_train/MKG-W-textual.pth"
        db_visual_f = r"../pre_train/MKG-W-visual.pth"
        # db_numeric_f = r"../pre_train/DB15K-visual.pth"

        # # db_visual = torch.load(db_visual_f)
        # # print(db_visual.shape)
        db_ling, db_visual = torch.load(db_ling_f, weights_only=True), \
                                         torch.load(db_visual_f, weights_only=True)
        #                                  torch.load(db_numeric_f, weights_only=True)
        
        self.img_vec = db_visual.to(torch.float32).to('cuda')
        self.ling_vec = db_ling.to(torch.float32).to('cuda')

        self.img_dimension = db_visual.shape[-1]
        self.ling_dimension = db_ling.shape[-1]
         
        # self.img_embeddings = nn.Embedding.from_pretrained(img_emb).requires_grad_(False)
        # self.text_embeddings = nn.Embedding.from_pretrained(text_emb).requires_grad_(True)

        # print(self.img_embeddings.shape)
        
        # self.dim_e = 2 * rank
        # self.img_proj = nn.Sequential(
        #     nn.Linear(self.img_dimension, self.dim_e),
        #     nn.ReLU(),
        #     nn.Linear(self.dim_e, self.dim_e)
        # )
        # self.text_proj = nn.Sequential(
        #     nn.Linear(self.ling_dimension, self.dim_e),
        #     nn.ReLU(),
        #     nn.Linear(self.dim_e, self.dim_e)
        # )

        # print(db_ling.shape)
        # print(db_visual.shape)

        # self.numeric_vec = db_numeric.to(torch.float32)

        # self.numeric_dimension = db_numeric.shape[-1]
        
        # Projection matrices
        # self.mats_img = nn.Parameter(torch.Tensor(self.img_dimension, 4 * rank), requires_grad=True)
        # nn.init.xavier_uniform_(self.mats_img)
        # self.mats_ling = nn.Parameter(torch.Tensor(self.ling_dimension, 4 * rank), requires_grad=True)
        # nn.init.xavier_uniform_(self.mats_ling)

    
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, rank, sparse=True) for s in sizes[:2]
        ])

        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        # self.embeddings1[0].weight.data *= init_size
        # self.embeddings1[1].weight.data *= init_size    

        # Instead of directly learning A, we learn B and set A = exp(B) to guarantee invertibility.
        # self.affine_B = nn.Parameter(torch.zeros(num_relations, 4, 4))  # Unconstrained parameter
        # # t is a learnable translation (of shape [4, rank]) applied to each quaternion component.
        # self.affine_t = nn.Parameter(torch.zeros(num_relations, 4, rank))

        # self.mobius_a = nn.Parameter(torch.randn(num_relations, 4, rank))
        # self.mobius_b = nn.Parameter(torch.randn(num_relations, 4, rank))
        # self.mobius_c = nn.Parameter(torch.randn(num_relations, 4, rank))
        # self.mobius_d = nn.Parameter(torch.randn(num_relations, 4, rank))
    
class model_mkgy(KBCModel):
    def __init__(self, sizes: Tuple[int, int, int], rank: int, init_size: float = 1e-3, num_relations = 2000):
        super(model_mkgy, self).__init__()    
        self.sizes = sizes
        self.rank = rank
        
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.gamma = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        # self.beta = nn.Parameter(torch.tensor(0.3), requires_grad=True)
        self.scale = 0.1

        db_ling_f = r"../pre_train/MKG-Y-textual.pth"
        db_visual_f = r"../pre_train/MKG-Y-visual.pth"
        # db_numeric_f = r"../pre_train/DB15K-visual.pth"

        # # db_visual = torch.load(db_visual_f)
        # # print(db_visual.shape)
        db_ling, db_visual = torch.load(db_ling_f, weights_only=True), torch.load(db_visual_f, weights_only=True)
        
        self.img_vec = db_visual.to(torch.float32).to('cuda')
        self.ling_vec = db_ling.to(torch.float32).to('cuda')

        self.img_dimension = db_visual.shape[-1]
        self.ling_dimension = db_ling.shape[-1]
         
        # self.img_embeddings = nn.Embedding.from_pretrained(img_emb).requires_grad_(False)
        # self.text_embeddings = nn.Embedding.from_pretrained(text_emb).requires_grad_(True)

        # Projection matrices
        # self.mats_img = nn.Parameter(torch.Tensor(self.img_dimension, 4 * rank), requires_grad=True)
        # nn.init.xavier_uniform_(self.mats_img)
        # self.mats_ling = nn.Parameter(torch.Tensor(self.ling_dimension, 4 * rank), requires_grad=True)
        # nn.init.xavier_uniform_(self.mats_ling)
    
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, rank, sparse=True) for s in sizes[:2]
        ])

        
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        # self.embeddings1[0].weight.data *= init_size
        # self.embeddings1[1].weight.data *= init_size

        # Instead of directly learning A, we learn B and set A = exp(B) to guarantee invertibility.
        # self.affine_B = nn.Parameter(torch.zeros(num_relations, 4, 4))  # Unconstrained parameter
        # # t is a learnable translation (of shape [4, rank]) applied to each quaternion component.
        # self.affine_t = nn.Parameter(torch.zeros(num_relations, 4, rank))

        # self.mobius_a = nn.Parameter(torch.randn(num_relations, 4, rank))
        # self.mobius_b = nn.Parameter(torch.randn(num_relations, 4, rank))
        # self.mobius_c = nn.Parameter(torch.randn(num_relations, 4, rank))
        # self.mobius_d = nn.Parameter(torch.randn(num_relations, 4, rank))
