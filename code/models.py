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

from efficient_kan import KAN

class GatedFusion(nn.Module):
    def __init__(self, embed_dim):
        """
        Initialize the gating mechanism.
        
        Args:
            embed_dim (int): The dimension of each embedding.
        """
        super(GatedFusion, self).__init__()
        # A linear transformation to compute the gate.
        # We concatenate the two embeddings, so input dimension is 2 * embed_dim,
        # and we produce an output of size embed_dim.
        self.gate_fc = KAN([2 * embed_dim, embed_dim])
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, embedding1, embedding2):
        """
        Forward pass to compute the fused embedding.
        
        Args:
            embedding1 (torch.Tensor): Tensor of shape (batch_size, embed_dim)
            embedding2 (torch.Tensor): Tensor of shape (batch_size, embed_dim)
        
        Returns:
            torch.Tensor: Fused embedding of shape (batch_size, embed_dim)
        """
        # Concatenate embeddings along the feature dimension
        combined = torch.cat([embedding1, embedding2], dim=1)
        
        # Compute the gate vector using the linear layer followed by sigmoid activation
        gate = self.sigmoid(self.gate_fc(combined))
        
        # Fuse the embeddings
        fused_embedding = gate * embedding1 + (1 - gate) * embedding2
        return fused_embedding

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

        fft_img = torch.fft.fft(img_embeddings, dim=1)
        fft_ling = torch.fft.fft(ling_embeddings, dim=1)

        img_embeddings = torch.real(torch.fft.ifft(fft_img, dim=1))
        ling_embeddings = torch.real(torch.fft.ifft(fft_ling, dim=1))

        # lhs_ling = ling_embeddings[x[:, 0]]
        # lhs_visl = img_embeddings[x[:, 0]]
        # fused_text, fused_visl = self.m_encoder(lhs_ling, lhs_visl)
        
        fused_embeddings = GatedFusion(self.rank)(img_embeddings, ling_embeddings)
        # print(fused_embeddings.shape)
        # import sys
        # sys.exit()

        mats_fused = nn.Parameter(torch.Tensor(self.rank, 4 * self.rank), requires_grad=True).to(device)
        nn.init.xavier_uniform_(mats_fused)

        fused_embeddings = fused_embeddings.to(device).mm(mats_fused.to(device))   

        embedding = (self.embeddings[0].weight.to(device) * self.alpha + fused_embeddings * self.gamma) * self.scale

        lhs = embedding[x[:, 0]]
        rel = self.embeddings[1](x[:, 1])
        rhs = embedding[x[:, 2]]
        # rhs = embedding

        # lhs_transformed = self.mobius_transform(lhs, x[:, 1])

        # Project entity embeddings into quaternion space
        lhs_quat = (lhs[:, :self.rank],
                    lhs[:, self.rank:2*self.rank],
                    lhs[:, 2*self.rank:3*self.rank],
                    lhs[:, 3*self.rank:])
        rhs_quat = (rhs[:, :self.rank],
                    rhs[:, self.rank:2*self.rank],
                    rhs[:, 2*self.rank:3*self.rank],
                    rhs[:, 3*self.rank:])
        rel_quat = (rel[:, :self.rank],
                    rel[:, self.rank:2*self.rank],
                    rel[:, 2*self.rank:3*self.rank],
                    rel[:, 3*self.rank:])
        
        # --- AFFINE TRANSFORMATION IN QUATERNION SPACE ---
        # Apply the learnable affine transformation to the quaternion embeddings.
        # lhs_quat = self.quat_affine(lhs_quat, x[:,1])
        # rhs_quat = self.quat_affine(rhs_quat, x[:,1])
        # ----------------------------------------------------

        # Compute the score (using the Hamilton product and inner product) against all entity embeddings
        to_score = self.embeddings[0].weight
        to_score = (to_score[:, :self.rank],
                    to_score[:, self.rank:2*self.rank],
                    to_score[:, 2*self.rank:3*self.rank],
                    to_score[:, 3*self.rank:])
        score = self._calc(lhs_quat, rel_quat, to_score)

        factor = (torch.sqrt(lhs_quat[0] ** 2 + lhs_quat[1] ** 2 + lhs_quat[2] ** 2 + lhs_quat[3] ** 2),
                   torch.sqrt(rel_quat[0] ** 2 + rel_quat[1] ** 2 + rel_quat[2] ** 2 + rel_quat[3] ** 2),
                   torch.sqrt(rhs_quat[0] ** 2 + rhs_quat[1] ** 2 + rhs_quat[2] ** 2 + rhs_quat[3] ** 2))
        
        # complex
        # lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        # rel = rel[:, :self.rank], rel[:, self.rank:]
        # rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        # to_score = self.embeddings[0].weight
        # to_score = to_score[:, :self.rank], to_score[:, self.rank:]  
              
        # score = ( (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) 
        #         + (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1))  # (h, r, t) 

        # factor = ( 
        #     torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
        #     torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
        #     torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2),   
        # )

        return score, factor

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
            nn.Embedding(s, 4 * rank, sparse=True) for s in sizes[:2]
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
            nn.Embedding(s, 4 * rank, sparse=True) for s in sizes[:2]
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
            nn.Embedding(s, 4 * rank, sparse=True) for s in sizes[:2]
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
