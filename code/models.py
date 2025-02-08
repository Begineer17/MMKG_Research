from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Optional

import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import pickle
import math
from kan import KAN


# # Quaternion Fusion (Gated Mechanism -> complicated version)
# class QuaternionFusion_gm(nn.Module):
#     def __init__(self, ling_dim, img_dim, rank):
#         super(QuaternionFusion_gm, self).__init__()
#         assert ling_dim == img_dim, "Linguistic and visual dimensions must match for Quaternion Fusion."
#         self.rank = rank
#         # self.ling_proj = nn.Linear(ling_dim, rank)
#         # self.img_proj = nn.Linear(img_dim, rank)
#         # self.real_proj = nn.Linear(rank, rank)
#         # self.i_proj = nn.Linear(rank, rank)
#         # self.j_proj = nn.Linear(rank, rank)
#         # self.k_proj = nn.Linear(rank, rank)
#         device = 'cpu'
#         if torch.cuda.is_available():
#             device = 'cuda'
#         self.ling_proj = KAN(width=[ling_dim, rank], grid=3, k=3, seed=1, device=device)
#         self.img_proj = KAN(width=[img_dim, rank], grid=3, k=3, seed=1, device=device)
#         self.real_proj = KAN(width=[rank, rank], grid=3, k=3, seed=1, device=device)
#         self.i_proj = KAN(width=[rank, rank], grid=3, k=3, seed=1, device=device)
#         self.j_proj = KAN(width=[rank, rank], grid=3, k=3, seed=1, device=device)
#         self.k_proj = KAN(width=[rank, rank], grid=3, k=3, seed=1, device=device)

#         # Gating layers for shared (addition) and differential (subtraction) components
#         # self.real_gate = nn.Linear(2 * rank, rank)  # Gate for real part
#         # self.i_gate = nn.Linear(2 * rank, rank)     # Gate for i part
#         # self.j_gate = nn.Linear(2 * rank, rank)     # Gate for j part
#         # self.k_gate = nn.Linear(2 * rank, rank)     # Gate for k part

#         self.real_gate = KAN(width=[2 * rank, rank], grid=3, k=3, seed=1, device=device)  # Gate for real part
#         self.i_gate = KAN(width=[2 * rank, rank], grid=3, k=3, seed=1, device=device)    # Gate for i part
#         self.j_gate = KAN(width=[2 * rank, rank], grid=3, k=3, seed=1, device=device)     # Gate for j part
#         self.k_gate = KAN(width=[2 * rank, rank], grid=3, k=3, seed=1, device=device)   # Gate for k part

#     def forward(self, ling_embeddings, img_embeddings):
#         device = ling_embeddings.device
#         # Ensure all tensors are on the same device
#         ling_embeddings = ling_embeddings.to(device)
#         img_embeddings = img_embeddings.to(device)
#         self.ling_proj = self.ling_proj.to(device)
#         self.img_proj = self.img_proj.to(device)
#         self.real_proj = self.real_proj.to(device)
#         self.i_proj = self.i_proj.to(device)
#         self.j_proj = self.j_proj.to(device)
#         self.k_proj = self.k_proj.to(device)
#         self.real_gate = self.real_gate.to(device)
#         self.i_gate = self.i_gate.to(device)
#         self.j_gate = self.j_gate.to(device)
#         self.k_gate = self.k_gate.to(device)

#         # Project to rank dimension
#         ling_proj = self.ling_proj(ling_embeddings)
#         img_proj = self.img_proj(img_embeddings)

#         # Construct quaternion components
#         shared_real = self.real_proj(ling_proj) + self.real_proj(img_proj)
#         diff_real = self.real_proj(ling_proj) - self.real_proj(img_proj)

#         shared_i = self.i_proj(ling_proj) + self.i_proj(img_proj)
#         diff_i = self.i_proj(ling_proj) - self.i_proj(img_proj)

#         shared_j = self.j_proj(ling_proj) + self.j_proj(img_proj)
#         diff_j = self.j_proj(ling_proj) - self.j_proj(img_proj)

#         shared_k = self.k_proj(ling_proj) + self.k_proj(img_proj)
#         diff_k = self.k_proj(ling_proj) - self.k_proj(img_proj)

#         # Apply gating mechanism
#         gated_real = torch.sigmoid(self.real_gate(torch.cat([shared_real, diff_real], dim=-1))) * shared_real + \
#                      (1 - torch.sigmoid(self.real_gate(torch.cat([shared_real, diff_real], dim=-1)))) * diff_real

#         gated_i = torch.sigmoid(self.i_gate(torch.cat([shared_i, diff_i], dim=-1))) * shared_i + \
#                   (1 - torch.sigmoid(self.i_gate(torch.cat([shared_i, diff_i], dim=-1)))) * diff_i

#         gated_j = torch.sigmoid(self.j_gate(torch.cat([shared_j, diff_j], dim=-1))) * shared_j + \
#                   (1 - torch.sigmoid(self.j_gate(torch.cat([shared_j, diff_j], dim=-1)))) * diff_j

#         gated_k = torch.sigmoid(self.k_gate(torch.cat([shared_k, diff_k], dim=-1))) * shared_k + \
#                   (1 - torch.sigmoid(self.k_gate(torch.cat([shared_k, diff_k], dim=-1)))) * diff_k

#         # Concatenate gated components into quaternion representation
#         quaternion_fusion = torch.cat([gated_real, gated_i, gated_j, gated_k], dim=-1)  # Shape: [batch_size, 4 * rank]
#         return quaternion_fusion
    
# # Quaternion Fusion (Weighted Sum -> simple version)
# class QuaternionFusion_ws(nn.Module):
#     def __init__(self, ling_dim, img_dim, rank):
#         super(QuaternionFusion_ws, self).__init__()
#         assert ling_dim == img_dim, "Linguistic and visual dimensions must match for Quaternion Fusion."
#         self.rank = rank
#         self.ling_proj = nn.Linear(ling_dim, rank)
#         self.img_proj = nn.Linear(img_dim, rank)
#         self.real_proj = nn.Linear(rank, rank)
#         self.i_proj = nn.Linear(rank, rank)
#         self.j_proj = nn.Linear(rank, rank)
#         self.k_proj = nn.Linear(rank, rank)

#         self.r1 = nn.Parameter(torch.tensor(0.5), requires_grad=True)
#         self.r2 = nn.Parameter(torch.tensor(0.5), requires_grad=True)

#         self.i1 = nn.Parameter(torch.tensor(0.5), requires_grad=True)
#         self.i2 = nn.Parameter(torch.tensor(0.5), requires_grad=True)

#         self.j1 = nn.Parameter(torch.tensor(0.5), requires_grad=True)
#         self.j2 = nn.Parameter(torch.tensor(0.5), requires_grad=True)

#         self.k1 = nn.Parameter(torch.tensor(0.5), requires_grad=True)
#         self.k2 = nn.Parameter(torch.tensor(0.5), requires_grad=True)

#     def forward(self, ling_embeddings, img_embeddings):
#         device = ling_embeddings.device
#         # Ensure all tensors are on the same device
#         ling_embeddings = ling_embeddings.to(device)
#         img_embeddings = img_embeddings.to(device)
#         self.ling_proj = self.ling_proj.to(device)
#         self.img_proj = self.img_proj.to(device)
#         self.real_proj = self.real_proj.to(device)
#         self.i_proj = self.i_proj.to(device)
#         self.j_proj = self.j_proj.to(device)
#         self.k_proj = self.k_proj.to(device)

#         # Project to rank dimension
#         ling_proj = self.ling_proj(ling_embeddings)
#         img_proj = self.img_proj(img_embeddings)

#         # Construct quaternion components
#         shared_real = (self.a1 / (self.a1 + self.a2)) * (self.real_proj(ling_proj) + self.real_proj(img_proj)) + (self.a2 / (self.a1 + self.a2)) * (self.real_proj(ling_proj) - self.real_proj(img_proj))
#         shared_i = (self.i1 / (self.i1 + self.i2)) * (self.i_proj(ling_proj) + self.i_proj(img_proj)) + (self.i2 / (self.i1 + self.i2)) * (self.i_proj(ling_proj) - self.i_proj(img_proj))
#         shared_j = (self.j1 / (self.j1 + self.j2)) * (self.j_proj(ling_proj) + self.j_proj(img_proj)) + (self.j2 / (self.j1 + self.j2)) * (self.j_proj(ling_proj) - self.j_proj(img_proj))
#         shared_k = (self.k1 / (self.k1 + self.k2)) * (self.k_proj(ling_proj) + self.k_proj(img_proj)) + (self.k2 / (self.k1 + self.k2)) * (self.k_proj(ling_proj) - self.k_proj(img_proj))

#         # Concatenate gated components into quaternion representation
#         quaternion_fusion = torch.cat([shared_real, shared_i, shared_j, shared_k], dim=-1)  # Shape: [batch_size, 4 * rank]
#         return quaternion_fusion

# class MixtureOfExperts(nn.Module):
#     def __init__(self, ling_dim, img_dim, output_dim):
#         super(MixtureOfExperts, self).__init__()
#         self.fusion1 = nn.Linear(ling_dim + img_dim, output_dim)  # Simple concatenation
#         self.fusion2 = nn.Bilinear(ling_dim, img_dim, output_dim)  # Bilinear
#         self.gate = nn.Linear(output_dim * 2, 2)

#     def forward(self, ling_emb, img_emb):
#         concat_fused = self.fusion1(torch.cat([ling_emb, img_emb], dim=-1))
#         bilinear_fused = self.fusion2(ling_emb, img_emb)
#         gate = torch.softmax(self.gate(torch.cat([concat_fused, bilinear_fused], dim=-1)), dim=-1)
#         fused = gate[:, 0].unsqueeze(-1) * concat_fused + gate[:, 1].unsqueeze(-1) * bilinear_fused
#         return torch.relu(fused)

# class KBCModel(nn.Module, ABC):
#     def get_ranking(
#             self, queries: torch.Tensor,
#             filters: Dict[Tuple[int, int], List[int]],
#             batch_size: int = 1000, chunk_size: int = -1
#     ):
#         ranks = torch.ones(len(queries))
#         fb_ling_f=r'../pre_train/matrix_fb_ling.npy'
#         fb_visual_f=r'../pre_train/matrix_fb_visual.npy'
#         wn_ling_f=r"../pre_train/matrix_wn_ling.npy"
#         wn_visual_f=r"../pre_train/matrix_wn_visual.npy"
#         fb_ling,fb_visual,wn_ling,wn_visual=torch.tensor(np.load(fb_ling_f)),torch.tensor(np.load(fb_visual_f)),torch.tensor(np.load(wn_ling_f)),torch.tensor(np.load(wn_visual_f))        
#         multimodal_embeddings=[wn_ling,wn_visual]
#         multimodal_embeddings1=[fb_ling,fb_visual]
        
#         with tqdm(total=queries.shape[0], unit='ex') as bar:
#             bar.set_description(f'Evaluation')
#             with torch.no_grad():
#                 b_begin = 0
#                 while b_begin < len(queries):
#                     these_queries = queries[b_begin:b_begin + batch_size]
#                     target_idxs = these_queries[:, 2].cpu().tolist()
#                     scores, _ = self.forward(these_queries,multimodal_embeddings)
#                     targets = torch.stack([scores[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)

#                     for i, query in enumerate(these_queries):
#                         filter_out = filters[(query[0].item(), query[1].item())]
#                         filter_out += [queries[b_begin + i, 2].item()]  
#                         scores[i, torch.LongTensor(filter_out)] = -1e6
#                     ranks[b_begin:b_begin + batch_size] += torch.sum(
#                         (scores >= targets).float(), dim=1
#                     ).cpu()
#                     b_begin += batch_size
#                     bar.update(batch_size)
#         return ranks

#     def _calc(self, lhs, rel, rhs, forward=True):
#         denominator = torch.sqrt(rel[0] ** 2 + rel[1] ** 2 + rel[2] ** 2 + rel[3] ** 2)
#         #print(denominator)
#         rel_r = rel[0] / denominator
#         rel_i = rel[1] / denominator
#         rel_j = rel[2] / denominator
#         rel_k = rel[3] / denominator
        

#         A = lhs[0] * rel_r - lhs[1] * rel_i - lhs[2] * rel_j - lhs[3] * rel_k
#         B = lhs[0] * rel_i + rel_r * lhs[1] + lhs[2] * rel_k - rel_j * lhs[3]
#         C = lhs[0] * rel_j + rel_r * lhs[2] + lhs[3] * rel_i - rel_k * lhs[1]
#         D = lhs[0] * rel_k + rel_r * lhs[3] + lhs[1] * rel_j - rel_i * lhs[2]
         
#         if forward:
#             score_r = A @ rhs[0].transpose(0, 1) + B @ rhs[1].transpose(0, 1) + C @ rhs[2].transpose(0, 1) + D @ rhs[3].transpose(0, 1)
#         else:
#             score_r = A * rhs[0] + B * rhs[1] + C * rhs[2] + D * rhs[3]
#         return score_r
    
#     def forward(self, x, multi_modal):
#         device = x.device

#         img_embeddings = self.img_vec.to(device).mm(self.mats_img.to(device))
#         ling_embeddings = self.ling_vec.to(device).mm(self.mats_ling.to(device))
        
#         ## Normal fusion
#         # fused_embeddings = (img_embeddings + ling_embeddings) * self.scale

#         # Concatenation with Non-Linear Transformation
#         fused_embeddings = torch.cat([ling_embeddings, img_embeddings], dim=-1).to(device)
#         fused_embeddings = torch.relu(KAN(width=[2 * (4 * self.rank), 4 * self.rank], grid=3, k=3, seed=1, device=device).to(device)(fused_embeddings)) * self.scale

#         ## Weighted Fusion with Learnable Parameters
#         # self.ling_weight = nn.Parameter(torch.tensor(0.5), requires_grad=True)
#         # self.img_weight = nn.Parameter(torch.tensor(0.5), requires_grad=True)

#         # weights_sum = self.ling_weight + self.img_weight
#         # fused_embeddings = (self.ling_weight / weights_sum * ling_embeddings + 
#         #                     self.img_weight / weights_sum * img_embeddings) * self.scale

#         ## Bilinear Fusion
#         # self.bilinear = nn.Bilinear(4 * self.rank, 4 * self.rank, 4 * self.rank)
#         # fused_embeddings = torch.relu(self.bilinear(ling_embeddings, img_embeddings)) * self.scale

#         ## Gated Fusion
#         # self.gate = nn.Linear(2 * (4 * self.rank), 1).to(device)
#         # # Concatenate along the last dimension
#         # concat_embeddings = torch.cat([ling_embeddings, img_embeddings], dim=-1)  # Shape: [12842, 8000]
#         # # Apply the gating mechanism
#         # gate = torch.sigmoid(self.gate(concat_embeddings))  # Shape: [12842, 1]
#         # # Fuse embeddings with gating
#         # fused_embeddings = gate * ling_embeddings + (1 - gate) * img_embeddings

#         ## Mixture of Experts
#         # experts = MixtureOfExperts(4 * self.rank, 4 * self.rank, 4 * self.rank)
#         # fused_embeddings = experts(ling_embeddings, img_embeddings)

#         ## Quaternion Fusion (Gated Mechanism)
#         # quaternion_fusion = QuaternionFusion_gm(4 * self.rank, 4 * self.rank, self.rank)
#         # fused_embeddings = quaternion_fusion(ling_embeddings, img_embeddings)

#         ## Quaternion Fusion (Weighted Sum)
#         # quaternion_fusion = QuaternionFusion_ws(4 * self.rank, 4 * self.rank, self.rank)
#         # fused_embeddings = quaternion_fusion(ling_embeddings, img_embeddings)

#         # print(fused_embeddings.shape)

#         #normal
#         # structure_embedding = self.embeddings[0].weight.to(device) * self.scale

#         embedding = (self.embeddings[0].weight.to(device) * self.alpha + fused_embeddings * self.gamma) * self.scale

#         lhs = embedding[x[:, 0]]
#         rel = self.embeddings[1](x[:, 1])
#         rhs = embedding[x[:, 2]]
#         # rhs = embedding

#         # Split embeddings into real and quaternion parts for scoring
#         lhs = lhs[:, :self.rank], lhs[:, self.rank:2*self.rank], lhs[:, 2*self.rank:3*self.rank], lhs[:, 3*self.rank:]
#         rel = rel[:, :self.rank], rel[:, self.rank:2*self.rank], rel[:, 2*self.rank:3*self.rank], rel[:, 3*self.rank:]
#         rhs = rhs[:, :self.rank], rhs[:, self.rank:2*self.rank], rhs[:, 2*self.rank:3*self.rank], rhs[:, 3*self.rank:]

#         to_score = self.embeddings[0].weight
#         to_score = to_score[:, :self.rank], to_score[:, self.rank:2*self.rank], to_score[:, 2*self.rank:3*self.rank], to_score[:, 3*self.rank:]

#         score = self._calc(lhs, rel, to_score)
#         factors = ( torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2 + lhs[2] ** 2 + lhs[3] ** 2),
#                     torch.sqrt(rel[0] ** 2 + rel[1] ** 2 + rel[2] ** 2 + rel[3] ** 2),
#                     torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2 + rhs[2] ** 2 + rhs[3] ** 2))

#         return score, factors
    

# class model_wn(KBCModel):
#     def __init__(self, sizes: Tuple[int, int, int], rank: int, init_size: float = 1e-3):
#         super(model_wn, self).__init__()    
#         self.sizes = sizes
#         self.rank = rank
        
#         self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
#         self.gamma = nn.Parameter(torch.tensor(0.5), requires_grad=True)
#         # self.beta = nn.Parameter(torch.tensor(0.3), requires_grad=True)
#         self.scale = 0.1

#         # Pre-trained embeddings
#         wn_ling_f = r"../pre_train/matrix_wn_ling.npy"
#         wn_visual_f = r"../pre_train/matrix_wn_visual.npy"
        
#         wn_ling, wn_visual = torch.tensor(np.load(wn_ling_f)), torch.tensor(np.load(wn_visual_f))
#         self.img_vec = wn_visual.to(torch.float32)
#         self.ling_vec = wn_ling.to(torch.float32)

#         self.img_dimension = wn_visual.shape[-1]
#         self.ling_dimension = wn_ling.shape[-1]
        
#         self.mats_img = nn.Parameter(torch.Tensor(self.img_dimension, 4 * rank), requires_grad=True)
#         nn.init.xavier_uniform_(self.mats_img)
#         self.mats_ling = nn.Parameter(torch.Tensor(self.ling_dimension, 4 * rank), requires_grad=True)
#         nn.init.xavier_uniform_(self.mats_ling)
        
#         self.embeddings = nn.ModuleList([
#             nn.Embedding(s, 4 * rank, sparse=True) for s in sizes[:2]
#         ])

#         self.embeddings[0].weight.data *= init_size
#         self.embeddings[1].weight.data *= init_size
#         # self.embeddings1[0].weight.data *= init_size
#         # self.embeddings1[1].weight.data *= init_size

# #FBIMG    

# class model_fb(KBCModel):
#     def __init__(self, sizes: Tuple[int, int, int], rank: int, init_size: float = 1e-3):
#         super(model_fb, self).__init__()    
#         self.sizes = sizes
#         self.rank = rank
        
#         self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
#         self.gamma = nn.Parameter(torch.tensor(0.5), requires_grad=True)
#         self.scale = 0.2

#         # Pre-trained embeddings
#         fb_ling_f = r"../pre_train/matrix_fb_ling.npy"
#         fb_visual_f = r"../pre_train/matrix_fb_visual.npy"
        
#         fb_ling, fb_visual = torch.tensor(np.load(fb_ling_f)), torch.tensor(np.load(fb_visual_f))
#         self.img_vec = fb_visual.to(torch.float32)
#         self.ling_vec = fb_ling.to(torch.float32)

#         self.img_dimension = fb_visual.shape[-1]
#         self.ling_dimension = fb_ling.shape[-1]
        
#         # Projection matrices
#         self.mats_img = nn.Parameter(torch.Tensor(self.img_dimension, 4 * rank), requires_grad=True)
#         nn.init.xavier_uniform_(self.mats_img)
#         self.mats_ling = nn.Parameter(torch.Tensor(self.ling_dimension, 4 * rank), requires_grad=True)
#         nn.init.xavier_uniform_(self.mats_ling)
        
#         self.embeddings = nn.ModuleList([
#             nn.Embedding(s, 4 * rank, sparse=True) for s in sizes[:2]
#         ])

#         self.embeddings[0].weight.data *= init_size
#         self.embeddings[1].weight.data *= init_size
#         # self.embeddings1[0].weight.data *= init_size
#         # self.embeddings1[1].weight.data *= init_size
    
# #db15k
# class model_db(KBCModel):
#     def __init__(self, sizes: Tuple[int, int, int], rank: int, init_size: float = 1e-3):
#         super(model_db, self).__init__()    
#         self.sizes = sizes
#         self.rank = rank
        
#         self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
#         self.gamma = nn.Parameter(torch.tensor(0.5), requires_grad=True)
#         # self.beta = nn.Parameter(torch.tensor(0.3), requires_grad=True)
#         self.scale = 0.1

#         db_ling_f = r"../pre_train/DB15K-textual.pth"
#         db_visual_f = r"../pre_train/DB15K-visual.pth"
#         # db_numeric_f = r"../pre_train/DB15K-visual.pth"

#         # db_visual = torch.load(db_visual_f)
#         # print(db_visual.shape)
#         db_ling, db_visual = torch.load(db_ling_f, weights_only=True), torch.load(db_visual_f, weights_only=True) #, torch.load(db_numeric_f, weights_only=True)
        
#         self.img_vec = db_visual.to(torch.float32).to('cuda')
#         self.ling_vec = db_ling.to(torch.float32).to('cuda')

#         self.img_dimension = db_visual.shape[-1]
#         self.ling_dimension = db_ling.shape[-1]

#         # print(self.img_vec.shape)
#         # print(self.img_dimension)
#         # print(self.ling_vec.shape)
#         # print(self.ling_dimension)
         
#         self.mats_img = nn.Parameter(torch.Tensor(self.img_dimension, 4 * rank), requires_grad=True)
#         nn.init.xavier_uniform_(self.mats_img)
#         self.mats_ling = nn.Parameter(torch.Tensor(self.ling_dimension, 4 * rank), requires_grad=True)
#         nn.init.xavier_uniform_(self.mats_ling)
        
#         self.embeddings = nn.ModuleList([
#             nn.Embedding(s, 4 * rank, sparse=True) for s in sizes[:2]
#         ])

#         self.embeddings[0].weight.data *= init_size
#         self.embeddings[1].weight.data *= init_size
#         # self.embeddings1[0].weight.data *= init_size
#         # self.embeddings1[1].weight.data *= init_size


# #mkgw

# class model_mkgw(KBCModel):
#     def __init__(self, sizes: Tuple[int, int, int], rank: int, init_size: float = 1e-3):
#         super(model_mkgw, self).__init__()    
#         self.sizes = sizes
#         self.rank = rank
        
#         self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
#         self.gamma = nn.Parameter(torch.tensor(0.5), requires_grad=True)
#         # self.beta = nn.Parameter(torch.tensor(0.3), requires_grad=True)
#         self.scale = 0.1

#         db_ling_f = r"../pre_train/MKG-W-textual.pth"
#         db_visual_f = r"../pre_train/MKG-W-visual.pth"
#         # db_numeric_f = r"../pre_train/DB15K-visual.pth"

#         # # db_visual = torch.load(db_visual_f)
#         # # print(db_visual.shape)
#         db_ling, db_visual = torch.load(db_ling_f, weights_only=True), \
#                                          torch.load(db_visual_f, weights_only=True)
#         #                                  torch.load(db_numeric_f, weights_only=True)
        
#         self.img_vec = db_visual.to(torch.float32).to('cuda')
#         self.ling_vec = db_ling.to(torch.float32).to('cuda')

#         self.img_dimension = db_visual.shape[-1]
#         self.ling_dimension = db_ling.shape[-1]
         
#         # self.img_embeddings = nn.Embedding.from_pretrained(img_emb).requires_grad_(False)
#         # self.text_embeddings = nn.Embedding.from_pretrained(text_emb).requires_grad_(True)

#         # print(self.img_embeddings.shape)
        
#         # self.dim_e = 2 * rank
#         # self.img_proj = nn.Sequential(
#         #     nn.Linear(self.img_dimension, self.dim_e),
#         #     nn.ReLU(),
#         #     nn.Linear(self.dim_e, self.dim_e)
#         # )
#         # self.text_proj = nn.Sequential(
#         #     nn.Linear(self.ling_dimension, self.dim_e),
#         #     nn.ReLU(),
#         #     nn.Linear(self.dim_e, self.dim_e)
#         # )

#         # print(db_ling.shape)
#         # print(db_visual.shape)

#         # self.numeric_vec = db_numeric.to(torch.float32)

#         # self.numeric_dimension = db_numeric.shape[-1]
        
#         # Projection matrices
#         self.mats_img = nn.Parameter(torch.Tensor(self.img_dimension, 4 * rank), requires_grad=True)
#         nn.init.xavier_uniform_(self.mats_img)
#         self.mats_ling = nn.Parameter(torch.Tensor(self.ling_dimension, 4 * rank), requires_grad=True)
#         nn.init.xavier_uniform_(self.mats_ling)

    
#         self.embeddings = nn.ModuleList([
#             nn.Embedding(s, 4 * rank, sparse=True) for s in sizes[:2]
#         ])

        
#         self.embeddings[0].weight.data *= init_size
#         self.embeddings[1].weight.data *= init_size
#         # self.embeddings1[0].weight.data *= init_size
#         # self.embeddings1[1].weight.data *= init_size      
        
    
# class model_mkgy(KBCModel):
#     def __init__(self, sizes: Tuple[int, int, int], rank: int, init_size: float = 1e-3):
#         super(model_mkgy, self).__init__()    
#         self.sizes = sizes
#         self.rank = rank
        
#         self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
#         self.gamma = nn.Parameter(torch.tensor(0.5), requires_grad=True)
#         # self.beta = nn.Parameter(torch.tensor(0.3), requires_grad=True)
#         self.scale = 0.1

#         db_ling_f = r"../pre_train/MKG-Y-textual.pth"
#         db_visual_f = r"../pre_train/MKG-Y-visual.pth"
#         # db_numeric_f = r"../pre_train/DB15K-visual.pth"

#         # # db_visual = torch.load(db_visual_f)
#         # # print(db_visual.shape)
#         db_ling, db_visual = torch.load(db_ling_f, weights_only=True), torch.load(db_visual_f, weights_only=True)
        
#         self.img_vec = db_visual.to(torch.float32).to('cuda')
#         self.ling_vec = db_ling.to(torch.float32).to('cuda')

#         self.img_dimension = db_visual.shape[-1]
#         self.ling_dimension = db_ling.shape[-1]
         
#         # self.img_embeddings = nn.Embedding.from_pretrained(img_emb).requires_grad_(False)
#         # self.text_embeddings = nn.Embedding.from_pretrained(text_emb).requires_grad_(True)

#         # Projection matrices
#         self.mats_img = nn.Parameter(torch.Tensor(self.img_dimension, 4 * rank), requires_grad=True)
#         nn.init.xavier_uniform_(self.mats_img)
#         self.mats_ling = nn.Parameter(torch.Tensor(self.ling_dimension, 4 * rank), requires_grad=True)
#         nn.init.xavier_uniform_(self.mats_ling)
    
#         self.embeddings = nn.ModuleList([
#             nn.Embedding(s, 4 * rank, sparse=True) for s in sizes[:2]
#         ])

        
#         self.embeddings[0].weight.data *= init_size
#         self.embeddings[1].weight.data *= init_size
#         # self.embeddings1[0].weight.data *= init_size
#         # self.embeddings1[1].weight.data *= init_size

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class EfficientTransformer(nn.Module):
    """Transformer using custom cross-attention mechanisms."""
    def __init__(self, input_dim, hidden_dim, num_heads, ff_dim, num_layers, attention_type="multi_head"):
        super(EfficientTransformer, self).__init__()
        self.num_layers = num_layers
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if attention_type == "multi_head":
            self.attention_layers = nn.ModuleList([
                nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True).to(device) for _ in range(num_layers)
            ])
        else:
            raise ValueError("Unsupported attention type")

        # Feed-Forward Network (FFN)
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, ff_dim),
                nn.ReLU(),
                nn.Linear(ff_dim, hidden_dim)
            ).to(device) for _ in range(num_layers)
        ])

        # Layer Normalization and Residual Connections
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim).to(device) for _ in range(num_layers * 2)])

        # Input Projection for embeddings
        self.input_proj = nn.Linear(input_dim, hidden_dim).to(device) if input_dim != hidden_dim else nn.Identity()
        self.input_proj_2 = nn.Linear(input_dim, hidden_dim).to(device) if input_dim != hidden_dim else nn.Identity()
        self.input_proj_3 = nn.Linear(input_dim, hidden_dim).to(device) if input_dim != hidden_dim else nn.Identity()

    def forward(self, queries, keys, values):
        """Apply cross-attention mechanism."""
        queries = self.input_proj(queries)
        keys = self.input_proj_2(keys)
        values = self.input_proj_3(values)
        
        for i in range(self.num_layers):
            attn_out, _ = self.attention_layers[i](queries, keys, values)
            queries = self.layer_norms[2 * i](queries + attn_out)  # Residual Connection
            
            # Feed-Forward Network
            ff_out = self.ff_layers[i](queries)
            queries = self.layer_norms[2 * i + 1](queries + ff_out)  # Residual Connection

        return queries
    
class EarlyFusionTransformer(nn.Module):
    def __init__(self, ling_dim, img_dim, fused_dim, hidden_dim, num_heads, ff_dim, num_layers):
        super(EarlyFusionTransformer, self).__init__()
        self.ling_proj = nn.Linear(ling_dim, hidden_dim)
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, fused_dim)
        self.transformer = EfficientTransformer(hidden_dim, hidden_dim, num_heads, ff_dim, num_layers)

    def forward(self, ling_emb, img_emb):
        """Applies cross-attention between linguistic and visual embeddings."""
        device = ling_emb.device
        ling_emb = ling_emb.to(device)
        img_emb = img_emb.to(device)
        self.ling_proj = self.ling_proj.to(device)
        self.img_proj = self.img_proj.to(device)
        self.output_proj = self.output_proj.to(device)

        ling_emb = self.ling_proj(ling_emb)
        img_emb = self.img_proj(img_emb)
        self.transformer = self.transformer.to(device)
        
        output = self.transformer(queries=ling_emb, keys=img_emb, values=img_emb)
        output = self.output_proj(output)
        return output


# # Octonion Fusion (Gated Mechanism)
# class OctonionFusion_gm(nn.Module):
#     def __init__(self, ling_dim, img_dim, rank):
#         super(OctonionFusion_gm, self).__init__()
#         assert ling_dim == img_dim, "Linguistic and visual dimensions must match for Octonion Fusion."
#         self.rank = rank

#         # Projection layers for eight octonion components
#         self.projections = nn.ModuleList([nn.Linear(ling_dim, rank) for _ in range(8)])

#         # Gating layers for shared (addition) and differential (subtraction) components
#         self.gates = nn.ModuleList([nn.Linear(2 * rank, rank) for _ in range(8)])

#     def forward(self, ling_embeddings, img_embeddings):
#         device = ling_embeddings.device
#         for proj in self.projections:
#             proj.to(device)
#         for gate in self.gates:
#             gate.to(device)
            
#         # Project linguistic and visual embeddings into octonion components
#         ling_components = [proj(ling_embeddings) for proj in self.projections[:4]]
#         img_components = [proj(img_embeddings) for proj in self.projections[4:]]

#         # Compute shared and differential components
#         shared_components = [ling + img for ling, img in zip(ling_components, img_components)]
#         diff_components = [ling - img for ling, img in zip(ling_components, img_components)]

#         # Apply gating mechanism to each component
#         gated_components = []
#         for i in range(8):
#             shared = shared_components[i % 4]  # Loop through shared components
#             diff = diff_components[i % 4]  # Loop through differential components

#             gate_input = torch.cat([shared, diff], dim=-1)  # Concatenate shared and differential
#             gate = torch.sigmoid(self.gates[i](gate_input))  # Compute gate
#             gated = gate * shared + (1 - gate) * diff  # Weighted sum
#             gated_components.append(gated)

#         # Concatenate all gated components into the octonion representation
#         octonion_fusion = torch.cat(gated_components, dim=-1)  # Shape: [batch_size, 8 * rank]
#         return octonion_fusion

# # Octonion Fusion (Weighted Sum)
# class OctonionFusion_ws(nn.Module):
#     def __init__(self, ling_dim, img_dim, rank):
#         super(OctonionFusion_ws, self).__init__()
#         assert ling_dim == img_dim, "Linguistic and visual dimensions must match for Octonion Fusion."
#         self.rank = rank

#         # Projection layers for eight octonion components
#         self.ling_proj = nn.Linear(ling_dim, rank)
#         self.img_proj = nn.Linear(img_dim, rank)
#         self.projections = nn.ModuleList([nn.Linear(rank, rank) for _ in range(8)])

#         # Learnable weights for shared (addition) and differential (subtraction) components
#         self.weights = nn.ParameterList([nn.Parameter(torch.tensor(0.5), requires_grad=True) for _ in range(16)])

#     def forward(self, ling_embeddings, img_embeddings):
#         device = ling_embeddings.device

#         # Ensure all tensors are on the same device
#         ling_embeddings = ling_embeddings.to(device)
#         img_embeddings = img_embeddings.to(device)
#         self.ling_proj = self.ling_proj.to(device)
#         self.img_proj = self.img_proj.to(device)
#         for proj in self.projections:
#             proj.to(device)

#         # Project linguistic and visual embeddings to rank dimensions
#         ling_proj = self.ling_proj(ling_embeddings)
#         img_proj = self.img_proj(img_embeddings)

#         # Compute octonion components with weighted sharing
#         components = []
#         for i in range(8):
#             shared = (self.weights[2 * i] / (self.weights[2 * i] + self.weights[2 * i + 1])) * (
#                 self.projections[i](ling_proj) + self.projections[i](img_proj)
#             )
#             diff = (self.weights[2 * i + 1] / (self.weights[2 * i] + self.weights[2 * i + 1])) * (
#                 self.projections[i](ling_proj) - self.projections[i](img_proj)
#             )
#             components.append(shared + diff)

#         # Concatenate all components into the octonion representation
#         octonion_fusion = torch.cat(components, dim=-1)  # Shape: [batch_size, 8 * rank]
#         return octonion_fusion


# class MixtureOfExperts(nn.Module):
#     def __init__(self, ling_dim, img_dim, output_dim):
#         super(MixtureOfExperts, self).__init__()
#         self.fusion1 = nn.Linear(ling_dim + img_dim, output_dim)  # Simple concatenation
#         self.fusion2 = nn.Bilinear(ling_dim, img_dim, output_dim)  # Bilinear
#         self.gate = nn.Linear(output_dim * 2, 2)

#     def forward(self, ling_emb, img_emb):
#         concat_fused = self.fusion1(torch.cat([ling_emb, img_emb], dim=-1))
#         bilinear_fused = self.fusion2(ling_emb, img_emb)
#         gate = torch.softmax(self.gate(torch.cat([concat_fused, bilinear_fused], dim=-1)), dim=-1)
#         fused = gate[:, 0].unsqueeze(-1) * concat_fused + gate[:, 1].unsqueeze(-1) * bilinear_fused
#         return torch.relu(fused)

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
        # Extract components
        lhs_r, lhs_i, lhs_j, lhs_k, lhs_l, lhs_il, lhs_jl, lhs_kl = lhs
        rel_r, rel_i, rel_j, rel_k, rel_l, rel_il, rel_jl, rel_kl = rel
        rhs_r, rhs_i, rhs_j, rhs_k, rhs_l, rhs_il, rhs_jl, rhs_kl = rhs

        # Octonion multiplication (following standard rules)
        A = (
            lhs_r * rel_r - lhs_i * rel_i - lhs_j * rel_j - lhs_k * rel_k -
            lhs_l * rel_l - lhs_il * rel_il - lhs_jl * rel_jl - lhs_kl * rel_kl
        )
        B = (
            lhs_r * rel_i + lhs_i * rel_r + lhs_j * rel_k - lhs_k * rel_j +
            lhs_l * rel_il - lhs_il * rel_l - lhs_jl * rel_kl + lhs_kl * rel_jl
        )
        C = (
            lhs_r * rel_j - lhs_i * rel_k + lhs_j * rel_r + lhs_k * rel_i +
            lhs_l * rel_jl + lhs_il * rel_kl - lhs_jl * rel_l - lhs_kl * rel_il
        )
        D = (
            lhs_r * rel_k + lhs_i * rel_j - lhs_j * rel_i + lhs_k * rel_r +
            lhs_l * rel_kl - lhs_il * rel_jl + lhs_jl * rel_il - lhs_kl * rel_l
        )
        E = (
            lhs_r * rel_l - lhs_i * rel_il - lhs_j * rel_jl - lhs_k * rel_kl +
            lhs_l * rel_r + lhs_il * rel_i + lhs_jl * rel_j + lhs_kl * rel_k
        )
        F = (
            lhs_r * rel_il + lhs_i * rel_l - lhs_j * rel_kl + lhs_k * rel_jl -
            lhs_l * rel_i + lhs_il * rel_r - lhs_jl * rel_k + lhs_kl * rel_j
        )
        G = (
            lhs_r * rel_jl + lhs_i * rel_kl + lhs_j * rel_l - lhs_k * rel_il -
            lhs_l * rel_j + lhs_il * rel_k + lhs_jl * rel_r - lhs_kl * rel_i
        )
        H = (
            lhs_r * rel_kl - lhs_i * rel_jl + lhs_j * rel_il + lhs_k * rel_l -
            lhs_l * rel_k - lhs_il * rel_j + lhs_jl * rel_i + lhs_kl * rel_r
        )

        if forward:
            score_r = (
                A @ rhs_r.transpose(0, 1) + B @ rhs_i.transpose(0, 1) +
                C @ rhs_j.transpose(0, 1) + D @ rhs_k.transpose(0, 1) +
                E @ rhs_l.transpose(0, 1) + F @ rhs_il.transpose(0, 1) +
                G @ rhs_jl.transpose(0, 1) + H @ rhs_kl.transpose(0, 1)
            )
        else:
            score_r = (
                A * rhs_r + B * rhs_i + C * rhs_j + D * rhs_k +
                E * rhs_l + F * rhs_il + G * rhs_jl + H * rhs_kl
            )
        return score_r
    
    def forward(self, x, multi_modal):
        device = x.device

        # img_embeddings = self.img_vec.to(device).mm(self.mats_img.to(device))
        # ling_embeddings = self.ling_vec.to(device).mm(self.mats_ling.to(device))
        
        ## Normal fusion
        # fused_embeddings = (img_embeddings + ling_embeddings) * self.scale

        ## Concatenation with Non-Linear Transformation
        # fused_embeddings = torch.cat([ling_embeddings, img_embeddings], dim=-1).to(device)
        # fused_embeddings = torch.relu(nn.Linear(2 * (8 * self.rank), 8 * self.rank).to(device)(fused_embeddings)) * self.scale

        ## Weighted Fusion with Learnable Parameters
        # self.ling_weight = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        # self.img_weight = nn.Parameter(torch.tensor(0.5), requires_grad=True)

        # weights_sum = self.ling_weight + self.img_weight
        # fused_embeddings = (self.ling_weight / weights_sum * ling_embeddings + 
        #                     self.img_weight / weights_sum * img_embeddings) * self.scale

        ## Bilinear Fusion
        # self.bilinear = nn.Bilinear(8 * self.rank, 8 * self.rank, 8 * self.rank)
        # fused_embeddings = torch.relu(self.bilinear(ling_embeddings, img_embeddings)) * self.scale

        ## Gated Fusion
        # self.gate = nn.Linear(2 * (8 * self.rank), 1).to(device)
        # # Concatenate along the last dimension
        # concat_embeddings = torch.cat([ling_embeddings, img_embeddings], dim=-1)  # Shape: [12842, 8000]
        # # Apply the gating mechanism
        # gate = torch.sigmoid(self.gate(concat_embeddings))  # Shape: [12842, 1]
        # # Fuse embeddings with gating
        # fused_embeddings = gate * ling_embeddings + (1 - gate) * img_embeddings

        ## Mixture of Experts
        # experts = MixtureOfExperts(8 * self.rank, 8 * self.rank, 8 * self.rank)
        # fused_embeddings = experts(ling_embeddings, img_embeddings)

        ## Octonion Fusion (Gated Mechanism)
        # octonion_fusion = OctonionFusion_gm(8 * self.rank, 8 * self.rank, self.rank)
        # fused_embeddings = octonion_fusion(ling_embeddings, img_embeddings)

        ## Octonion Fusion (Weighted Sum)
        # octonion_fusion = OctonionFusion_ws(8 * self.rank, 8 * self.rank, self.rank)
        # fused_embeddings = octonion_fusion(ling_embeddings, img_embeddings)

        ## Early Fusion Transformer
        mats_img = nn.Parameter(torch.Tensor(self.img_dimension, self.rank), requires_grad=True).to(device)
        nn.init.xavier_uniform_(mats_img)
        mats_ling = nn.Parameter(torch.Tensor(self.ling_dimension, self.rank), requires_grad=True).to(device)
        nn.init.xavier_uniform_(mats_ling)

        img_embeddings = self.img_vec.to(device).mm(mats_img.to(device))
        ling_embeddings = self.ling_vec.to(device).mm(mats_ling.to(device))

        # self.gate = nn.Linear(2 * self.rank, 1).to(device)
        # # Concatenate along the last dimension
        # concat_embeddings = torch.cat([ling_embeddings, img_embeddings], dim=-1)  # Shape: [12842, rank]
        # # Apply the gating mechanism
        # gate = torch.sigmoid(self.gate(concat_embeddings))  # Shape: [12842, 1]
        # # Fuse embeddings with gating
        # fused_embeddings = gate * ling_embeddings + (1 - gate) * img_embeddings

        early_fusion = EarlyFusionTransformer(ling_dim=self.rank, img_dim=self.rank, fused_dim=self.rank, hidden_dim=2 * self.rank, num_heads=8, ff_dim=4 * self.rank, num_layers=4)
        fused_embeddings = early_fusion(ling_embeddings, img_embeddings).to(device)

        mats_fused = nn.Parameter(torch.Tensor(self.rank, 8 * self.rank), requires_grad=True).to(device)
        nn.init.xavier_uniform_(mats_fused)

        fused_embeddings = fused_embeddings.mm(mats_fused)

        # print(fused_embeddings.shape)

        #normal
        # structure_embedding = self.embeddings[0].weight.to(device) * self.scale

        embedding = (self.embeddings[0].weight.to(device) * self.alpha + fused_embeddings * self.gamma) * self.scale

        lhs = embedding[x[:, 0]]
        rel = self.embeddings[1](x[:, 1])
        rhs = embedding[x[:, 2]]
        # rhs = embedding

        # Split embeddings into real and quaternion parts for scoring
        lhs = tuple(lhs[:, i*self.rank:(i+1)*self.rank] for i in range(8))
        rel = tuple(rel[:, i*self.rank:(i+1)*self.rank] for i in range(8))
        rhs = tuple(rhs[:, i*self.rank:(i+1)*self.rank] for i in range(8))

        to_score = self.embeddings[0].weight
        to_score = tuple(to_score[:, i*self.rank:(i+1)*self.rank] for i in range(8))

        score = self._calc(lhs, rel, to_score)
        factors = (
            torch.sqrt(sum(lhs[i] ** 2 for i in range(8))),
            torch.sqrt(sum(rel[i] ** 2 for i in range(8))),
            torch.sqrt(sum(rhs[i] ** 2 for i in range(8)))
        )

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
        
        # self.mats_img = nn.Parameter(torch.Tensor(self.img_dimension, 8 * rank), requires_grad=True)
        # nn.init.xavier_uniform_(self.mats_img)
        # self.mats_ling = nn.Parameter(torch.Tensor(self.ling_dimension, 8 * rank), requires_grad=True)
        # nn.init.xavier_uniform_(self.mats_ling)
        
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 8 * rank, sparse=True) for s in sizes[:2]
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
        self.scale = 0.2

        # Pre-trained embeddings
        fb_ling_f = r"../pre_train/matrix_fb_ling.npy"
        fb_visual_f = r"../pre_train/matrix_fb_visual.npy"
        
        fb_ling, fb_visual = torch.tensor(np.load(fb_ling_f)), torch.tensor(np.load(fb_visual_f))
        self.img_vec = fb_visual.to(torch.float32)
        self.ling_vec = fb_ling.to(torch.float32)

        self.img_dimension = fb_visual.shape[-1]
        self.ling_dimension = fb_ling.shape[-1]
        
        # Projection matrices
        # self.mats_img = nn.Parameter(torch.Tensor(self.img_dimension, 8 * rank), requires_grad=True)
        # nn.init.xavier_uniform_(self.mats_img)
        # self.mats_ling = nn.Parameter(torch.Tensor(self.ling_dimension, 8 * rank), requires_grad=True)
        # nn.init.xavier_uniform_(self.mats_ling)
        
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 8 * rank, sparse=True) for s in sizes[:2]
        ])

        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        # self.embeddings1[0].weight.data *= init_size
        # self.embeddings1[1].weight.data *= init_size
    
#db15k
class model_db(KBCModel):
    def __init__(self, sizes: Tuple[int, int, int], rank: int, init_size: float = 1e-3):
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
         
        # self.mats_img = nn.Parameter(torch.Tensor(self.img_dimension, 8 * rank), requires_grad=True)
        # nn.init.xavier_uniform_(self.mats_img)
        # self.mats_ling = nn.Parameter(torch.Tensor(self.ling_dimension, 8 * rank), requires_grad=True)
        # nn.init.xavier_uniform_(self.mats_ling)
        
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 8 * rank, sparse=True) for s in sizes[:2]
        ])

        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        # self.embeddings1[0].weight.data *= init_size
        # self.embeddings1[1].weight.data *= init_size


#mkgw

class model_mkgw(KBCModel):
    def __init__(self, sizes: Tuple[int, int, int], rank: int, init_size: float = 1e-3):
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
        # self.mats_img = nn.Parameter(torch.Tensor(self.img_dimension, 8 * rank), requires_grad=True)
        # nn.init.xavier_uniform_(self.mats_img)
        # self.mats_ling = nn.Parameter(torch.Tensor(self.ling_dimension, 8 * rank), requires_grad=True)
        # nn.init.xavier_uniform_(self.mats_ling)
    
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 8 * rank, sparse=True) for s in sizes[:2]
        ])

        
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        # self.embeddings1[0].weight.data *= init_size
        # self.embeddings1[1].weight.data *= init_size      
        
    
class model_mkgy(KBCModel):
    def __init__(self, sizes: Tuple[int, int, int], rank: int, init_size: float = 1e-3):
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
        # self.mats_img = nn.Parameter(torch.Tensor(self.img_dimension, 8 * rank), requires_grad=True)
        # nn.init.xavier_uniform_(self.mats_img)
        # self.mats_ling = nn.Parameter(torch.Tensor(self.ling_dimension, 8 * rank), requires_grad=True)
        # nn.init.xavier_uniform_(self.mats_ling)
    
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 8 * rank, sparse=True) for s in sizes[:2]
        ])

        
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        # self.embeddings1[0].weight.data *= init_size
        # self.embeddings1[1].weight.data *= init_size
