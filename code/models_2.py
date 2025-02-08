from abc import ABC, abstractmethod
from typing import Tuple, List, Dict

import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import pickle
import math

# Octonion Fusion (Gated Mechanism)
class OctonionFusion_gm(nn.Module):
    def __init__(self, ling_dim, img_dim, rank):
        super(OctonionFusion_gm, self).__init__()
        assert ling_dim == img_dim, "Linguistic and visual dimensions must match for Octonion Fusion."
        self.rank = rank

        # Projection layers for eight octonion components
        self.projections = nn.ModuleList([nn.Linear(ling_dim, rank) for _ in range(8)])

        # Gating layers for shared (addition) and differential (subtraction) components
        self.gates = nn.ModuleList([nn.Linear(2 * rank, rank) for _ in range(8)])

        for proj in self.projections:
            nn.init.xavier_uniform_(proj.weight)
        for gate in self.gates:
            nn.init.xavier_uniform_(gate.weight)

    def forward(self, ling_embeddings, img_embeddings):
        device = ling_embeddings.device
        for proj in self.projections:
            proj.to(device)
        for gate in self.gates:
            gate.to(device)
            
        # Project linguistic and visual embeddings into octonion components
        ling_components = [proj(ling_embeddings) for proj in self.projections[:4]]
        img_components = [proj(img_embeddings) for proj in self.projections[4:]]

        # Compute shared and differential components
        shared_components = [ling + img for ling, img in zip(ling_components, img_components)]
        diff_components = [ling - img for ling, img in zip(ling_components, img_components)]

        # Apply gating mechanism to each component
        gated_components = []
        for i in range(8):
            shared = shared_components[i % 4]  # Loop through shared components
            diff = diff_components[i % 4]  # Loop through differential components

            gate_input = torch.cat([shared, diff], dim=-1)  # Concatenate shared and differential
            gate = torch.sigmoid(self.gates[i](gate_input))  # Compute gate
            gated = gate * shared + (1 - gate) * diff  # Weighted sum
            gated_components.append(gated)

        # Concatenate all gated components into the octonion representation
        octonion_fusion = torch.cat(gated_components, dim=-1)  # Shape: [batch_size, 8 * rank]
        return octonion_fusion

# Octonion Fusion (Weighted Sum)
class OctonionFusion_ws(nn.Module):
    def __init__(self, ling_dim, img_dim, rank):
        super(OctonionFusion_ws, self).__init__()
        assert ling_dim == img_dim, "Linguistic and visual dimensions must match for Octonion Fusion."
        self.rank = rank

        # Projection layers for eight octonion components
        self.ling_proj = nn.Linear(ling_dim, rank)
        self.img_proj = nn.Linear(img_dim, rank)
        self.projections = nn.ModuleList([nn.Linear(rank, rank) for _ in range(8)])

        # Learnable weights for shared (addition) and differential (subtraction) components
        self.weights = nn.ParameterList([nn.Parameter(torch.tensor(0.5), requires_grad=True) for _ in range(16)])

    def forward(self, ling_embeddings, img_embeddings):
        device = ling_embeddings.device

        # Ensure all tensors are on the same device
        ling_embeddings = ling_embeddings.to(device)
        img_embeddings = img_embeddings.to(device)
        self.ling_proj = self.ling_proj.to(device)
        self.img_proj = self.img_proj.to(device)
        for proj in self.projections:
            proj.to(device)

        # Project linguistic and visual embeddings to rank dimensions
        ling_proj = self.ling_proj(ling_embeddings)
        img_proj = self.img_proj(img_embeddings)

        # Compute octonion components with weighted sharing
        components = []
        for i in range(8):
            shared = (self.weights[2 * i] / (self.weights[2 * i] + self.weights[2 * i + 1])) * (
                self.projections[i](ling_proj) + self.projections[i](img_proj)
            )
            diff = (self.weights[2 * i + 1] / (self.weights[2 * i] + self.weights[2 * i + 1])) * (
                self.projections[i](ling_proj) - self.projections[i](img_proj)
            )
            components.append(shared + diff)

        # Concatenate all components into the octonion representation
        octonion_fusion = torch.cat(components, dim=-1)  # Shape: [batch_size, 8 * rank]
        return octonion_fusion


class MixtureOfExperts(nn.Module):
    def __init__(self, ling_dim, img_dim, output_dim):
        super(MixtureOfExperts, self).__init__()
        self.fusion1 = nn.Linear(ling_dim + img_dim, output_dim)  # Simple concatenation
        self.fusion2 = nn.Bilinear(ling_dim, img_dim, output_dim)  # Bilinear
        self.gate = nn.Linear(output_dim * 2, 2)

    def forward(self, ling_emb, img_emb):
        concat_fused = self.fusion1(torch.cat([ling_emb, img_emb], dim=-1))
        bilinear_fused = self.fusion2(ling_emb, img_emb)
        gate = torch.softmax(self.gate(torch.cat([concat_fused, bilinear_fused], dim=-1)), dim=-1)
        fused = gate[:, 0].unsqueeze(-1) * concat_fused + gate[:, 1].unsqueeze(-1) * bilinear_fused
        return torch.relu(fused)

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

        img_embeddings = self.img_vec.to(device).mm(self.mats_img.to(device))
        ling_embeddings = self.ling_vec.to(device).mm(self.mats_ling.to(device))
        
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

        ## Quaternion Fusion (Gated Mechanism)
        octonion_fusion = OctonionFusion_gm(8 * self.rank, 8 * self.rank, self.rank)
        fused_embeddings = octonion_fusion(ling_embeddings, img_embeddings)

        ## Quaternion Fusion (Weighted Sum)
        # quaternion_fusion = QuaternionFusion_ws(8 * self.rank, 8 * self.rank, self.rank)
        # fused_embeddings = quaternion_fusion(ling_embeddings, img_embeddings)

        # print(fused_embeddings.shape)

        #normal
        # structure_embedding = self.embeddings[0].weight.to(device) * self.scale

        embedding = (self.embeddings[0].weight.to(device) * 0.9 + fused_embeddings * 0.1) * self.scale

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
        
        self.mats_img = nn.Parameter(torch.Tensor(self.img_dimension, 8 * rank), requires_grad=True)
        nn.init.xavier_uniform_(self.mats_img)
        self.mats_ling = nn.Parameter(torch.Tensor(self.ling_dimension, 8 * rank), requires_grad=True)
        nn.init.xavier_uniform_(self.mats_ling)
        
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
        self.mats_img = nn.Parameter(torch.Tensor(self.img_dimension, 8 * rank), requires_grad=True)
        nn.init.xavier_uniform_(self.mats_img)
        self.mats_ling = nn.Parameter(torch.Tensor(self.ling_dimension, 8 * rank), requires_grad=True)
        nn.init.xavier_uniform_(self.mats_ling)
        
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
         
        self.mats_img = nn.Parameter(torch.Tensor(self.img_dimension, 8 * rank), requires_grad=True)
        nn.init.xavier_uniform_(self.mats_img)
        self.mats_ling = nn.Parameter(torch.Tensor(self.ling_dimension, 8 * rank), requires_grad=True)
        nn.init.xavier_uniform_(self.mats_ling)
        
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
        self.mats_img = nn.Parameter(torch.Tensor(self.img_dimension, 8 * rank), requires_grad=True)
        nn.init.xavier_uniform_(self.mats_img)
        self.mats_ling = nn.Parameter(torch.Tensor(self.ling_dimension, 8 * rank), requires_grad=True)
        nn.init.xavier_uniform_(self.mats_ling)
    
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
        self.mats_img = nn.Parameter(torch.Tensor(self.img_dimension, 8 * rank), requires_grad=True)
        nn.init.xavier_uniform_(self.mats_img)
        self.mats_ling = nn.Parameter(torch.Tensor(self.ling_dimension, 8 * rank), requires_grad=True)
        nn.init.xavier_uniform_(self.mats_ling)
    
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 8 * rank, sparse=True) for s in sizes[:2]
        ])

        
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        # self.embeddings1[0].weight.data *= init_size
        # self.embeddings1[1].weight.data *= init_size
