import tqdm
import torch
from torch import nn
from torch import optim
import numpy as np
from models import KBCModel
from regularizers import Regularizer

class KBCOptimizer(object):
    def __init__(
            self, model: KBCModel, regularizer: list, optimizer: optim.Optimizer, batch_size: int = 256,
            verbose: bool = True
    ):
        self.model = model
        self.regularizer = regularizer[0]
        self.regularizer2 = regularizer[1]
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose

        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def epoch(self, examples: torch.LongTensor, e=0, weight=None):
        self.model.train()
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        loss = nn.CrossEntropyLoss(reduction='mean', weight=weight)    

        fb_ling_f=r'../pre_train/matrix_fb_ling.npy'
        fb_visual_f=r'../pre_train/matrix_fb_visual.npy'
        wn_ling_f=r"../pre_train/matrix_wn_ling.npy"
        wn_visual_f=r"../pre_train/matrix_wn_visual.npy"
        fb_ling,fb_visual,wn_ling,wn_visual=torch.tensor(np.load(fb_ling_f)),torch.tensor(np.load(fb_visual_f)),torch.tensor(np.load(wn_ling_f)),torch.tensor(np.load(wn_visual_f))        
        multimodal_embeddings=[wn_ling,wn_visual]
        multimodal_embeddings1=[fb_ling,fb_visual]
        
        with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[
                    b_begin:b_begin + self.batch_size
                ].cuda()

                predictions, factors = self.model.forward(input_batch,multimodal_embeddings) 
                truth = input_batch[:, 2]
                l_fit = loss(predictions, truth)
                # l_fit = loss(predictions, truth)
                l_reg = self.regularizer.forward(factors)

                l = l_fit + l_reg
                # print(l_fit)
                # print(l_reg)

                self.optimizer.zero_grad()
                l.backward()

                self.optimizer.step()
                torch.cuda.empty_cache()
                    
                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                bar.set_postfix(loss=f'{l.item():.5f}', reg=f'{l_reg.item():.5f}')
        # if (e + 1) % 5 == 0:  # In mỗi 5 epoch
            # print(f"w1: {self.model.weight_struc}, w2: {self.model.weight_multimodal}, w3: {self.model.weight_img}, w4: {self.model.weight_ling}")
        # if (e + 1) % 5 == 0:  # In mỗi 5 epoch
        #     print(f"alpha: {self.model.alpha}, gamma: {self.model.gamma}")


        return l
        