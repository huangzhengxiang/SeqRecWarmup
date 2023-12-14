from .base import BaseModel
from .bert_modules.bert import BERT

import torch
import torch.nn as nn


class BERTModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.bert = BERT(args)
        self.out = nn.Linear(self.bert.hidden, args.num_items + 1)
        self.mask_token = None
        self.bert_backtrack_len = args.bert_backtrack_len

    @classmethod
    def code(cls):
        return 'bert'

    def forward(self, x, backtrack=False):
        if backtrack and not (self.mask_token is None):
            # x: [B, H]
            # assume no masks presented
            with torch.no_grad():
                if x[0,-1]==self.mask_token:
                    # if during prediction phase
                    x = torch.cat(
                        [self.backtrack(
                            torch.cat([torch.zeros_like(x[:,-1]).reshape(-1,1),x[:,:-1]],dim=-1)
                            )[:,1:],
                         x[:,-1].reshape(-1,1)],
                        dim=-1)
                else:
                    # if just padding
                    x = self.backtrack(x)
        x = self.bert(x)
        return self.out(x)

    @torch.no_grad()
    def backtrack_once(self,x):
        device = x.device
        # backtrack the items here!
        indices = (0 == x).sum(dim=-1) - 1
        valid_indices = indices != -1
        x[torch.arange(len(valid_indices)).to(device=device)[valid_indices], indices[valid_indices]] = self.mask_token
        predict = torch.argmax(self.out(self.bert(x)),dim=-1)
        x[torch.arange(len(valid_indices)).to(device=device)[valid_indices], indices[valid_indices]] = predict[valid_indices, indices[valid_indices]]
        return x

    @torch.no_grad()
    def backtrack(self, x):
        if self.bert_backtrack_len is None:
            # 补全
            while (0==x).flatten().sum() > 0:
                x = self.backtrack_once(x)
        else:
            # 补指定数目的
            for _ in range(min(self.bert_backtrack_len,x.shape[-1])):
                x = self.backtrack_once(x)
        return x
        
    def set_mask_token(self, token):
        self.mask_token = token