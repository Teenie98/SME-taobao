import torch
import torch.nn as nn
from gen_model.base import BaseGenModel
from rec_model import device
from data import one_hot_feat, dense_feat, item_feat
import pickle


class MetaEmb(BaseGenModel):
    def __init__(self, args):
        super().__init__(args)
        # self.generated_emb_layer = nn.Sequential(
        #     nn.Linear(3 * args.embedding_size, 16),
        #     nn.LeakyReLU(),
        #     nn.Linear(16, args.embedding_size),
        #     nn.Tanh()
        # )
        self.generated_emb_layer = nn.Sequential(
            nn.Linear(4 * args.embedding_size, 16),
            nn.LeakyReLU(),
            nn.Linear(16, args.embedding_size),
            nn.Tanh()
        )


    def forward(self, rec_model, x_s, x_d):
        sparse_emb = [rec_model.embeddings[feat](x_s[:, idx + 1]) for idx, feat in enumerate(item_feat)]

        attr_emb = torch.cat(sparse_emb + [x_d], dim=1)
        output = self.generated_emb_layer(attr_emb)
        return output, 0

