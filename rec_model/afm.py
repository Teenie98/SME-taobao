import torch
import torch.nn as nn
from rec_model.base import BaseRecModel, num_words_dict


class AFM(BaseRecModel):
    def __init__(self, args):
        super().__init__(args)

        sum_emb_size = len(self.one_hot_feat) * self.args.embedding_size + len(self.dense_feat)
        self.linear_layer = nn.Linear(sum_emb_size, 1)

        self.attention_layer = nn.Linear(self.args.embedding_size, 16)
        self.attention_h = nn.Linear(16, 1, bias=False)
        self.fc_layer = nn.Linear(self.args.embedding_size, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward_with_embs(self, x_sparse, x_dense, item_id_emb):
        one_hot_emb = [self.embeddings[col](x_sparse[:, idx]) for idx, col in enumerate(self.one_hot_feat) if col != 'item_id']

        linear_emb = torch.cat([item_id_emb] + one_hot_emb + [x_dense], dim=1)
        linear_output = self.linear_layer(linear_emb)
        afm_emb = torch.stack([item_id_emb] + one_hot_emb, dim=1)

        row, col = list(), list()
        for i in range(len(self.one_hot_feat) - 1):
            for j in range(i + 1, len(self.one_hot_feat)):
                row.append(i), col.append(j)

        afm_emb1, afm_emb2 = afm_emb[:, row], afm_emb[:, col]
        inner_product = afm_emb1 * afm_emb2
        attention_score = self.attention_layer(inner_product)
        attention_score = self.relu(attention_score)
        attention_score = self.attention_h(attention_score)
        attention_score = self.softmax(attention_score)
        attention_output = torch.sum(attention_score * inner_product, dim=1)
        attention_output = self.fc_layer(attention_output)

        output = self.sigmoid(linear_output + attention_output)

        return output
