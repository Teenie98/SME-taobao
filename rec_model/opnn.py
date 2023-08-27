import torch
import torch.nn as nn
from rec_model.base import BaseRecModel, num_words_dict


class OPNN(BaseRecModel):
    def __init__(self, args):
        super().__init__(args)

        self.cross_attr_num = len(self.one_hot_feat) * (len(self.one_hot_feat) - 1) // 2
        sum_emb_size = self.args.embedding_size * len(self.one_hot_feat) + len(self.dense_feat) + self.cross_attr_num

        self.kernel_shape = (self.args.embedding_size, self.cross_attr_num, self.args.embedding_size)
        self.kernel = nn.Parameter(torch.randn(self.kernel_shape))
        nn.init.xavier_uniform_(self.kernel.data)

        self.dnn_layers = nn.Sequential(
            nn.Linear(sum_emb_size, self.args.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(self.args.hidden_layer_size, self.args.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(self.args.hidden_layer_size, self.args.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(self.args.hidden_layer_size, 1)
        )

        self.act_func = nn.Sigmoid()

    def forward_with_embs(self, x_sparse, x_dense, item_id_emb):
        one_hot_emb = [self.embeddings[col](x_sparse[:, idx]) for idx, col in enumerate(self.one_hot_feat) if
                       col != 'item_id']
        pnn_emb = torch.stack([item_id_emb] + one_hot_emb, dim=1)
        row, col = list(), list()
        for i in range(len(self.one_hot_feat) - 1):
            for j in range(i + 1, len(self.one_hot_feat)):
                row.append(i), col.append(j)

        pnn_emb1, pnn_emb2 = pnn_emb[:, row], pnn_emb[:, col]
        p1 = torch.sum(pnn_emb1.unsqueeze(1) * self.kernel, dim=-1).permute(0, 2, 1)
        pnn_emb = torch.sum(p1 * pnn_emb2, dim=-1)

        dnn_emb = torch.cat([item_id_emb] + one_hot_emb + [x_dense], dim=1)

        dnn_output = self.dnn_layers(torch.cat([pnn_emb, dnn_emb], dim=1))

        output = self.act_func(dnn_output)
        return output

