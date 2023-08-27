import torch
import torch.nn as nn
from rec_model.base import BaseRecModel, num_words_dict


class DeepFM(BaseRecModel):
    def __init__(self, args):
        super().__init__(args)

        sum_emb_size = len(self.one_hot_feat) * self.args.embedding_size + len(self.dense_feat)
        self.dnn_layers = nn.Sequential(
            nn.Linear(sum_emb_size, self.args.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(self.args.hidden_layer_size, self.args.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(self.args.hidden_layer_size, self.args.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(self.args.hidden_layer_size, 1)
        )

        # fm
        self.fm_layer = nn.Linear(sum_emb_size, 1)
        self.act_func = nn.Sigmoid()

    def forward_with_embs(self, x_sparse, x_dense, item_id_emb):
        one_hot_emb = [self.embeddings[col](x_sparse[:, idx]) for idx, col in enumerate(self.one_hot_feat) if
                       col != 'item_id']
        # fm
        fm_emb_1 = torch.cat([item_id_emb] + one_hot_emb + [x_dense], dim=1)
        fm_emb_2 = torch.stack([item_id_emb] + one_hot_emb, dim=1)

        fm_1st = self.fm_layer(fm_emb_1)

        square_of_sum = torch.pow(torch.sum(fm_emb_2, dim=1), 2)
        sum_of_square = torch.sum(torch.pow(fm_emb_2, 2), dim=1)
        cross_term = square_of_sum - sum_of_square
        fm_2nd = 0.5 * torch.sum(cross_term, dim=1, keepdim=True)

        # dnn
        dnn_emb = torch.cat([item_id_emb] + one_hot_emb + [x_dense], dim=1)

        dnn_output = self.dnn_layers(dnn_emb)

        output = fm_1st + fm_2nd + dnn_output
        output = self.act_func(output)
        return output
