import torch
import torch.nn as nn
from rec_model.base import BaseRecModel, num_words_dict


class WideAndDeep(BaseRecModel):
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

        # wide
        self.wide_layer = nn.Linear(sum_emb_size, 1)
        self.act_func = nn.Sigmoid()

    def forward_with_embs(self, x_sparse, x_dense, item_id_emb):
        one_hot_emb = [self.embeddings[col](x_sparse[:, idx]) for idx, col in enumerate(self.one_hot_feat) if
                       col != 'item_id']
        # wide
        wide_emb = torch.cat([item_id_emb] + one_hot_emb + [x_dense], dim=1)
        wide_output = self.wide_layer(wide_emb)

        # dnn
        dnn_emb = torch.cat([item_id_emb] + one_hot_emb + [x_dense], dim=1)
        dnn_output = self.dnn_layers(dnn_emb)

        output = wide_output + dnn_output
        output = self.act_func(output)
        return output

