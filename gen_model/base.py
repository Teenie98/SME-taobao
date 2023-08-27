import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import Dataset, num_words_dict, one_hot_feat
from rec_model import device


class BaseGenModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, rec_model, x_s, x_d):
        raise NotImplementedError

    def generate_train(self, rec_model, batch_size, lr, cold_lr, alpha):
        self.train()
        rec_model.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_func = nn.BCELoss(reduction='mean')

        for Da, Db in [['a', 'b'], ['c', 'd']]:
            train_loader_a = DataLoader(Dataset('../data/train_oneshot_' + Da), batch_size=batch_size, shuffle=False)
            train_loader_b = DataLoader(Dataset('../data/train_oneshot_' + Db), batch_size=batch_size, shuffle=False)

            tot_loss = 0.0
            tot_epoch = 0
            for (x_sparse_a, x_dense_a, y_a), (x_sparse_b, x_dense_b, y_b) in zip(train_loader_a, train_loader_b):
                x_sparse_a, x_dense_a, y_a = x_sparse_a.to(device), x_dense_a.to(device), y_a.float().to(device)
                x_sparse_b, x_dense_b, y_b = x_sparse_b.to(device), x_dense_b.to(device), y_b.float().to(device)

                # embs = rec_model.get_embs(x_a)
                one_hot_emb = [rec_model.embeddings[col](x_sparse_a[:, idx]) for idx, col in enumerate(one_hot_feat) if
                               col != 'item_id']
                item_id_emb = rec_model.embeddings['item_id'](x_sparse_a[:, 0])
                embs = torch.cat([item_id_emb] + one_hot_emb + [x_dense_a], dim=1)
                generate_emb, generate_idx = self(rec_model, x_sparse_a, x_dense_a)
                embs[generate_idx] = generate_emb
                pred_a = rec_model.forward_with_embs(embs)
                loss_a = loss_func(pred_a, y_a)

                grad_a = torch.autograd.grad(loss_a, generate_emb, retain_graph=True)
                generate_emb = generate_emb - cold_lr * grad_a[0]
                embs[generate_idx] = generate_emb
                pred_b = rec_model.forward_with_embs(embs)
                loss_b = loss_func(pred_b, y_b)

                loss = loss_a * alpha + loss_b * (1 - alpha)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tot_loss += loss.item()
                tot_epoch += 1
            print('generator train loss:{:.4f}'.format(tot_loss / tot_epoch))

    def init_id_embedding(self, rec_model):
        # 每20个数据里包含的item是相同的
        test_loader = DataLoader(Dataset('../data/test_oneshot_a'), batch_size=10, shuffle=False)

        self.eval()
        rec_model.eval()
        with torch.no_grad():
            for x_s, x_d, y in test_loader:
                x_s, x_d, y = x_s.to(device), x_d.to(device), y.float()
                generate_emb, generate_idx = self(rec_model, x_s[:1], x_d[:1])

                col = num_words_dict[generate_idx]
                idx = x_s[0, generate_idx]
                rec_model.embeddings[col].weight.data[idx].copy_(generate_emb.squeeze())
