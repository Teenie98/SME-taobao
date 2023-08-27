import time
import torch
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from data import num_words_dict, one_hot_feat, dense_feat, Dataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def set_seed(seed, cudnn=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # note: the below slows down the code but makes it reproducible
    if (seed is not None) and cudnn:
        torch.backends.cudnn.deterministic = True


class BaseRecModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.one_hot_feat = one_hot_feat
        self.dense_feat = dense_feat
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(num_words_dict[col], self.args.embedding_size) for col in self.one_hot_feat
        })

        for key, val in self.embeddings.items():
            nn.init.normal_(val.weight, mean=0, std=1.0/self.args.embedding_size)

    def forward_with_embs(self, x_sparse, x_dense, item_id_emb):
        raise NotImplementedError

    def forward(self, x_sparse, x_dense):
        item_id_emb = self.embeddings['item_id'](x_sparse[:, 0])
        output = self.forward_with_embs(x_sparse, x_dense, item_id_emb)
        return output

    def predict(self, batch_size=1024):
        test_loader = DataLoader(Dataset('../data/test_test'), batch_size=batch_size)
        self.eval()
        with torch.no_grad():
            pred, target = [], []
            for x_sparse, x_dense, y in test_loader:
                x_sparse, x_dense, y= x_sparse.int().to(device), x_dense.float().to(device) ,y.float()
                pred_y = self(x_sparse, x_dense)
                pred.extend(pred_y.cpu().numpy().reshape(-1).tolist())
                target.extend(y.numpy().reshape(-1).tolist())
        auc = roc_auc_score(target, pred)
        logloss = log_loss(target, pred)
        return auc, logloss

    def pre_train(self, batch_size, lr, filepath='../data/big_train_main', epochs=1):
        train_loader = DataLoader(Dataset(filepath), batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr)
        loss_func = nn.BCELoss(reduction='mean')
        tot_loss = 0.0
        tot_epoch = 0

        print('start pre-train...')
        self.train()
        start_time = time.time()

        for i in range(epochs):
            for x_sparse, x_dense, y in train_loader:
                x_sparse, x_dense, y = x_sparse.int().to(device), x_dense.float().to(device), y.float().to(device)
                pred_y = self(x_sparse, x_dense)
                loss = loss_func(pred_y, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tot_loss += loss.item()
                tot_epoch += 1
            end_time = time.time()
            print('epoch {:2d}/{:2d} pre-train loss:{:.4f}, cost {:.2f}s'.format(
                i + 1, epochs, tot_loss / tot_epoch, end_time - start_time))
            start_time = end_time

    def warm_up_train(self, batch_size, lr, learnable_col):
        self.train()
        optimizer = torch.optim.Adam(self.embeddings[learnable_col].parameters(), lr)
        loss_func = nn.BCELoss(reduction='mean')
        tot_loss = 0.0
        tot_epoch = 0

        for idx in ['a', 'b', 'c']:
            train_loader = DataLoader(Dataset('../data/test_oneshot_' + idx), batch_size=batch_size, shuffle=True)
            for x_sparse, x_dense, y in train_loader:
                x_sparse, x_dense, y = x_sparse.int().to(device), x_dense.float().to(device), y.float().to(device)
                pred_y = self(x_sparse, x_dense)
                loss = loss_func(pred_y, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tot_loss += loss.item()
                tot_epoch += 1
            print('warm-up {} train loss:{:.4f}'.format(idx, tot_loss / tot_epoch))
            test_auc, test_logloss = self.predict()
            print('test auc: {:.4f}, logloss: {:.4f}'.format(test_auc, test_logloss))
