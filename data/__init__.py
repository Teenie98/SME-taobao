import pickle
import numpy as np
from torch.utils.data import Dataset as _Dataset
import torch.nn as nn

num_words_dict = {
    'user_id': 692163,
    'item_id': 1432,
    'pid': 2,
    'cate_id': 139,
    'campaign_id': 1188,
    'customer': 971,
    'brand': 517,
    'cms_segid': 98,
    'cms_group_id': 14,
    'final_gender_code': 3,
    'age_level': 8,
    'pvalue_level': 4,
    'shopping_level': 4,
    'occupation': 3,
    'new_user_class_level ' : 5
}

one_hot_feat = ['item_id', 'cate_id', 'campaign_id', 'customer', 'brand', 'user_id', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level', 'pvalue_level',
               'shopping_level', 'occupation', 'new_user_class_level ', 'pid']
dense_feat = ['price']
item_feat = ['cate_id', 'campaign_id', 'customer', 'brand', 'price']

def read_pkl(file):
    with open(file, 'rb') as f:
        t = pickle.load(f)
    return t

def get_data(data):
    data_y = data['clk'].values.reshape(-1, 1)
    data_sparse = data[one_hot_feat].to_numpy()
    data_dense = data[dense_feat].to_numpy()
    return data_y, data_sparse, data_dense

class Dataset(_Dataset):
    def __init__(self, name):
        super().__init__()
        data = read_pkl(name + '.pkl')
        self.y, self.x_sparse, self.x_dense = get_data(data)

    def __getitem__(self, idx):
        return self.x_sparse[idx], self.x_dense[idx], self.y[idx]

    def __len__(self):
        return len(self.x_sparse)


