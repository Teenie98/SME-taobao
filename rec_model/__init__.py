from rec_model.afm import AFM
from rec_model.deepfm import DeepFM
from rec_model.wideanddeep import WideAndDeep
from rec_model.ipnn import IPNN
from rec_model.opnn import OPNN
from rec_model.base import device, set_seed
import torch
import argparse

model_dict = {'afm': AFM, 'deepfm': DeepFM, 'wideanddeep': WideAndDeep, 'ipnn': IPNN, 'opnn': OPNN}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--embedding_size', type=int, default=16)
    parser.add_argument('--hidden_layer_size', type=int, default=64)
    parser.add_argument('--warm_up_batch_size', type=int, default=400)
    parser.add_argument('--warm_up_learning_rate', type=float, default=1e-3)
    parser.add_argument('--pretrain', type=int, default=1)
    parser.add_argument('--model', type=str, default='opnn', help="deepfm wideanddeep ipnn opnn afm")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    assert args.model in model_dict
    model = model_dict[args.model](args).to(device)

    if args.pretrain == 2:
        print('cross domain dataset: training model {}...'.format(args.model))
        model.pre_train(args.batch_size, args.learning_rate, '../data/dataset_sys1')
        torch.save(model.state_dict(), "./cross_domain_p/{}_parameter.pkl".format(args.model))
    elif args.pretrain == 1:
        print('training model {}...'.format(args.model))
        model.pre_train(args.batch_size, args.learning_rate)
        torch.save(model.state_dict(), "./save_p/{}_parameter.pkl".format(args.model))
    else:
        print('load model {}...'.format(args.model))
        model.load_state_dict(torch.load("./save_p/{}_parameter.pkl".format(args.model)))

    test_auc, test_logloss = model.predict()
    print('test auc: {:.4f}, logloss: {:.4f}'.format(test_auc, test_logloss))

    print('warm up training...')
    model.warm_up_train(args.warm_up_batch_size, args.warm_up_learning_rate, 'item_id')

