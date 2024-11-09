import io
import time
import argparse
import pickle
from contextlib import redirect_stdout

from model import *
from model import CombineGraph
from utils import *
from utils import handle_adj
from loguru import logger
from torchsummary import summary



def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='tmall', help='diginetica/Nowplaying/tmall')
parser.add_argument('--hiddenSize', type=int, default=256)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--activate', type=str, default='relu')
parser.add_argument('--n_sample_all', type=int, default=12)
parser.add_argument('--n_sample', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=3)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay.')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay.')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty ')
parser.add_argument('--n_iter', type=int, default=1)                                    # [1, 2]
parser.add_argument('--dropout_gcn', type=float, default=0, help='Dropout rate.')       # [0, 0.2, 0.4, 0.6, 0.8]
parser.add_argument('--dropout_local', type=float, default=0, help='Dropout rate.')     # [0, 0.5]
parser.add_argument('--dropout_global', type=float, default=0.4, help='Dropout rate.')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion')
parser.add_argument('--leaky_alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=3)
# 融合AM添加的
parser.add_argument('--norm', default=True, help='adapt NISER, l2 norm over item and session embedding')
parser.add_argument('--scale', default=True, help='scaling factor sigma')
parser.add_argument('--train_flag', type=str, default="True")
# parser.add_argument('--PATH', default='../checkpoint/gowalla.pt', help='checkpoint path')
parser.add_argument('--dot', default=True, action='store_true')
parser.add_argument('--last_k', type=int, default=7)
parser.add_argument('--l_p', type=int, default=4)
parser.add_argument('--use_attn_conv', type=str, default="True")
parser.add_argument('--heads', type=int, default=2, help='number of attention heads')

opt = parser.parse_args()


def main():
    init_seed(2020)
    logger.add("./trainLog/"+opt.dataset + "_{time}.log")

    if opt.dataset == 'diginetica':
        num_node = 43098
        opt.n_iter = 2
        opt.dropout_gcn = 0.2
        opt.dropout_local = 0.3
        opt.heads = 4
    elif opt.dataset == 'Nowplaying':
        num_node = 60417
        opt.n_iter = 1
        opt.dropout_gcn = 0.0
        opt.dropout_local = 0.0
        opt.heads = 16
    elif opt.dataset == 'tmall':
        num_node = 40728
        opt.n_iter = 1
        opt.dropout_gcn = 0.6
        opt.dropout_local = 0.5
        opt.heads = 8
    else:
        num_node = 310
    train_data = pickle.load(open('datasets/' + opt.dataset + '/train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('datasets/' + opt.dataset + '/test.txt', 'rb'))

    adj = pickle.load(open('datasets/' + opt.dataset + '/adj_' + str(opt.n_sample_all) + '.pkl', 'rb'))
    num = pickle.load(open('datasets/' + opt.dataset + '/num_' + str(opt.n_sample_all) + '.pkl', 'rb'))
    train_data = Data(train_data)
    test_data = Data(test_data)
    adj, num = handle_adj(adj, num_node, opt.n_sample_all, num)   # adj、num的shape是 node*n_sample_all
    model = trans_to_cuda(CombineGraph(opt, num_node, adj, num))

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params}')
    logger.debug(opt)
    start = time.time()
    best_result = [0, 0, 0, 0]
    best_epoch20 = [0, 0]
    best_epoch10 = [0, 0]
    bad_counter = 0

    for epoch in range(opt.epoch):
        logger.debug('-------------------------------------------------------')
        logger.debug('epoch: %d' % epoch)
        hit20, mrr20, hit10, mrr10 = train_test(model, train_data, test_data)
        flag = 0
        if hit20 >= best_result[0]:
            best_result[0] = hit20
            best_epoch20[0] = epoch
            flag = 1
        if mrr20 >= best_result[1]:
            best_result[1] = mrr20
            best_epoch20[1] = epoch
            flag = 1
        if hit10 >= best_result[2]:
            best_result[2] = hit10
            best_epoch10[0] = epoch
        if mrr10 >= best_result[3]:
            best_result[3] = mrr10
            best_epoch10[1] = epoch


        logger.debug('Current Result:')
        logger.debug('\tRecall@20:\t%.4f\tMRR@20:\t%.4f\tRecall@10:\t%.4f\tMRR@10:\t%.4f' % (hit20, mrr20, hit10, mrr10))
        logger.debug('Best Result:')
        logger.debug('\tRecall@20:\t%.4f\tMRR@20:\t%.4f\tEpoch:\t%d,\t%d' % (
            best_result[0], best_result[1], best_epoch20[0], best_epoch20[1]))
        logger.debug('\tRecall@10:\t%.4f\tMRR@10:\t%.4f\tEpoch:\t%d,\t%d' % (
            best_result[2], best_result[3], best_epoch10[0], best_epoch10[1]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    logger.debug('-------------------------------------------------------')
    end = time.time()
    logger.debug("Run time: %f s" % (end - start))




if __name__ == '__main__':
    main()
