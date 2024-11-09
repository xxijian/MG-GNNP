import datetime
import io
import math
from contextlib import redirect_stdout

import numpy as np
from torchsummary import summary

from tqdm import tqdm
from torch.nn import Module, Parameter
import torch.nn.functional as F
from loguru import logger
import torch
import torch.nn as nn
from aggregator import LocalAggregator, GlobalAggregator

class LastAttenion(Module):

    def __init__(self, hidden_size, heads, dot, l_p, last_k=3, use_attn_conv=False, area_func=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        self.last_k = last_k
        self.linear_zero = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, self.heads, bias=False)
        self.linear_four = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.linear_five = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.dropout = 0.1
        self.dot = dot
        self.l_p = l_p
        self.use_attn_conv = use_attn_conv
        self.ccattn = area_func
        self.last_layernorm = torch.nn.LayerNorm(hidden_size, eps=1e-8)
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            weight.data.normal_(std=0.1)

    def forward(self, ht1, hidden, mask):
        q0 = self.linear_zero(ht1).view(-1, ht1.size(1), self.hidden_size // self.heads)
        q1 = self.linear_one(hidden).view(-1, hidden.size(1), self.hidden_size // self.heads)
        q2 = self.linear_two(hidden).view(-1, hidden.size(1), self.hidden_size // self.heads)
        assert not torch.isnan(q0).any()
        assert not torch.isnan(q1).any()

        alpha = torch.sigmoid(torch.matmul(q0, q1.permute(0, 2, 1)))
        assert not torch.isnan(alpha).any()
        alpha = alpha.view(-1, q0.size(1) * self.heads, hidden.size(1)).permute(0, 2, 1)
        alpha = torch.softmax(2 * alpha, dim=1)
        assert not torch.isnan(alpha).any()
        if self.use_attn_conv == "True":
            m = torch.nn.LPPool1d(self.l_p, self.last_k, stride=self.last_k)
            alpha = m(alpha)
            alpha = torch.masked_fill(alpha, ~mask.bool().unsqueeze(-1), float('-inf'))
            alpha = torch.softmax(2 * alpha, dim=1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        a = torch.sum(
            (alpha.unsqueeze(-1) * q2.view(hidden.size(0), -1, self.heads, self.hidden_size // self.heads)).view(
                hidden.size(0), -1, self.hidden_size) * mask.view(mask.shape[0], -1, 1).float(), 1)
        return a, alpha   # a[batchsize,hiddensize] alpha[batchsize,item,]


class CombineGraph(Module):
    def __init__(self, opt, num_node, adj_all, num):
        super(CombineGraph, self).__init__()
        self.opt = opt

        self.batch_size = opt.batch_size
        self.num_node = num_node
        self.hidden_size = opt.hiddenSize
        self.dropout_local = opt.dropout_local
        self.dropout_global = opt.dropout_global
        self.hop = opt.n_iter
        self.sample_num = opt.n_sample
        self.adj_all = trans_to_cuda(torch.Tensor(adj_all)).long()  # 处理好后的adj num（weight）
        self.num = trans_to_cuda(torch.Tensor(num)).float()
        self.last_k = opt.last_k
        self.l_p = opt.l_p
        self.use_attn_conv = opt.use_attn_conv
        self.heads = opt.heads
        self.dot = opt.dot
        self.linear_q = nn.ModuleList()
        self.norm = opt.norm
        self.scale = opt.scale

        self.dataset = opt.dataset
        for i in range(self.last_k):
            self.linear_q.append(nn.Linear((i + 1) * self.hidden_size, self.hidden_size))
        self.mattn = LastAttenion(self.hidden_size, self.heads, self.dot, self.l_p, last_k=self.last_k,
                                  use_attn_conv=self.use_attn_conv)
        self.dropout = 0.2
        # Aggregator
        self.local_agg = LocalAggregator(self.hidden_size, self.opt.leaky_alpha, dropout=0.0)
        self.global_agg = []
        for i in range(self.hop): 
            if opt.activate == 'relu':
                agg = GlobalAggregator(self.hidden_size, opt.dropout_gcn, act=torch.relu)
            else:
                agg = GlobalAggregator(self.hidden_size, opt.dropout_gcn, act=torch.tanh)
            self.add_module('agg_gcn_{}'.format(i), agg)
            self.global_agg.append(agg)

        # Item representation & Position representation
        self.embedding = nn.Embedding(num_node, self.hidden_size)
        self.pos_embedding = nn.Embedding(200, self.hidden_size)

        # Parameters
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.hidden_size, self.hidden_size))
        self.w_2 = nn.Parameter(torch.Tensor(self.hidden_size, 1))
        self.glu1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.glu2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.linear_transform2 = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)

        self.leakyrelu = nn.LeakyReLU(opt.leaky_alpha)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def sample(self, target, n_sample):
        # neighbor = self.adj_all[target.view(-1)]
        # index = np.arange(neighbor.shape[1])
        # np.random.shuffle(index)
        # index = index[:n_sample]
        # return self.adj_all[target.view(-1)][:, index], self.num[target.view(-1)][:, index]
        return self.adj_all[target.view(-1)], self.num[target.view(-1)]


    def compute_scores(self, hidden, mask):    
        hts = [] 
        lengths = torch.sum(mask, dim=1)

        mask1 = mask.float().unsqueeze(-1)
        batch_size = hidden.shape[0]
        len = hidden.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        hs = torch.sum(hidden * mask1, -2) / torch.sum(mask1, 1)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask1
        select = torch.sum(beta * hidden, 1)     

        for i in range(self.last_k):
            hts.append(self.linear_q[i](torch.cat(
                [hidden[torch.arange(mask.size(0)).long(), torch.clamp(lengths - (j + 1), -1, 1000)] for j in
                 range(i + 1)], dim=-1)).unsqueeze(1))
        hts = torch.cat(hts, dim=1)   
        hts = hts.div(torch.norm(hts, p=2, dim=1, keepdim=True) + 1e-12)    
        hidden = hidden[:, :mask.size(1)]
        ht0 = hidden[torch.arange(mask.size(0)).long(), torch.sum(mask, 1) - 1]

        # ais[batchsize, hiddensize]
        ais, weight = self.mattn(hts, hidden, mask)
        a = self.linear_transform2(torch.cat((ais.squeeze(), ht0), 1))  

        b = self.embedding.weight[1:]      
        session = torch.mean(torch.stack((select, a)), dim=0)
        if self.norm:
            session = session.div(torch.norm(session, p=2, dim=1, keepdim=True) + 1e-12)
            b = b.div(torch.norm(b, p=2, dim=1, keepdim=True) + 1e-12)
        b = F.dropout(b, self.dropout, training=self.training)
        scores = torch.matmul(session, b.transpose(1, 0))
        if self.scale:
            scores = 16 * scores
        return scores


    def forward(self, inputs, adj, mask_item, item):
        batch_size = inputs.shape[0]  
        seqs_len = inputs.shape[1]  
        h = self.embedding(inputs)  


        h_local = self.local_agg(h, adj, mask_item)  

 
        item_neighbors = [inputs]  
        weight_neighbors = [] 
        support_size = seqs_len  

        for i in range(1, self.hop + 1): 

            item_sample_i, weight_sample_i = self.sample(item_neighbors[-1], self.sample_num)

            support_size *= self.sample_num
            item_neighbors.append(item_sample_i.view(batch_size, support_size))  
            weight_neighbors.append(weight_sample_i.view(batch_size, support_size))  

        entity_vectors = [self.embedding(i) for i in item_neighbors]
        weight_vectors = weight_neighbors

        session_info = []
        item_emb = self.embedding(item) * mask_item.float().unsqueeze(-1)

        sum_item_emb = torch.sum(item_emb, 1) / torch.sum(mask_item.float(), -1).unsqueeze(-1)



        sum_item_emb = sum_item_emb.unsqueeze(-2)
        for i in range(self.hop):
            session_info.append(sum_item_emb.repeat(1, entity_vectors[i].shape[1], 1))

        for n_hop in range(self.hop):
            entity_vectors_next_iter = []  
            shape = [batch_size, -1, self.sample_num, self.hidden_size] 


            for hop in range(self.hop - n_hop):
                aggregator = self.global_agg[n_hop]  
                vector = aggregator(self_vectors=entity_vectors[hop],
                                    neighbor_vector=entity_vectors[hop + 1].view(shape),
                                    masks=None,
                                    batch_size=batch_size,
                                    neighbor_weight=weight_vectors[hop].view(batch_size, -1, self.sample_num),
                                    extra_vector=session_info[hop])
                entity_vectors_next_iter.append(vector)  

            entity_vectors = entity_vectors_next_iter  

        h_global = entity_vectors[0].view(batch_size, seqs_len, self.hidden_size)  


        h_local = F.dropout(h_local, self.dropout_local, training=self.training)  
        h_global = F.dropout(h_global, self.dropout_global, training=self.training) 
        output = h_global+h_local

        return output


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, data):
    alias_inputs, adj, items, mask, targets, inputs = data
    alias_inputs = trans_to_cuda(alias_inputs).long()
    items = trans_to_cuda(items).long()
    adj = trans_to_cuda(adj).float()
    mask = trans_to_cuda(mask).long()
    inputs = trans_to_cuda(inputs).long()

    hidden = model(items, adj, mask, inputs)

    if model.norm:
        seq_shape = list(hidden.size())
        hidden = hidden.view(-1, model.hidden_size)
        norms = torch.norm(hidden, p=2, dim=1) 
        hidden = hidden.div(norms.unsqueeze(-1).expand_as(hidden))
        hidden = hidden.view(seq_shape)
    get = lambda index: hidden[index][alias_inputs[index]]   
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()]) 
    if model.norm:
        seq_shape = list(seq_hidden.size())
        seq_hidden = seq_hidden.view(-1, model.hidden_size)
        norms = torch.norm(seq_hidden, p=2, dim=1) 
        seq_hidden = seq_hidden.div(norms.unsqueeze(-1).expand_as(seq_hidden))
        seq_hidden = seq_hidden.view(seq_shape)

    scores = model.compute_scores(seq_hidden, mask)
    return targets, scores


def train_test(model, train_data, test_data):
    logger.debug('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=model.batch_size,
                                               shuffle=True, pin_memory=True)
    for data in tqdm(train_loader):
        model.optimizer.zero_grad()
        targets, scores = forward(model, data)
        targets = trans_to_cuda(targets).long()
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
    logger.debug('\tLoss:\t%.3f' % total_loss)
    model.scheduler.step()

    logger.debug('start predicting: ', datetime.datetime.now())
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=model.batch_size,
                                              shuffle=False, pin_memory=True)
    result = []
    hit10, hit20, mrr10, mrr20 = [], [], [], []
    for data in test_loader:
        targets, scores = forward(model, data)

        sub_scores20 = scores.topk(20)[1]
        sub_scores20 = trans_to_cpu(sub_scores20).detach().numpy()

        sub_scores10 = scores.topk(10)[1]
        sub_scores10 = trans_to_cpu(sub_scores10).detach().numpy()
        targets = targets.numpy()

        for score10, score20, target, mask in zip(sub_scores10, sub_scores20, targets, test_data.mask):

            hit10.append(np.isin(target - 1,score10))

            hit20.append(np.isin(target - 1, score20))

            if len(np.where(score10 == target - 1)[0]) == 0:
                mrr10.append(0)
            else:
                mrr10.append(1 / (np.where(score10 == target - 1)[0][0] + 1))

            if len(np.where(score20 == target - 1)[0]) == 0:
                mrr20.append(0)
            else:
                mrr20.append(1 / (np.where(score20 == target - 1)[0][0] + 1))


    result.append(np.mean(hit20) * 100)
    result.append(np.mean(mrr20) * 100)
    result.append(np.mean(hit10) * 100)
    result.append(np.mean(mrr10) * 100)

    return result
