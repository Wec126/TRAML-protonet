# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
loss function script.
"""
import math

import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.nn.loss.loss import LossBase
import mindspore as ms
import numpy as np
import msadapter.pytorch as torch


def word_to_vector(word, glove_vectors):
    return glove_vectors.get(word)


def cosine_similarity(vec1, vec2):
    dot_product = ops.matmul(vec1, vec2.T)
    norm_vec1 = ops.sqrt(ops.reduce_sum(vec1 ** 2, axis=1))
    norm_vec2 = ops.sqrt(ops.reduce_sum(vec2 ** 2))
    return dot_product / (norm_vec1[:, None] * norm_vec2)


def compute_similarity(tensor1, tensor2):
    dot_product = np.dot(tensor1, tensor2)
    norm_vector1 = np.linalg.norm(tensor1)
    norm_vector2 = np.linalg.norm(tensor2)
    return dot_product / (norm_vector1 * norm_vector2)

def get_index(category,classes_word):
    # 获得当前类别在prototypes中的下标
    index = 0
    for sub_classes in classes_word.keys():
        if sub_classes == category:
            return index
        else:
            index = index + 1
    return index

def get_key_by_value(dict_data, classes):
    class_word = {}
    for class_idx in classes.tolist():
        for key, val in dict_data.items():
            if val == class_idx:
                class_word[str(class_idx)] = key
    return class_word
def cos_similarity(vec1, vec2):
    vec1 = vec1.detach()
    vec2 = vec2.detach()
    dot_product = ms.ops.matmul(vec1, vec2.unsqueeze(1))
    matrix_norm = ms.ops.norm(vec1)
    vector_norm = ms.ops.norm(vec2)
    cosine_similarity = dot_product / (matrix_norm * vector_norm)
    return cosine_similarity
def m_function(glove_vectors,a,b,current_category_idx,classes,query_sample,prototypes):
    vector = {}
    for category in classes:
        category = category.item()
        vector[str(category)]=word_to_vector(str(category),glove_vectors)
    bottom_num = torch.zeros((1,len(classes)))
    for query_index in range(len(query_sample)):
        output = np.zeros((1,len(classes)))
        for category in classes:
            category = category.item()
            result = torch.zeros((1,))
            if category != current_category_idx:
                m = compute_similarity(vector[str(current_category_idx)],vector[str(category)])
                m = torch.tensor(a.item()*m+b.item())
                dist = cos_similarity(query_sample[query_index],prototypes[get_index(category, classes)])
                result = result + (dist + m).apply_(math.exp)
                output[0,get_index(category,classes)] = result.item()
        bottom_num = torch.cat((bottom_num,torch.from_numpy(output)))
    return bottom_num[1:]

class PrototypicalLoss(LossBase):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''
    def __init__(self,glove_vectors,n_support, n_query, n_class, is_train=True):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support
        self.n_query = n_query
        self.eq = ops.Equal()
        self.sum = ops.ReduceSum(True)
        self.log_softmax = nn.LogSoftmax(1)
        self.gather = ops.GatherD()
        self.squeeze = ops.Squeeze()
        self.max = ops.Argmax(2)
        self.cast = ops.Cast()
        self.stack = ops.Stack()
        self.reshape = ops.Reshape()
        self.topk = ops.TopK(sorted=True)
        self.expendDims = ops.ExpandDims()
        self.broadcastTo = ops.BroadcastTo((100, 20, 64))
        self.pow = ops.Pow()
        self.sum = ops.ReduceSum()
        self.zeros = Tensor(np.zeros(200), ms.float32)
        self.ones = Tensor(np.ones(200), ms.float32)
        self.print = ops.Print()
        self.unique = ops.Unique()
        self.samples_count = 10
        self.select = ops.Select()
        self.target_inds = Tensor(list(range(0, n_class)), ms.int32)
        self.is_train = is_train
        self.glove_vectors=glove_vectors
        self.a = ms.ops.rand(1)
        self.b = -ms.ops.rand(1)
        # self.acc_val = 0

    def construct(self,inp, target, classes):
        """
        loss construct
        """
        n_classes = len(classes)
        n_query = self.n_query
        support_idxs = ()
        query_idxs = ()

        for ind, _ in enumerate(classes):
            class_c = classes[ind]
            _, a = self.topk(self.cast(self.eq(target, class_c), ms.float32), self.n_support + self.n_query)
            support_idx = self.squeeze(a[:self.n_support])
            support_idxs += (support_idx,)
            query_idx = a[self.n_support:]
            query_idxs += (query_idx,)


        prototypes = ()
        for idx_list in support_idxs:
            prototypes += (inp[idx_list].mean(0),)
        prototypes = self.stack(prototypes)

        query_idxs = self.stack(query_idxs).view(-1)
        query_samples = inp[query_idxs]
        target_inds = self.target_inds.view(n_classes, 1, 1)
        target_inds = ops.BroadcastTo((n_classes, n_query, 1))(target_inds)  # to int64

        # FIXME when torch.unique will be available on cuda too
        classes = torch.unique(target)
        n_classes = len(classes)
        n_query = target.eq(classes[0].item()).sum().item() - self.n_support
        support_idxs = list(map(support_idxs, classes))
        prototypes = torch.stack([inp[idx_list].mean(0) for idx_list in support_idxs])
        query_idxs = torch.stack(list(map(lambda c: target.eq(c).nonzero()[self.n_support:], classes))).view(-1)
        query_samples = inp[query_idxs]
        top_num = torch.zeros((1, n_classes))
        bottom_num = torch.zeros((1, n_classes))
        index = 0
        for category_idx in classes.tolist():
            if index == 0:
                query_sample_idx = query_idxs[:n_query]
            else:
                query_sample_idx = query_idxs[index * n_query:(index + 1) * n_query]
            query_sample = inp[query_sample_idx]
            m = m_function(self.glove_vectors, self.a, self.b, category_idx, classes, query_sample, prototypes)
            dist = cosine_similarity(query_sample, prototypes[index]).repeat(1, n_classes)
            top_num = torch.cat((top_num, dist.apply_(math.exp)), dim=0)
            bottom_num = torch.cat((bottom_num, (dist.apply_(math.exp) + m)), dim=0)
            index = index + 1
        log_p = ((top_num[1:, :] / bottom_num[1:, :]).apply_(math.exp)).apply_(math.log)
        loss_val = log_p.view(n_classes, n_query, -1)
        target_inds = torch.arange(0, n_classes)
        target_inds = target_inds.view(n_classes, 1, 1)
        target_inds = target_inds.expand(n_classes, n_query, 1).long()
        _, y_hat = loss_val.max(2)
        acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()
        loss_val = loss_val.gather(2, target_inds).squeeze().view(-1).mean()
        if self.is_train:
            return loss_val
        return acc_val, loss_val

    def supp_idxs(self, target, c):
        return self.squeeze(self.nonZero(self.eq(target, c))[:self.n_support])

    def nonZero(self, inpbool):
        out = []
        for _, inp in enumerate(inpbool):
            if inp:
                out.append(inp)
        return Tensor(out, ms.int32)

    def acc(self):
        return self.acc_val


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.shape[0]
    m = y.shape[0]
    d = x.shape[1]

    expendDims = ops.ExpandDims()
    broadcastTo = ops.BroadcastTo((n, m, d))
    pow_op = ops.Pow()
    reducesum = ops.ReduceSum()

    x = broadcastTo(expendDims(x, 1))
    y = broadcastTo(expendDims(y, 0))
    return reducesum(pow_op(x-y, 2), 2)
