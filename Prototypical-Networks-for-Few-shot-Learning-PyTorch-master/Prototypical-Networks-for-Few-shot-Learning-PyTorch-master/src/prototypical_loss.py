# coding=utf-8
import torch
import math
import numpy as np
def word_to_vector(word, glove_vectors):
    return glove_vectors.get(word)
def cosine_similarity(vec1, vec2):
    vec1 = vec1.detach()
    vec2 = vec2.detach()
    dot_product = torch.matmul(vec1, vec2.unsqueeze(1))
    matrix_norm = torch.norm(vec1, dim=1)
    vector_norm = torch.norm(vec2)
    cosine_similarity = dot_product / (matrix_norm.unsqueeze(1) * vector_norm)
    return cosine_similarity
def compute_similarity(tensor1,tensor2):
    dot_product = np.dot(tensor1, tensor2)

    norm_vector1 = np.linalg.norm(tensor1)
    norm_vector2 = np.linalg.norm(tensor1)

    cosine_similarity = dot_product / (norm_vector1 * norm_vector2)
    return cosine_similarity

def cos_similarity(vec1, vec2):
    vec1 = vec1.detach()
    vec2 = vec2.detach()
    dot_product = torch.matmul(vec1, vec2.unsqueeze(1))
    matrix_norm = torch.norm(vec1)
    vector_norm = torch.norm(vec2)
    cosine_similarity = dot_product / (matrix_norm * vector_norm)
    return cosine_similarity
def m_function(glove_vectors,a,b,current_category_idx,current_category,class_word,query_sample,prototypes):

    vector = {}
    for index,category in class_word.items():
        vector[index]=word_to_vector(str(index),glove_vectors)
    bottom_num = torch.zeros((1,60))
    for query_index in range(len(query_sample)):
        output = np.zeros((1,60))
        for index,category in class_word.items():
            result = torch.zeros((1,))
            if index != current_category_idx:
                m = compute_similarity(vector[str(current_category_idx)],vector[str(index)])
                m = torch.tensor(a.item()*m+b.item())
                dist = cos_similarity(query_sample[query_index],prototypes[get_index(index, class_word)])
                result = result + (dist + m).apply_(math.exp)
                output[0,get_index(index,class_word)] = result.item()
        bottom_num = torch.cat((bottom_num,torch.from_numpy(output)))
    return bottom_num[1:]
def get_index(category,classes_word):
    index = 0
    for sub_classes in classes_word.keys():
        if sub_classes == category:
            return index
        else:
            index = index + 1
    return index

def euclidean_dist(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
def get_key_by_value(dict_data, classes):
    class_word = {}
    for class_idx in classes.tolist():
        for key, val in dict_data.items():
            if val == class_idx:
                class_word[str(class_idx)] = key
    return class_word

def prototypical_loss(glove_vectors,model,input, target, n_support,idx_classes):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')
    def supp_idxs(c):
        # FIXME when torch will support where as np
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)
    # FIXME when torch.unique will be available on cuda too
    classes = torch.unique(target_cpu)
    class_word = get_key_by_value(idx_classes,classes)
    n_classes = len(classes)
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support
    support_idxs = list(map(supp_idxs, classes))
    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)
    query_samples = input.to('cpu')[query_idxs]
    top_num = torch.zeros((1,n_query))
    bottom_num = torch.zeros((1,n_query))
    index = 0
    for category_idx in classes.tolist():
        if index == 0:
            query_sample_idx = query_idxs[:n_query]
        else:
            query_sample_idx = query_idxs[index*n_query:(index+1)*n_query]
        category_index = index
        query_sample = input.to('cpu')[query_sample_idx]
        category = class_word[str(category_idx)]
        m = m_function(glove_vectors,model.a,model.b,category_idx,category,class_word,query_sample,prototypes)
        dist = cosine_similarity(query_sample,prototypes[index]).repeat(1,n_query)
        top_num = torch.cat((top_num,dist.apply_(math.exp)),dim=0)
        bottom_num = torch.cat((bottom_num,(dist.apply_(math.exp)+m)),dim=0)
        index = index +1
    log_p = ((top_num[1:,:]/bottom_num[1:,:]).apply_(math.exp)).apply_(math.log)
    loss_val = -log_p.view(n_classes,n_query,-1)
    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()
    _, y_hat = loss_val.max(2)
    acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()
    loss_val = -loss_val.gather(2, target_inds).squeeze().view(-1).mean()
    return loss_val,  acc_val
