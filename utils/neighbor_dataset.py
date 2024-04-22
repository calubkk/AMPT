import torch
import numpy as np
from torch.utils.data import Dataset

class NeighborsDataset(Dataset):
    def __init__(self, dataset, indices, num_neighbors=None):
        super(NeighborsDataset, self).__init__()
        # transform = dataset.transform
        
        # if isinstance(transform, dict):
        #     self.anchor_transform = transform['standard']
        #     self.neighbor_transform = transform['augment']
        # else:
        #     self.anchor_transform = transform
        #     self.neighbor_transform = transform
       
        # dataset.transform = None
        # self.tok= AutoTokenizer.from_pretrained(tokenizer)
        self.dataset = dataset
        self.indices = indices # Nearest neighbor indices (np.array  [len(dataset) x k])
        if num_neighbors is not None:
            self.indices = self.indices[:, :num_neighbors+1]
        assert(self.indices.shape[0] == len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = {}
        anchor = list(self.dataset.__getitem__(index))
        
        neighbor_index = np.random.choice(self.indices[index], 1)[0]  #这是随机挑选一个邻居的意思哇
        neighbor = self.dataset.__getitem__(neighbor_index)

        # anchor['image'] = self.anchor_transform(anchor['image'])
        # neighbor['image'] = self.neighbor_transform(neighbor['image'])

        # output['anchor'] = anchor['image']
        # output['neighbor'] = neighbor['image'] 
        # anchor[0] = shuffle_tokens(anchor[0], self.tok)
        # neighbor[0] = shuffle_tokens(neighbor[0], self.tok)
        output['anchor'] = anchor[1:]  #除了label，其他的三样东西
        output['neighbor'] = neighbor[1:] #除了label，其他的三样东西
        output['possible_neighbors'] = torch.from_numpy(self.indices[index]) #注意他还把所有可能的邻居都择出来了
        # output['target'] = anchor['target']
        output['target'] = anchor[0]  #label
        output['index'] = index  #anchor的label
        return output
        #小结，就是每个样本，随机挑了一个邻居，然后其所有可能的邻居也进行了保存