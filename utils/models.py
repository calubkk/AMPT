#coding=utf-8
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM
from utils.commonVar import *

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from utils.contrastive import SupConLoss
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler

class BERT(nn.Module):
    def __init__(self, config,device):
        super(BERT, self).__init__()
        self.LMName = config['LMName']
        self.clsNum = config['clsNumber']
        self.device = device
        pretrain_model_name = "./saved_models/MLMtrain_bertforclnn"
        '''
        try:
            self.backbone = AutoModelForMaskedLM.from_pretrained(pretrain_model_name)
            print(f'BERT model loaded from {pretrain_model_name}.')
        except:
            self.backbone = AutoModelForMaskedLM.from_pretrained(self.LMName)
            print(f'Original BERT model loaded.')
        '''
        self.backbone = AutoModelForMaskedLM.from_pretrained(self.LMName)
        print(f'Original BERT model loaded.')
        self.linearClsfier = nn.Linear(768, self.clsNum)
        self.backbone.to(device)
        self.linearClsfier.to(device)
        self.dropout = nn.Dropout(0.1) 

    def loss_ce(self, logits, Y):
        loss = nn.CrossEntropyLoss()
        output = loss(logits, Y)
        return output

    def loss_mse(self, logits, Y):
        loss = nn.MSELoss()
        output = loss(torch.sigmoid(logits).squeeze(), Y)
        return output

    def loss_kl(self, logits, label):
        # KL-div loss
        probs = F.log_softmax(logits, dim=1)
        # label_probs = F.log_softmax(label, dim=1)
        loss = F.kl_div(probs, label, reduction='batchmean')
        return loss
    
    def forward(self, X):
        # BERT forward
        outputs = self.backbone(**X, output_hidden_states=True)

        # extract [CLS] for utterance representation
        CLSEmbedding_ = outputs.hidden_states[-1][:,0]

        # linear classifier
        CLSEmbedding = self.dropout(CLSEmbedding_)
        logits = self.linearClsfier(CLSEmbedding)

        return CLSEmbedding_ , logits
    
    def mlmForward(self, X, Y):
        # BERT forward
        outputs = self.backbone(**X, labels=Y)

        return outputs.loss

    def save(self, path):
        self.backbone.save_pretrained(path)
    
    def predict(self, X):
        # calculate word embedding
        # BERT forward
        x_embedding = self.backbone(**X, output_hidden_states=True).hidden_states[-1]

        # extract [CLS] for utterance representation
        Embedding = x_embedding[:,0]

        return Embedding


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(768, 256),  # 输入特征数为784，输出为256
            nn.LeakyReLU(0.2),  # 进行非线性映射
            nn.Linear(256, 256),  # 进行一个线性映射
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 也是一个激活函数，二分类问题中，
            # sigmoid可以班实数映射到【0,1】，作为概率值，
            # 多分类用softmax函数
        )
 
    def forward(self, x):
        x = self.dis(x)
        return x


class ContinuousPrompt(torch.nn.Module):
    def __init__(self, config,device,tokenizer,save_path = None):
        super(ContinuousPrompt, self).__init__()
        self.tokenizer = tokenizer
        self.embed_size = config['feat_dim']
        self.hidden_size = config['feat_dim']
        #self.model_name_or_path = config['model_name_or_path']
        self.LMName = config['LMName']
        self.clsNum = config['clsNumber']
        self.dropout = nn.Dropout(0.1) 
        self.prompt_length = 4 # The pattern_id is supposed to indicate the number of continuous prompt tokens.
        pretrain_model_name = save_path
        try:
            self.backbone = AutoModelForMaskedLM.from_pretrained(pretrain_model_name)
            print(f'BERT in ContinuousPrompt loaded from {pretrain_model_name}.')
        except:
            self.backbone = AutoModelForMaskedLM.from_pretrained(self.LMName)
            print(f'Original BERT in ContinuousPrompt loaded.')

        self.prompt_embeddings_source = torch.nn.Embedding(self.prompt_length, self.embed_size)
        self.prompt_embeddings_target = torch.nn.Embedding(self.prompt_length, self.embed_size)
        self.lstm_head_source = torch.nn.LSTM(input_size=self.hidden_size,
                                        hidden_size=self.hidden_size,
                                        num_layers=2,
                                        bidirectional=True,
                                        batch_first=True)
        self.lstm_head_target = torch.nn.LSTM(input_size=self.hidden_size,
                                        hidden_size=self.hidden_size,
                                        num_layers=2,
                                        bidirectional=True,
                                        batch_first=True)
        self.mlp_head_source = nn.Sequential(nn.Linear(2 * self.hidden_size, self.hidden_size),
                                        nn.ReLU(),
                                        nn.Linear(self.hidden_size, self.hidden_size))
        self.mlp_head_target = nn.Sequential(nn.Linear(2 * self.hidden_size, self.hidden_size),
                                        nn.ReLU(),
                                        nn.Linear(self.hidden_size, self.hidden_size))
        self.linearClsfier = nn.Linear(768, self.clsNum)
        
        self.backbone.to(device)
        self.prompt_embeddings_source.to(device)
        self.lstm_head_source.to(device)
        self.mlp_head_source.to(device)
        self.prompt_embeddings_target.to(device)
        self.lstm_head_target.to(device)
        self.mlp_head_target.to(device)
        self.linearClsfier.to(device)
        

    def forward(self, X, labels=None):

        outputs = self.backbone(**X, output_hidden_states=True)
        # extract [CLS] for utterance representation
        CLSEmbedding_ = outputs.hidden_states[-1][:,0]

        # linear classifier
        CLSEmbedding = self.dropout(CLSEmbedding_)
        logits = self.linearClsfier(CLSEmbedding)

        return CLSEmbedding_,logits
    
    def mlmForward(self, X, Y):
        # BERT forward
        outputs = self.backbone(**X, labels=Y, output_hidden_states=True)
        CLSEmbedding_ = outputs.hidden_states[-1][:,0]

        # linear classifier
        CLSEmbedding = self.dropout(CLSEmbedding_)
        logits = self.linearClsfier(CLSEmbedding)

        return outputs.loss,CLSEmbedding,logits

    
    def loss_ce(self, logits, Y):
        loss = nn.CrossEntropyLoss()
        output = loss(logits, Y)
        return output
    
    def loss_mse(self, logits, Y):
        loss = nn.MSELoss()
        output = loss(torch.sigmoid(logits).squeeze(), Y)
        return output

    def loss_kl(self, logits, label):
        # KL-div loss
        probs = F.log_softmax(logits, dim=1)
        # label_probs = F.log_softmax(label, dim=1)
        loss = F.kl_div(probs, label, reduction='batchmean')
        return loss
    
    def save(self, path,logger) -> None:
        logger.info("Saving models.")

        self.backbone.save_pretrained(path)
        #self.tokenizer.save_pretrained(path)
        #self._save_config(path)

        state = {
            "prompt_embeddings_source": self.prompt_embeddings_source.state_dict(),
            "lstm_head_source": self.lstm_head_source.state_dict(),
            "mlp_head_source": self.mlp_head_source.state_dict(),
            "prompt_embeddings_target": self.prompt_embeddings_target.state_dict(),
            "lstm_head_target": self.lstm_head_target.state_dict(),
            "mlp_head_target": self.mlp_head_target.state_dict(),
        }

        save_path_file = os.path.join(path, "embeddings.pth")
        torch.save(state, save_path_file)
    
    def loaded(self, path):
        """Load a pretrained wrapper from a given path."""

        save_path_file = os.path.join(path, "embeddings.pth")
        data = torch.load(save_path_file)
        self.prompt_embeddings_target.load_state_dict(data["prompt_embeddings_target"])
        self.prompt_embeddings_source.load_state_dict(data["prompt_embeddings_source"])
        assert ("mlp_head_target" in data)
        self.lstm_head_target.load_state_dict(data["lstm_head_target"])
        self.mlp_head_target.load_state_dict(data["mlp_head_target"])
        self.lstm_head_source.load_state_dict(data["lstm_head_source"])
        self.mlp_head_source.load_state_dict(data["mlp_head_source"])
        self.backbone.load_pretrained(path)


        return 

    



