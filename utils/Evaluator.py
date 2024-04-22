from utils.IntentDataset import IntentDataset
from utils.TaskSampler import MultiLabTaskSampler, UniformTaskSampler
from utils.tools import makeEvalExamples
from utils.printHelper import *
from utils.Logger import logger
from utils.commonVar import *
import logging
import torch
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans, AgglomerativeClustering
from utils.tools import *
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from utils.processor import handle_method,update_method
from transformers import InputExample, AdamW, get_linear_schedule_with_warmup, \
    BertForMaskedLM, RobertaForMaskedLM, BertConfig, BertTokenizer, RobertaConfig, \
    RobertaTokenizer, AlbertForMaskedLM, AlbertTokenizer, AlbertConfig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
##
# @brief  base class of evaluator
class EvaluatorBase():
    def __init__(self):
        self.roundN = 4
        pass

    def round(self, floatNum):
        return round(floatNum, self.roundN)

    def evaluate(self):
        raise NotImplementedError("train() is not implemented.")

class prompt_Evaluator(EvaluatorBase): #不针对backbone，而是后续分类器的few shot
    def __init__(self,  dataset: IntentDataset):
        super(prompt_Evaluator, self).__init__()

        self.dataset = dataset

    def evaluate(self, args ,model, tokenizer,device,plot_cm=True,save=True,mode = None,eval_or_test = None,head='source'):
        model.eval()
        processor_labeled = handle_method(dataset=self.dataset,tok = tokenizer)
        labTensorData = processor_labeled.get_input_features(label = self.dataset.getLabID())
        labdataloader = DataLoader(labTensorData, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True) #根据刚才的tensordataset生成dataloader
        with torch.no_grad():

            X_feature = torch.empty((0,model.backbone.config.hidden_size)).to(device)
            Y =torch.empty((0))

            for batch in tqdm(labdataloader, desc="Extracting representation"):
                label,_,_,_,_ = batch
                ##print(label.size())
                if head == 'target':
                    inputs = self.generate_default_inputs(batch ,model,device,label = True,source_or_target = 'target')
                else:
                    inputs = self.generate_default_inputs(batch ,model,device,label = True,source_or_target = 'source')
                feature , logits = model(inputs)
                #print(Y)
                #print(Y.size())
                X_feature = torch.cat((X_feature, feature))
                Y = torch.cat((Y,label))
            
            #print(X_feature.size())

            X_feature = X_feature.cpu().numpy()

            # k-means clustering
            km = KMeans(n_clusters = self.dataset.getLabNum()).fit(X_feature)
            
            y_pred = km.labels_
            y_true = Y.cpu()
            y_true = np.array(y_true,dtype='int')
            #y_true.dtype = 'int'
            #print(y_true)
            #print(y_pred)

            results = clustering_score(y_true, y_pred)
            print('results',results)
            
            # confusion matrix
            if plot_cm:
                ind, _ = hungray_aligment(y_true, y_pred)
                map_ = {i[0]:i[1] for i in ind}
                y_pred = np.array([map_[idx] for idx in y_pred])

                cm = confusion_matrix(y_true,y_pred)   
                print('confusion matrix',cm)
            if save:
                save_results(args,results,mode,eval_or_test)
        
        return results['ACC']
    
    def generate_default_inputs(self, batch ,model,device,source_or_target,label = False):

        batch = tuple(temp.to(device) for temp in batch)

        if label == True:
            label,block_flags,input_ids,token_type_ids,attention_mask = batch 
        else :
            block_flags,input_ids,token_type_ids,attention_mask = batch 

        bz = input_ids.shape[0]

        if source_or_target == 'source':
            raw_embeds = model.backbone.bert.embeddings.word_embeddings(input_ids)

            replace_embeds = model.prompt_embeddings_source(torch.LongTensor(list(range(model.prompt_length))).cuda())
            replace_embeds = replace_embeds.unsqueeze(0) # [batch_size, prompt_length, embed_size]

            replace_embeds = model.lstm_head_source(replace_embeds)[0]  # [batch_size, seq_len, 2 * hidden_dim]
            if model.prompt_length == 1:
                replace_embeds = model.mlp_head_source(replace_embeds)
            else:
                replace_embeds = model.mlp_head_source(replace_embeds).squeeze()

            blocked_indices = (block_flags == 1).nonzero().reshape((bz, model.prompt_length, 2))[:, :, 1]

            for bidx in range(bz):
                for i in range(blocked_indices.shape[1]):
                    raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]
        else:
            raw_embeds = model.backbone.bert.embeddings.word_embeddings(input_ids)

            replace_embeds = model.prompt_embeddings_target(torch.LongTensor(list(range(model.prompt_length))).cuda())
            replace_embeds = replace_embeds.unsqueeze(0) # [batch_size, prompt_length, embed_size]

            replace_embeds = model.lstm_head_target(replace_embeds)[0]  # [batch_size, seq_len, 2 * hidden_dim]
            if model.prompt_length == 1:
                replace_embeds = model.mlp_head_target(replace_embeds)
            else:
                replace_embeds = model.mlp_head_target(replace_embeds).squeeze()

            blocked_indices = (block_flags == 1).nonzero().reshape((bz, model.prompt_length, 2))[:, :, 1]

            for bidx in range(bz):
                for i in range(blocked_indices.shape[1]):
                    raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]

        inputs = {'inputs_embeds': raw_embeds.to(device), 'attention_mask': attention_mask.to(device)}

        inputs['token_type_ids'] = token_type_ids.to(device)

        return inputs

class Pre_Evaluator(EvaluatorBase): #不针对backbone，而是后续分类器的few shot
    def __init__(self,  dataset: IntentDataset):
        super(Pre_Evaluator, self).__init__()

        self.dataset = dataset

    def evaluate(self, args ,model, tokenizer,device,plot_cm=True,save=True,mode = 'MLM',eval_or_test=None):
        model.eval()
        X = self.dataset.getTokList()
        Y = self.dataset.getLabID()
        
        X, Y = makeEvalExamples(X, Y, tokenizer)  #只是做了一下padding的操作
        dataloader = DataLoader(X, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        #print(Y)
        Y = torch.tensor(Y, dtype=torch.long)  
        with torch.no_grad():

            X_feature = torch.empty((0,model.backbone.config.hidden_size)).to(device)

            for batch in tqdm(dataloader, desc="Extracting representation"):
                ids, types, masks = batch
                X = {'input_ids':ids.to(device),
                     'token_type_ids':types.to(device),
                     'attention_mask':masks.to(device)}
                feature,_ = model(X)

                X_feature = torch.cat((X_feature, feature))
            
            print(X_feature.size())


            X_feature = X_feature.cpu().numpy()

            # k-means clustering
            km = KMeans(n_clusters = self.dataset.getLabNum()).fit(X_feature)
            
            y_pred = km.labels_
            y_true = Y.cpu().numpy()

            results = clustering_score(y_true, y_pred)
            print('results',results)
            
            # confusion matrix
            if plot_cm:
                ind, _ = hungray_aligment(y_true, y_pred)
                map_ = {i[0]:i[1] for i in ind}
                y_pred = np.array([map_[idx] for idx in y_pred])

                cm = confusion_matrix(y_true,y_pred)   
                print('confusion matrix',cm)
            if save:
                save_results(args,results,mode,eval_or_test)
        
        return results['ACC']

class FewShotEvaluator(EvaluatorBase): #不针对backbone，而是后续分类器的few shot
    def __init__(self, evalParam, taskParam, dataset, device):
        super(FewShotEvaluator, self).__init__()
        self.way   = taskParam['way']
        self.shot  = taskParam['shot']
        self.query = taskParam['query']
        self.device = device

        self.dataset = dataset
        self.evalTaskNum = evalParam['evalTaskNum']

        self.taskSampler = UniformTaskSampler(self.dataset, self.way, self.shot, self.query)

    def generate_default_inputs(self,batch,model,source_or_target,tokenizer,label = False):


        batch = tuple(temp.to(self.device) for temp in batch)

        if label == True:
            label,block_flags,input_ids,token_type_ids,attention_mask = batch 
        else :
            block_flags,input_ids,token_type_ids,attention_mask = batch 

        mask_ids, mask_lb = mask_tokens(input_ids.cpu(), tokenizer,prompt_mask = block_flags.cpu())

        #mask_ids.to(self.device)
        #block_flags.to(self.device)

        bz = input_ids.shape[0]
        if source_or_target == 'source':
            raw_embeds = model.backbone.bert.embeddings.word_embeddings(mask_ids.cuda())
            model.lstm_head_source.flatten_parameters()
            replace_embeds = model.prompt_embeddings_source(torch.LongTensor(list(range(model.prompt_length))).cuda())
            replace_embeds = replace_embeds.unsqueeze(0) # [batch_size, prompt_length, embed_size]

            replace_embeds = model.lstm_head_source(replace_embeds)[0]  # [batch_size, seq_len, 2 * hidden_dim]
            if model.prompt_length == 1:
                replace_embeds = model.mlp_head_source(replace_embeds)
            else:
                replace_embeds = model.mlp_head_source(replace_embeds).squeeze()

            blocked_indices = (block_flags == 1).nonzero().reshape((bz, model.prompt_length, 2))[:, :, 1]

            for bidx in range(bz):
                for i in range(blocked_indices.shape[1]):
                    raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]
        else:
            model.lstm_head_target.flatten_parameters()
            raw_embeds = model.backbone.bert.embeddings.word_embeddings(mask_ids.cuda())

            replace_embeds = model.prompt_embeddings_target(torch.LongTensor(list(range(model.prompt_length))).cuda())
            replace_embeds = replace_embeds.unsqueeze(0) # [batch_size, prompt_length, embed_size]

            replace_embeds = model.lstm_head_target(replace_embeds)[0]  # [batch_size, seq_len, 2 * hidden_dim]
            if model.prompt_length == 1:
                replace_embeds = model.mlp_head_target(replace_embeds)
            else:
                replace_embeds = model.mlp_head_target(replace_embeds).squeeze()

            blocked_indices = (block_flags == 1).nonzero().reshape((bz, model.prompt_length, 2))[:, :, 1]

            for bidx in range(bz):
                for i in range(blocked_indices.shape[1]):
                    raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]


        inputs = {'inputs_embeds': raw_embeds.to(self.device), 'attention_mask': attention_mask.to(self.device)}

        inputs['token_type_ids'] = token_type_ids.to(self.device)

        return inputs,mask_lb

    def evaluate(self, model, tokenizer, logLevel='INFO'):
        
        performList = []   # acc, pre, rec, fsc
  
        for task in range(self.evalTaskNum):
            # sample a task
            task = self.taskSampler.sampleOneTask()

            # collect data
            supportX = task[META_TASK_SHOT_TOKEN]
            queryX = task[META_TASK_QUERY_TOKEN]
            supportY = task[META_TASK_SHOT_LAB]
            queryY = task[META_TASK_QUERY_LAB]
            supportD = task[META_TASK_SHOT_DOM]
            queryD = task[META_TASK_QUERY_DOM]

            processor_labeled = update_method(utt=supportX,label=supportY,dom=supportD,tok=tokenizer)
            labTensorData = processor_labeled.get_input_features(label = supportY)
            labdataloader = DataLoader(labTensorData, batch_size=100, shuffle=True, num_workers=4, pin_memory=True) #根据刚才的tensordataset生成dataloader
            labdataloader = iter(labdataloader)
            labTensorData = labdataloader.next()
            processor_unlabeled = update_method(utt=queryX,label=queryY,dom=queryD,tok=tokenizer)
            unlabTensorData = processor_unlabeled.get_input_features(label = queryY)
            unlabdataloader = DataLoader(unlabTensorData, batch_size=100, shuffle=True, num_workers=4, pin_memory=True) #根据刚才的tensordataset生成dataloader
            unlabdataloader = iter(unlabdataloader)
            unlabTensorData = unlabdataloader.next()

            cur_model = model

            # Prepare optimizer and schedule (linear warmup and decay)
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in cur_model.backbone.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay': 0.0},
                {'params': [p for n, p in cur_model.backbone.named_parameters() if any(nd in n for nd in no_decay)],'weight_decay': 0.0}
            ]

            embedding_parameters = [
                {'params': [p for p in cur_model.lstm_head_target.parameters()]},
                {'params': [p for p in cur_model.mlp_head_target.parameters()]},
                {'params': [p for p in cur_model.prompt_embeddings_target.parameters()]},
                {'params': [p for p in cur_model.lstm_head_source.parameters()]},
                {'params': [p for p in cur_model.mlp_head_source.parameters()]},
                {'params': [p for p in cur_model.prompt_embeddings_source.parameters()]}
            ]
            t_total = 1 
            optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, eps=1e-8)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)
            #optimizer.to(device)
            #scheduler.to(device)

            embedding_optimizer = AdamW(embedding_parameters, lr=1e-5, eps=1e-8)
            embedding_scheduler = get_linear_schedule_with_warmup(embedding_optimizer, num_warmup_steps=0, num_training_steps=t_total)

            cur_model.train()

            optimizer.zero_grad()
            embedding_optimizer.zero_grad()

            inputs_un,un_mlm_label = self.generate_default_inputs(labTensorData ,cur_model,label = True,source_or_target = 'target',tokenizer=tokenizer)
                # forward
            loss_mlmt,feature_t,logits_t = cur_model.mlmForward(inputs_un, un_mlm_label.to(self.device))

            loss_unlab =  loss_mlmt

            loss_unlab.backward()

            torch.nn.utils.clip_grad_norm_(cur_model.parameters(), 1.0)
            #optimizer.step()
            #scheduler.step()
            embedding_optimizer.step()
            embedding_scheduler.step()

            inputs_un,un_mlm_label = self.generate_default_inputs(labTensorData ,cur_model,label = True,source_or_target = 'target',tokenizer=tokenizer)
                # forward
            loss_mlmt,feature_support,logits_t = cur_model.mlmForward(inputs_un, un_mlm_label.to(self.device))

            inputs_un,un_mlm_label = self.generate_default_inputs(unlabTensorData ,cur_model,label = True,source_or_target = 'target',tokenizer=tokenizer)
                # forward
            loss_mlmt,feature_query,logits_t = cur_model.mlmForward(inputs_un, un_mlm_label.to(self.device))

            support_features = feature_support.cpu()

            query_features = feature_query.cpu()

            clf = LogisticRegression(penalty='l2',
                                     random_state=0,
                                     C=1.0,
                                     solver='lbfgs',
                                     max_iter=1000)
            # fit and predict
            clf.fit(support_features.detach().numpy(), supportY)

            queryPrediction = clf.predict(query_features.detach().numpy())

            del cur_model, optimizer, embedding_optimizer,scheduler,embedding_scheduler

            # calculate acc
            acc = accuracy_score(queryY, queryPrediction)   # acc
            
            performDetail = precision_recall_fscore_support(queryY, queryPrediction, average='macro', warn_for=tuple())

            performList.append([acc, performDetail[0], performDetail[1], performDetail[2]])
        
        # performance mean and std
        performMean = np.mean(np.stack(performList, 0), 0)
        performStd  = np.std(np.stack(performList, 0), 0)

        if logLevel == 'DEBUG':
            itemList = ["acc", "pre", "rec", "fsc"]
            logger.debug("Evaluate statistics: ")
            printMeanStd(performMean, performStd, itemList, debugLevel=logging.DEBUG)
        else:
            itemList = ["acc", "pre", "rec", "fsc"]
            logger.info("Evaluate statistics: ")
            printMeanStd(performMean, performStd, itemList, debugLevel=logging.INFO)
        # acc, pre, rec, F1
        return performMean[0], performMean[1], performMean[2], performMean[3]
    
    def normalize(self, x):
        norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
        out = x.div(norm)
        return out

