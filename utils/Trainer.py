from utils.IntentDataset import IntentDataset
from utils.Evaluator import EvaluatorBase
from utils.Logger import logger
from utils.commonVar import *
from utils.tools import mask_tokens, makeTrainExamples ,makefewshotExamples
from utils.TaskSampler import MultiLabTaskSampler, UniformTaskSampler
from utils.models import discriminator
from scipy.optimize import linear_sum_assignment
import torch.nn as nn
import time
import torch
from torch.utils.data import DataLoader
import numpy as np
import copy
from sklearn.metrics import accuracy_score, r2_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm_notebook, trange, tqdm
from sklearn.cluster import KMeans
from torch.autograd import Variable
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from utils.processor import handle_method
from transformers import InputExample, AdamW, get_linear_schedule_with_warmup, \
    BertForMaskedLM, RobertaForMaskedLM, BertConfig, BertTokenizer, RobertaConfig, \
    RobertaTokenizer, AlbertForMaskedLM, AlbertTokenizer, AlbertConfig
from copy import deepcopy

class TrainerBase():
    def __init__(self):
        self.finished=False
        self.bestModelStateDict = None
        self.roundN = 4
        pass

    def round(self, floatNum):
        return round(floatNum, self.roundN)

    def train(self):
        raise NotImplementedError("train() is not implemented.")

    def getBestModelStateDict(self):
        return self.bestModelStateDict

class metaphase(TrainerBase):
    def __init__(self,
                 trainingParam:dict,
                 device,
                 dataset:IntentDataset,
                 unlabeled:IntentDataset,
                 promptEvaluator:EvaluatorBase,
                 promptester:EvaluatorBase):
        super(metaphase, self).__init__()
        self.epoch       = trainingParam['epoch']
        self.batch_size  = trainingParam['batch']
        self.validation  = trainingParam['validation']
        self.patience    = trainingParam['patience']
        self.tensorboard = trainingParam['tensorboard']
        self.lambda_adv  = trainingParam['lambda adv']

        self.dataset       = dataset
        self.unlabeled     = unlabeled
        #self.optimizer     = optimizer
        self.promptEvaluator = promptEvaluator
        self.promptester = promptester
        self.device = device
        self.centroids = None

        if self.tensorboard:
            self.writer = SummaryWriter()
    
    def generate_default_inputs(self,batch,model,source_or_target,tokenizer,label = False):

        batch = tuple(temp.to(self.device) for temp in batch)

        if label == True:
            label,block_flags,input_ids,token_type_ids,attention_mask = batch 
        else :
            block_flags,input_ids,token_type_ids,attention_mask = batch 

        mask_ids, mask_lb = mask_tokens(input_ids.cpu(), tokenizer,special_tokens_mask = block_flags.cpu())

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

    def train(self, model, tokenizer,args):
        self.processor_labeled = handle_method(dataset=self.dataset,tok = tokenizer)
        self.processor_unlabeled = handle_method(dataset=self.unlabeled,tok = tokenizer)
        self.bestModelStateDict = copy.deepcopy(model.state_dict())
        durationOverallTrain = 0.0
        durationOverallVal = 0.0
        valBestAcc = -1
        accumulateStep = 0
        #=========定义discriminater相关的================
        self.discriminator = discriminator()
        self.discriminator.to(self.device)
        self.dis_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0003)
        self.criterion = nn.BCELoss() 
        labTensorData = self.processor_labeled.get_input_features(label = self.dataset.getLabID())
        labdataloader = DataLoader(labTensorData, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True) #根据刚才的tensordataset生成dataloader
        unlabTensorData = self.processor_unlabeled.get_input_features()
        unlabdataloader = DataLoader(unlabTensorData, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True) #根据刚才的tensordataset生成dataloader
        unlabelediter = iter(unlabdataloader)
        # evaluate before training  在训练之前针对这个valdata和test_data进行一些测试
        #valacc= self.valEvaluator.evaluate(args,model, tokenizer, self.device, save=False) 
        #teacc = self.testEvaluator.evaluate(args,model, tokenizer, self.device)
        #logger.info('---- Before training ----')
        #logger.info("ValAcc %f,valdomain %s", valacc ,args.valDomain)
        #logger.info("TestAcc %f,testdomain %s", teacc, args.testDomain)

        #cur_model = model.module if hasattr(model, 'module') else model

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.backbone.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay': 0.0},
            {'params': [p for n, p in model.backbone.named_parameters() if any(nd in n for nd in no_decay)],'weight_decay': 0.0}
        ]

        embedding_parameters = [
            {'params': [p for p in model.lstm_head_target.parameters()]},
            {'params': [p for p in model.mlp_head_target.parameters()]},
            {'params': [p for p in model.prompt_embeddings_target.parameters()]},
            {'params': [p for p in model.lstm_head_source.parameters()]},
            {'params': [p for p in model.mlp_head_source.parameters()]},
            {'params': [p for p in model.prompt_embeddings_source.parameters()]}
        ]
        t_total = len(labdataloader) // 1 * self.epoch
        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)
        #optimizer.to(device)
        #scheduler.to(device)


        embedding_optimizer = AdamW(embedding_parameters, lr=1e-5, eps=1e-8)
        embedding_scheduler = get_linear_schedule_with_warmup(embedding_optimizer, num_warmup_steps=0, num_training_steps=t_total)
        #embedding_optimizer.to(device)
        #embedding_scheduler.to(device)

        for epoch in range(self.epoch):  # an epoch means all sampled tasks are done
            model.train()
            batchTrAccSum     = 0.0
            batchTrLossG = 0.0
            batchTrLossD = 0.0
            timeEpochStart    = time.time()

            for batch in labdataloader:
                sum_gradients=[]
                loss_G = 0 
                loss_D = 0
                self.discriminator.train()
                model.train()
                for param in self.discriminator.parameters():
                    param.requires_grad = False
                optimizer.zero_grad()
                embedding_optimizer.zero_grad()
                self.dis_optimizer.zero_grad()

                fast_model = deepcopy(model)
                #fast_model.to(self.device)
                fast_model.train()

                #fast_cur_model =  fast_model.module if hasattr(fast_model, 'module') else  fast_model
                

                fast_optimizer_grouped_parameters = [
                    {'params': [p for n, p in fast_model.backbone.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay': 0.0},
                    {'params': [p for n, p in fast_model.backbone.named_parameters() if any(nd in n for nd in no_decay)],'weight_decay': 0.0}
                ]
                fast_embedding_parameters = [
                    {'params': [p for p in fast_model.lstm_head_target.parameters()]},
                    {'params': [p for p in fast_model.mlp_head_target.parameters()]},
                    {'params': [p for p in fast_model.prompt_embeddings_target.parameters()]},
                    {'params': [p for p in fast_model.lstm_head_source.parameters()]},
                    {'params': [p for p in fast_model.mlp_head_source.parameters()]},
                    {'params': [p for p in fast_model.prompt_embeddings_source.parameters()]}
                ]
                fast_t_total = len(labdataloader) 
                fast_optimizer = AdamW(fast_optimizer_grouped_parameters, lr=1e-5, eps=1e-8)
                fast_scheduler = get_linear_schedule_with_warmup(fast_optimizer, num_warmup_steps=0, num_training_steps=fast_t_total)
                fast_embedding_optimizer = AdamW(fast_embedding_parameters, lr=1e-5, eps=1e-8)
                fast_embedding_scheduler = get_linear_schedule_with_warmup(fast_embedding_optimizer, num_warmup_steps=0, num_training_steps=fast_t_total)
                fast_optimizer.zero_grad()
                fast_embedding_optimizer.zero_grad()

                try:
                    unlabatch = unlabelediter.next()
                except StopIteration:
                    unlabelediter = iter(unlabdataloader)
                    unlabatch = unlabelediter.next()
                
                inputs_un,un_mlm_label = self.generate_default_inputs(unlabatch ,fast_model,label = False,source_or_target = 'target',tokenizer=tokenizer)
                # forward
                loss_mlmt,feature_t,logits_t = fast_model.mlmForward(inputs_un, un_mlm_label.to(self.device))

                loss_mlmt.backward()
                fast_optimizer.step()
                fast_scheduler.step()
                fast_embedding_optimizer.step()
                fast_embedding_scheduler.step()
                fast_optimizer.zero_grad()
                fast_embedding_optimizer.zero_grad()
                

                # task data
                inputs,mlm_label = self.generate_default_inputs(batch ,fast_model,label = True,source_or_target = 'source',tokenizer=tokenizer)
                # forward
                loss_mlms,feature_s,logits_s = fast_model.mlmForward(inputs, mlm_label.to(self.device))

                Y_s,_,_,_,_ = batch
                # loss
                
                lossSP = fast_model.loss_ce(logits_s, Y_s.to(self.device))

                loss_s = 0.01 * lossSP + 0.01 * loss_mlms 

                loss_G = loss_G + loss_s
                
                loss_s.backward()
                
                
                inputs_un,un_mlm_label = self.generate_default_inputs(unlabatch ,fast_model,label = False,source_or_target = 'target',tokenizer=tokenizer)
                # forward
                loss_mlmt,feature_t,logits_t = fast_model.mlmForward(inputs_un, un_mlm_label.to(self.device))


                dis_out = self.discriminator(feature_t)
                dis_out=dis_out.squeeze(1)
                loss_adv=self.criterion(dis_out,Variable(torch.FloatTensor(dis_out.data.size()).fill_(0)).to(self.device))
                loss_adv = self.lambda_adv * loss_adv

                loss_unlab =  0.001 * loss_adv + 0.01*loss_mlmt

                loss_G = loss_G +loss_adv

                loss_unlab.backward()


                # bring back requires_grad
                for param in self.discriminator.parameters():
                    param.requires_grad = True

                # train with source
                feats_source = feature_s.detach()
                dis_out = self.discriminator(feats_source)
                dis_out=dis_out.squeeze(1)
                loss_d=self.criterion(dis_out,Variable(torch.FloatTensor(dis_out.data.size()).fill_(0)).to(self.device))

                loss_D = loss_d + loss_D

                loss_d.backward()


                # train with target
                feats_target = feature_t.detach()
                dis_out = self.discriminator(feats_target)
                dis_out=dis_out.squeeze(1)
                loss_d=self.criterion(dis_out,Variable(torch.FloatTensor(dis_out.data.size()).fill_(1)).to(self.device))

                loss_D = loss_d + loss_D

                loss_d.backward()

                for i, params in enumerate(fast_model.parameters()):
                    sum_gradients.append(deepcopy(params.grad))
                for i, params in enumerate(model.parameters()):
                    params.grad = sum_gradients[i]

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                self.dis_optimizer.step()
                embedding_optimizer.step()
                embedding_scheduler.step()

                # calculate train acc
                YTensor = Y_s.cpu()
                logits = logits_s.detach().clone()
                if torch.cuda.is_available():
                    logits = logits.cpu()
                logits = logits.numpy()
                predictResult = np.argmax(logits, 1)
                acc = accuracy_score(YTensor, predictResult)

                del fast_model, fast_optimizer, fast_embedding_optimizer,fast_scheduler,fast_embedding_scheduler
                torch.cuda.empty_cache()

                # accumulate statistics
                batchTrAccSum += acc
                batchTrLossG += loss_G.item()
                batchTrLossD += loss_D.item()


            # current epoch training done, collect data
            durationTrain         = self.round(time.time() - timeEpochStart)
            durationOverallTrain += durationTrain
            batchTrAccAvrg        = self.round(batchTrAccSum/len(labdataloader))
            batchTrLossG    = batchTrLossG/len(labdataloader)
            batchTrLossD    = batchTrLossD/len(labdataloader)

            # display current epoch's info
            logger.info("---- epoch: %d/%d, train_time %f ----", epoch, self.epoch, durationTrain)
            logger.info("GLoss %f, DLoss %f, TrainAcc %f", batchTrLossG, batchTrLossD, batchTrAccAvrg)
            valAcc  = self.promptEvaluator.evaluate(args, model, tokenizer, self.device,save=False,mode='meta-train',eval_or_test='eval',head='target')
            teAcc  = self.promptester.evaluate(args, model, tokenizer, self.device,save=False,mode='meta-train',eval_or_test='test',head='source')
            logger.info("ValAcc %f,valdomain %s", valAcc,args.valDomain)
            logger.info("TestAcc %f", teAcc)
            
            if (valAcc >= valBestAcc):   # better validation result
                print("[INFO] Find a better model. Val acc: %f -> %f"%(valBestAcc, valAcc))
                valBestAcc = valAcc
                accumulateStep = 0

                # cache current model, used for evaluation later
                self.bestModelStateDict = copy.deepcopy(model.state_dict())
            else:
               accumulateStep += 1
               if accumulateStep > self.patience/2:
                   print('[INFO] accumulateStep: ', accumulateStep)
                   if accumulateStep == self.patience:  # early stop
                       logger.info('Early stop.')
                       logger.debug("Overall training time %f", durationOverallTrain)
                       logger.debug("Overall validation time %f", durationOverallVal)
                       logger.debug("best_val_acc: %f", valBestAcc)
                       break
            '''
            if self.tensorboard:
                self.writer.add_scalar('train loss', batchTrLossG+self.lambda_mlm*batchTrLossD, global_step=epoch)
                self.writer.add_scalar('val acc', valAcc, global_step=epoch)
                self.writer.add_scalar('test acc', teAcc, global_step=epoch)
            '''
        '''
        valAcc,_,_,_ = self.valEvaluator.evaluate(model, tokenizer)
        teAcc,_,_,_  = self.testEvaluator.evaluate(model, tokenizer)
        logger.info("ValAcc %f,valdomain %s", valAcc,args.valDomain)
        logger.info("TestAcc %f,testdomain %s", teAcc,args.testDomain)
        logger.info("best_val_acc: %f", valBestAcc)
        '''

class SpTrainer(TrainerBase):
    def __init__(self,
                 trainingParam:dict,
                 optimizer,
                 device,
                 dataset:IntentDataset,
                 valEvaluator:EvaluatorBase,
                 testEvaluator:EvaluatorBase):
        super(SpTrainer, self).__init__()
        self.epoch       = trainingParam['epoch']
        self.batch_size  = trainingParam['batch']
        self.mlm         = trainingParam['mlm']
        self.validation  = trainingParam['validation']
        self.tensorboard = trainingParam['tensorboard']
        self.patience    = trainingParam['patience']
        self.device = device
        self.dataset       = dataset
        self.optimizer     = optimizer
        self.testEvaluator = testEvaluator
        self.valEvaluate = valEvaluator

    def train(self, model, tokenizer, args):
        valBestAcc = -1
        durationOverallTrain = 0.0
        TensorData = makeTrainExamples(self.dataset.getTokList(), tokenizer, self.dataset.getLabID()) #搞一个TensorDataset 里面有bert需要的4个元素
        dataloader = DataLoader(TensorData, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True) #根据刚才的tensordataset生成dataloader
        for epoch in range(self.epoch):
            model.train()
            batchTrAccSum = 0.0
            batchTrLossSum = 0.0
            timeEpochStart = time.time()
            for batch in dataloader:
                self.optimizer.zero_grad()
                Y, ids, types, masks = batch
                X = {'input_ids':ids.to(self.device),
                    'token_type_ids':types.to(self.device),
                    'attention_mask':masks.to(self.device)}
                _ ,logits = model.forward(X)
                lossSP = model.loss_ce(logits, Y.to(self.device))
                lossSP.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.optimizer.step()
                # calculate train acc
                YTensor = Y.cpu()
                logits = logits.detach().clone()
                if torch.cuda.is_available():
                    logits = logits.cpu()
                logits = logits.numpy()
                predictResult = np.argmax(logits, 1)
                acc = accuracy_score(YTensor, predictResult)
                # accumulate statistics
                batchTrAccSum += acc

            # current epoch training done, collect data
            durationTrain         = self.round(time.time() - timeEpochStart)
            durationOverallTrain += durationTrain
            batchTrAccAvrg        = self.round(batchTrAccSum/len(dataloader))

            # display current epoch's info
            logger.info("---- epoch: %d/%d, train_time %f ----", epoch, self.epoch, durationTrain)
            logger.info(" TrainAcc %f",batchTrAccAvrg)
            valAcc  = self.valEvaluate.evaluate(args, model,tokenizer, self.device, plot_cm=True,save=False,mode='SP',eval_or_test= 'eval') 
            teAcc  = self.testEvaluator.evaluate(args, model,tokenizer, self.device, plot_cm=True,save=False,mode='SP',eval_or_test= 'test') 
            logger.info("ValAcc %f,valdomain %s", valAcc,args.valDomain)
            logger.info("TestAcc %f", teAcc)
            
            if (valAcc >= valBestAcc):   # better validation result
                print("[INFO] Find a better model. Val acc: %f -> %f"%(valBestAcc, valAcc))
                valBestAcc = valAcc
                accumulateStep = 0
                # cache current model, used for evaluation later
                self.bestModelStateDict = copy.deepcopy(model.state_dict())
            else:
                accumulateStep += 1
                if accumulateStep > self.patience/2:
                    print('[INFO] accumulateStep: ', accumulateStep)
                    if accumulateStep == self.patience:  # early stop
                        logger.info('Early stop.')
                        logger.debug("Overall training time %f", durationOverallTrain)
                        logger.debug("best_val_acc: %f", valBestAcc)
                        break

class MLMOnlyTrainer(TrainerBase):
    def __init__(self,
                 trainingParam:dict,
                 optimizer,
                 device,
                 dataset:IntentDataset,
                 valEvaluator:EvaluatorBase,
                 testEvaluator:EvaluatorBase):
        super(MLMOnlyTrainer, self).__init__()
        self.epoch       = trainingParam['epoch']
        self.batch_size  = trainingParam['batch']
        self.mlm         = trainingParam['mlm']
        self.validation  = trainingParam['validation']
        self.tensorboard = trainingParam['tensorboard']
        self.patience    = trainingParam['patience']
        self.device = device
        self.dataset       = dataset
        self.optimizer     = optimizer
        self.testEvaluator = testEvaluator
        self.valEvaluate = valEvaluator

    def train(self, model, tokenizer, args):
        valBestAcc = -1
        durationOverallTrain = 0.0
        valacc= self.valEvaluate.evaluate(args, model, tokenizer, self.device,eval_or_test = 'eval') 
        teacc = self.testEvaluator.evaluate(args,model, tokenizer, self.device,eval_or_test = 'test')
        logger.info('---- Before training ----')
        logger.info("ValAcc %f,valdomain %s", valacc ,args.valDomain)
        logger.info("TestAcc %f,testdomain %s", teacc, args.testDomain)
        TensorData = makeTrainExamples(self.dataset.getTokList(), tokenizer, self.dataset.getLabID()) #搞一个TensorDataset 里面有bert需要的4个元素
        dataloader = DataLoader(TensorData, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True) #根据刚才的tensordataset生成dataloader
        
        for epoch in range(self.epoch):  # an epoch means all sampled tasks are done
            model.train()
            batchTrLossSum = 0.0
            timeEpochStart = time.time()
            for batch in dataloader:
                self.optimizer.zero_grad()
                Y, ids, types, masks = batch
                Y = Y.to(self.device)
                X = {'input_ids':ids.to(self.device),
                     'token_type_ids':types.to(self.device),
                     'attention_mask':masks.to(self.device)}
                mask_ids, mask_lb = mask_tokens(X['input_ids'].cpu(), tokenizer)
                X = {'input_ids':mask_ids.to(self.device),
                     'token_type_ids':X['token_type_ids'],
                     'attention_mask':X['attention_mask']}
                with torch.set_grad_enabled(True):
                    _,logits = model(X)
                    loss_src = model.loss_ce(logits, Y)
                    loss_mlm = model.mlmForward(X, mask_lb.to(self.device))
                    loss = loss_src + loss_mlm
                batchTrLossSum= batchTrLossSum + loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.optimizer.step()
            durationTrain = self.round(time.time() - timeEpochStart)
            durationOverallTrain += durationTrain
            batchTrLossAvrg = batchTrLossSum / len(dataloader)
            logger.info("---- epoch: %d/%d, train_time %f ----", epoch, self.epoch, durationTrain)
            logger.info("TrainLoss %f", batchTrLossAvrg)
            valAcc= self.valEvaluate.evaluate(args, model, tokenizer, self.device, plot_cm=True,save=False,mode='MLM',eval_or_test= 'eval') 

            if (valAcc >= valBestAcc):   # better validation result
                print("[MLM] Find a better model. Val acc: %f -> %f"%(valBestAcc, valAcc))
                valBestAcc = valAcc
                accumulateStep = 0
                self.bestModelStateDict = copy.deepcopy(model.state_dict())
            else:
               accumulateStep += 1
               if accumulateStep > self.patience/2:
                   print('[MLM] accumulateStep: ', accumulateStep)
                   if accumulateStep == self.patience:  # early stop
                       logger.info('Early stop.')
                       logger.debug("Overall training time %f", durationOverallTrain)
                       logger.debug("best_val_acc: %f", valBestAcc)
                       break
                    
class updatephase(TrainerBase):
    def __init__(self,
                 trainingParam:dict,
                 device,
                 dataset:IntentDataset,
                 testEvaluator:EvaluatorBase):
        super(updatephase, self).__init__()
        self.epoch       = trainingParam['epoch']
        self.batch_size  = trainingParam['batch']
        self.validation  = trainingParam['validation']
        self.patience    = trainingParam['patience']
        self.tensorboard = trainingParam['tensorboard']
        self.lambda_adv  = trainingParam['lambda adv']


        self.dataset       = dataset
        self.testEvaluator = testEvaluator
        self.device = device
        self.centroids = None
    
    def generate_default_inputs(self,batch,model,source_or_target,tokenizer,label = False):

        batch = tuple(temp.to(self.device) for temp in batch)

        if label == True:
            label,block_flags,input_ids,token_type_ids,attention_mask = batch 
        else :
            block_flags,input_ids,token_type_ids,attention_mask = batch 

        mask_ids, mask_lb = mask_tokens(input_ids.cpu(), tokenizer,special_tokens_mask = block_flags.cpu())

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

    def metatest(self, model, tokenizer,args):
        self.processor_labeled = handle_method(dataset=self.dataset,tok = tokenizer)
        self.bestModelStateDict = copy.deepcopy(model.state_dict())
        durationOverallTrain = 0.0
        durationOverallVal = 0.0
        valBestAcc = -1
        accumulateStep = 0

        labTensorData = self.processor_labeled.get_input_features(label = self.dataset.getLabID())
        labdataloader = DataLoader(labTensorData, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True) #根据刚才的tensordataset生成dataloader

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
        t_total = len(labdataloader) // 1 
        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)
        #optimizer.to(device)
        #scheduler.to(device)

        embedding_optimizer = AdamW(embedding_parameters, lr=1e-5, eps=1e-8)
        embedding_scheduler = get_linear_schedule_with_warmup(embedding_optimizer, num_warmup_steps=0, num_training_steps=t_total)
        #embedding_optimizer.to(device)
        #embedding_scheduler.to(device)

        for epoch in range(1):  # an epoch means all sampled tasks are don
            cur_model.train()
            for batch in labdataloader:
                cur_model.train()
                optimizer.zero_grad()
                embedding_optimizer.zero_grad()
                inputs_un,un_mlm_label = self.generate_default_inputs(batch ,cur_model,label = True,source_or_target = 'source',tokenizer=tokenizer)
                loss_mlmt,feature_t,logits_t = cur_model.mlmForward(inputs_un, un_mlm_label.to(self.device))
                loss_unlab =  loss_mlmt
                loss_unlab.backward()
                torch.nn.utils.clip_grad_norm_(cur_model.parameters(), 1.0)
                #scheduler.step()
                embedding_optimizer.step()
                optimizer.step()
                embedding_scheduler.step()
                scheduler.step()

        del  optimizer, embedding_optimizer,scheduler,embedding_scheduler


        teAcc  = self.testEvaluator.evaluate(args,cur_model, tokenizer, self.device,save=True,mode='meta_update',eval_or_test='meta_test',head='source')
        print('final test acc:')
        print(teAcc)


