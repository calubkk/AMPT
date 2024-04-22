import os
os.environ['CUDA_VISIBLE_DEVICES']='2,3'
import torch
import torch.optim as optim
import argparse
import time
import copy
from transformers import AutoTokenizer
import random
from init_parameter import init_model
from utils.models import BERT,ContinuousPrompt
from utils.IntentDataset import IntentDataset
from utils.Trainer import metaphase,MLMOnlyTrainer,updatephase,SpTrainer
from utils.Evaluator import Pre_Evaluator,FewShotEvaluator,prompt_Evaluator
from utils.commonVar import *
from utils.printHelper import *
from utils.tools import *
from utils.Logger import logger
from utils.memory import MemoryBank,fill_memory_bank
from utils.neighbor_dataset import NeighborsDataset
from sklearn.cluster import KMeans
#This code is used for the evaluation of baseline
from scipy.optimize import linear_sum_assignment
import warnings
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    # ======= process arguments ======
    args = init_model()
    print(args)
    if not args.saveModel:  #是否会在训练后保存模型
        logger.info("The model will not be saved after training!")
    # ==== setup logger ====   #确定logging的模式，也就是info或者是debug
    if args.loggingLevel == LOGGING_LEVEL_INFO:  
        loggingLevel = logging.INFO
    elif args.loggingLevel == LOGGING_LEVEL_DEBUG:
        loggingLevel = logging.DEBUG
    else:
        raise NotImplementedError("Not supported logging level %s", args.loggingLevel)
    logger.setLevel(loggingLevel)
    # ==== set seed ====
    set_seed(args.seed)  #设置随机种子
    # ======= process data ======
    # tokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    # load raw dataset
    logger.info(f"Loading data from {args.dataDir}") #这个文件夹中会有好多种data
    dataset = IntentDataset()
    dataset.loadDataset(splitName(args.dataDir))
    dataset.tokenize(tok) #是在这里去做的分词
    # spit data into training, validation and testing
    logger.info("----- Training Data -----")
    trainData = dataset.splitDomain(splitName(args.sourceDomain)) 
    targetData = dataset.splitDomain(splitName(args.targetDomain))
    train_target, test_target= targetData.train_test_split(perc=0.7)  #先将数据分成7:3
    train_target, val_target=  train_target.train_test_split(perc=0.7)  #再将训练数据分成7:3
    test_targetone = targetData.splitDomain(splitName(args.splitargetDomainone))
    test_targettwo = targetData.splitDomain(splitName(args.splitargetDomaintwo))
    logger.info(f"Source domain is  {args.sourceDomain}.") 
    logger.info(f"Target domain is  {args.targetDomain}.") 
    logger.info("----- Validation Data -----")
    #valData = dataset.splitDomain(splitName(args.valDomain))
    #valData = copy.deepcopy(test_target)
    valData = copy.deepcopy(train_target)
    logger.info(f"Val domain is  {args.valDomain}.") 
    logger.info("----- Testing Data -----")
    testData = dataset.splitDomain(splitName(args.testDomain))
    testData1 = dataset.splitDomain(splitName(args.testDomain1))
    testData2 = dataset.splitDomain(splitName(args.testDomain2))
    #testData = dataset.splitDomain(splitName(args.testDomain))
    #testData = dataset.splitDomain(splitName(args.testDomain))
    logger.info(f"Test domain is  {args.testDomain}.") 
    # shuffle word order
    if args.shuffle:
        trainData.shuffle_words()
    unlabeledData = None
    #unlabeledData = copy.deepcopy(train_target)
    unlabeledData = copy.deepcopy(train_target)
    if args.shuffle_mlm:
        unlabeledData.shuffle_words()
    if args.mlm:
        # ======= prepare model ======
        # initialize model
        modelConfig = {}
        #device = torch.device('cuda' if not args.disableCuda else 'cpu')
        modelConfig['clsNumber'] = trainData.getLabNum()
        modelConfig['LMName'] = args.LMName
        modelConfig['feat_dim'] = args.feat_dim
        if args.disableCuda:
            device = torch.device('cpu')
            model = BERT(modelConfig,device)
            logger.info(f"CPU set.") 
        else:
            device = torch.device('cuda')
            model = BERT(modelConfig,device)
            model.to(device)
            logger.info(f"GPU set.") 
            n_gpu = torch.cuda.device_count()
            if n_gpu > 1:
                model = nn.DataParallel(model)
            logger.info(f"There are {n_gpu} GPUs.") 
            model = model.module if isinstance(model, torch.nn.DataParallel) else model #https://blog.csdn.net/qq_31347869/article/details/124737982
        logger.info("----- BERT for pretraining has been initialized -----")

        # setup validator
        
        Pre_validator = Pre_Evaluator(train_target)
        Pre_target_tester = Pre_Evaluator(val_target)
        Pre_tester = Pre_Evaluator(testData)
        Pre_tester1 = Pre_Evaluator(testData1)
        Pre_tester2 = Pre_Evaluator(testData2)
        Pre_tester_one = Pre_Evaluator(test_targetone)
        Pre_tester_two = Pre_Evaluator(test_targettwo)
        Pre_meta_tester = Pre_Evaluator(testData)
        
        logger.info("----- Evaluator for pretraining has been initialized -----")

        # setup trainer
        optimizer = None
        if args.optimizer == OPTER_ADAM:
            optimizer = optim.Adam(model.parameters(), lr=args.learningRate, weight_decay=args.weightDecay)
        elif args.optimizer == OPTER_SGD:
            optimizer = optim.SGD(model.parameters(), lr=args.learningRate, weight_decay=args.weightDecay)
        else:
            raise NotImplementedError("Not supported optimizer %s"%(args.optimizer))
        logger.info(f"----- Optimizer for pretraining is {args.optimizer} -----")

        trainingParam = {"epoch"      : args.epochs, \
                        "batch"      : args.batch_size, \
                        "validation" : args.validation, \
                        "patience"   : args.patience, \
                        "tensorboard": args.tensorboard, \
                        "mlm"        : args.mlm, \
                        "lambda mlm" : args.lambda_mlm}
        '''
        trainer = SpTrainer(trainingParam, optimizer,device, trainData, Pre_validator, Pre_tester)
        trainer.train(model, tok,args)
        bestModelStateDict = trainer.getBestModelStateDict()
        model.load_state_dict(bestModelStateDict)
        logger.info("Source domain pretraining finished.")
        '''
        

        trainer = MLMOnlyTrainer(trainingParam, optimizer,device, trainData, Pre_validator, Pre_tester)
        trainer.train(model, tok,args)
        bestModelStateDict = trainer.getBestModelStateDict()
        model.load_state_dict(bestModelStateDict)
        logger.info("Source domain pretraining finished.")

        '''
        trainer = MLMOnlyTrainer(trainingParam, optimizer,device, unlabeledData, Pre_validator, Pre_tester)
        trainer.train(model, tok,args)
        bestModelStateDict = trainer.getBestModelStateDict()
        model.load_state_dict(bestModelStateDict)
        logger.info("Target domain pretraining finished.")
        '''
        
        valAcc  = Pre_target_tester.evaluate(args, model,tok, device, plot_cm=True,save=True,mode='MLM',eval_or_test= 'eval') 
        teAcc  = Pre_tester.evaluate(args, model, tok, device, plot_cm=True,save=True,mode='MLM',eval_or_test= 'test') 
        teAcc1  = Pre_tester1.evaluate(args, model, tok, device, plot_cm=True,save=True,mode='MLM',eval_or_test= 'test1') 
        teAcc2  = Pre_tester2.evaluate(args, model, tok, device, plot_cm=True,save=True,mode='MLM',eval_or_test= 'test2') 
        tt=(teAcc+teAcc1+teAcc2)/3
        #teAcc_one  = Pre_tester_one.evaluate(args, model, tok, device, plot_cm=True,save=True,mode='MLM_one',eval_or_test= 'test') 
        #teAcc_two  = Pre_tester_two.evaluate(args, model, tok, device, plot_cm=True,save=True,mode='MLM_two',eval_or_test= 'test') 
        #meta_teAcc  = Pre_meta_tester.evaluate(args, model, tok, device, plot_cm=True,save=True,mode='MLM',eval_or_test= 'meta-test') 
        logger.info("val acc:%f,test acc:%f.",valAcc,tt)
        
        if args.saveModel:
            # decide the save name
            if args.saveName == 'none':
                prefix = "MLM" 
                save_path = os.path.join(SAVE_PATH, f'{prefix}')
                if args.shuffle:
                    save_path += "_shufflesource"
                if args.shuffle_mlm:
                    save_path += "_shuffletarget"
            else:
                save_name = "MTPtrain_multi"  + str(args.seed) + args.dataDir
                save_path = os.path.join(SAVE_PATH, save_name)
            save_path = save_path
            logger.info("Saving model.pth into folder: %s", save_path)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            model.save(save_path)
            logger.info(f"----- Pretrained model_MLM has been saved in {save_path} -----")  
        logger.info("----- MLM finished -----")
        
    

    logger.info("----- Starting Meta_train -----")
    modelConfig = {}
    modelConfig['clsNumber'] = trainData.getLabNum()
    modelConfig['LMName'] = args.LMName
    modelConfig['feat_dim'] = args.feat_dim

   #if args.mlm == False:
    save_path = "./saved_models/"+ "MTPtrain_multi"  + str(args.seed) + args.dataDir
    if args.disableCuda:
        device = torch.device('cpu')
        model = ContinuousPrompt(modelConfig,device,tok,save_path)
        logger.info(f"CPU set.") 
    else:
        device = torch.device('cuda')
        model = ContinuousPrompt(modelConfig,device,tok,save_path)
        model.to(device)
        logger.info(f"GPU set.") 
        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            model = nn.DataParallel(model)
        model = model.module if isinstance(model, torch.nn.DataParallel) else model #https://blog.csdn.net/qq_31347869/article/details/124737982
        logger.info("----- ContinuousPrompt for meta_train has been initialized -----")
    # setup validator
    Prompt_validator = prompt_Evaluator(train_target)
    Prompt_target_tester = prompt_Evaluator(val_target)
    Prompt_tester = prompt_Evaluator(testData)
    Prompt_tester1 = prompt_Evaluator(testData1)
    Prompt_tester2 = prompt_Evaluator(testData2)
    Prompt_tester_one = prompt_Evaluator(test_targetone)
    Prompt_tester_two = prompt_Evaluator(test_targettwo)
    Meta_tester = prompt_Evaluator(testData)
    #Fewshot_validator = FewShotEvaluator(evalParam, taskParam, valData,device)
    #Fewshot_tester = FewShotEvaluator(evalParam, taskParam, testData,device)
    logger.info("----- Evaluator for meta_train has been initialized -----")
    
    trainingParam = {"epoch"      : args.epochs, \
                     "batch"      : args.batch_size, \
                     "validation" : args.validation, \
                     "patience"   : args.patience, \
                     "lambda adv" : args.lambda_adv,\
                     "tensorboard": args.tensorboard
                     }
    
    trainer = metaphase(trainingParam, device, trainData, unlabeledData, Prompt_validator, Prompt_tester)
    trainer.train(model, tok,args)
    logger.info("-----  Meta_train has finished. -----")
    bestModelStateDict = trainer.getBestModelStateDict()
    model.load_state_dict(bestModelStateDict)
    valAcc  = Prompt_target_tester.evaluate(args, model,tok, device, plot_cm=True,save=True,mode='meta-train',eval_or_test= 'eval',head='target') 
    teAcc   = Prompt_tester.evaluate(args, model, tok, device, plot_cm=True,save=True,mode='meta-train',eval_or_test= 'test',head='source') 
    teAcc1   = Prompt_tester1.evaluate(args, model, tok, device, plot_cm=True,save=True,mode='meta-train',eval_or_test= 'test1',head='source') 
    teAcc2   = Prompt_tester2.evaluate(args, model, tok, device, plot_cm=True,save=True,mode='meta-train',eval_or_test= 'test2',head='source') 
    tt=(teAcc+teAcc1+teAcc2)/3
    logger.info("val acc:%f,test acc:%f.",valAcc,tt)
    if args.saveModel:
        # decide the save name
        if args.saveName == 'none':
            prefix = "MLM" 
            save_path = os.path.join(SAVE_PATH, f'{prefix}')
            if args.shuffle:
                save_path += "_shufflesource"
            if args.shuffle_mlm:
                save_path += "_shuffletarget"
        else:
            save_name = "AMPTtrain_multi"  + str(args.seed) + args.dataDir
            save_path = os.path.join(SAVE_PATH, save_name)
        save_path = save_path
        logger.info("Saving model.pth into folder: %s", save_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        model.save(save_path,logger)
        logger.info(f"-----  model has been saved in {save_path} -----")  




if __name__ == "__main__":
    main()
    exit(0)
