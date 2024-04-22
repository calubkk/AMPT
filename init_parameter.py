"""
modified from
https://github.com/thuiar/DeepAligned-Clustering/blob/main/init_parameter.py
"""

from argparse import ArgumentParser

def init_model():
    parser = ArgumentParser()
    # ==== model ====
    parser.add_argument('--seed', default=1, type=int)  ############
    parser.add_argument('--mode', default='no-meaning',
                        help='Choose from multi-class')
    parser.add_argument('--tokenizer', default='bert-base-uncased',  ##########
                        help="Name of tokenizer")
    parser.add_argument('--LMName', default='bert-base-uncased',  #######
                        help='Name for models and path to saved model')
    parser.add_argument("--feat_dim", default=768, type=int, ######
                        help="Bert feature dimension.")
    parser.add_argument("--lambda_adv", default=1, type=float,
                        help="lambda of adv")
    
    # ==== CLNN related ====
    parser.add_argument("--rtr_prob", default=0.25, type=float, help="Probability for random token replacement")

    parser.add_argument("--view_strategy", default="rtr", type=str, help="Choose from rtr|shuffle|none")

    # ==== dataset ====
    parser.add_argument('--dataDir',  ##########
                        help="Dataset names included in this experiment and separated by comma. "
                        "For example:'OOS,bank77,hwu64'")
    parser.add_argument('--sourceDomain',  ###########
                        help="Source domain names and separated by comma. "
                        "For example:'travel,banking,home'")
    parser.add_argument('--testDomain',  ###########
                        help="test domain names and separated by comma. "
                        "For example:'travel,banking,home'")
    parser.add_argument('--testDomain1',  ###########
                        help="test domain names and separated by comma. "
                        "For example:'travel,banking,home'")
    parser.add_argument('--testDomain2',  ###########
                        help="test domain names and separated by comma. "
                        "For example:'travel,banking,home'")
    parser.add_argument('--valDomain',  #########
                        help='Validation domain names and separated by comma')
    parser.add_argument('--targetDomain',###########
                        help='Target domain names and separated by comma')
    parser.add_argument('--splitargetDomainone',###########
                        help='Target domain names and separated by comma')
    parser.add_argument('--splitargetDomaintwo',###########
                        help='Target domain names and separated by comma')

    # ==== evaluation task ====
    parser.add_argument("--save_results_path", type=str, default='results',
                        help="The path to save results.")

    # ==== optimizer ====
    parser.add_argument('--optimizer', default='Adam', #########
                        help='Choose from SGD|Adam')
    parser.add_argument('--learningRate', type=float, default=2e-5) #####
    parser.add_argument('--weightDecay', type=float, default=0)########

    # ==== pretraining arguments ====
    parser.add_argument('--disableCuda', action="store_true") ########
    parser.add_argument('--validation', default=True)  ########
    parser.add_argument('--epochs', type=int, default=100)  ###########
    parser.add_argument('--batch_size', type=int, default=64)      ##########
    parser.add_argument('--patience', type=int, default=5,   ###########
                        help="Early stop when performance does not go better")
    parser.add_argument('--mlm', default =False,      ##########
                        help="If use mlm as auxiliary loss")
    parser.add_argument('--lambda_mlm', type=float, default=1.0,  #########
                        help="The weight for mlm loss")
    parser.add_argument('--shuffle_mlm', action="store_true")  ############
    parser.add_argument('--shuffle', action="store_true") #############
    
    # ==== cl_training arguments ====
    parser.add_argument('--cl_epochs', type=int, default=100)
    parser.add_argument('--cl_batch_size', type=int, default=128)
    parser.add_argument('--cl_patience', type=int, default=5,
                        help="Early stop when performance does not go better")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Warmup proportion for optimizer.")
    parser.add_argument("--lr", default=1e-5, type=float,
                        help="The learning rate for training.")
    parser.add_argument("--update_per_epoch", default=5, type=int,
                        help="Update pseudo labels after certain amount of epochs")
    parser.add_argument("--topk", default=30, type=int,
                        help="Select topk nearest neighbors")
    parser.add_argument("--temp", default=0.07, type=float,
                        help="Temperature for contrastive loss")
    parser.add_argument("--grad_clip", default=1, type=float,
                        help="Value for gradient clipping.")

    
    # ==== other things ====
    parser.add_argument('--loggingLevel', default='INFO',  #########
                        help="python logging level")
    parser.add_argument('--method', default='None',
                        help="method")
    parser.add_argument('--saveModel', action='store_true',   ##########
                        help="Whether to save pretrained model")
    parser.add_argument('--saveName', default='none', ###########
                        help="Specify a unique name to save your model"
                        "If none, then there will be a specific name controlled by how the model is trained")
    parser.add_argument('--tensorboard', action='store_true', ####
                        help="Enable tensorboard to log training and validation accuracy")
    
    # ==== evaluation task ====
    parser.add_argument('--way', type=int, default=4)
    parser.add_argument('--shot', type=int, default=2)
    parser.add_argument('--query', type=int, default=5)
    parser.add_argument('--evalTaskNum', type=int, default=500)
    parser.add_argument('--clsFierName', default='Linear',
                        help="Classifer name for few-shot evaluation"
                        "Choose from Linear|SVM|NN|Cosine|MultiLabel")

    args = parser.parse_args()

    return args


