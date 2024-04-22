import random
import string
from collections import defaultdict
from typing import Tuple, List, Union, Dict
from torch.utils.data import TensorDataset

import torch
from transformers import PreTrainedTokenizer, GPT2Tokenizer
import log
from utils.tools import mask_tokens

class handle_method():
    def __init__(self, dataset,tok,pattern_id = 4, seed = 42 , mlm = False):

        self.pattern_id = pattern_id
        self.rng = random.Random(seed)
        self.dataset = dataset
        self.tokenizer = tok
        self.mlm = mlm

    
    def mask(self) -> str:
        """Return the underlying LM's mask token"""
        return self.tokenizer.mask_token

    def regeneratedata(self):
        examples = []
        for idx,line in enumerate(self.dataset.uttList):
            example = InputExample(text=line,domain=self.dataset.domList[idx], label=self.dataset.labList[idx])
            examples.append(example)
        return examples

    
    def shortenable(self,s):
        """Return an instance of this string that is marked as shortenable"""
        return s, True
    
    def get_parts(self, example):

        text = self.shortenable(example.text)

        # searched patterns in fully-supervised learning
        # string_list_a = [passage, '.', 'the', 'Question:', question, '?', 'the', 'Answer:', self.mask]
        # string_list_a = [passage, '.', 'the', question, '?', 'the', self.mask]
        # string_list_a = [passage, 'the', question, '?', 'the', self.mask]
        #string_list_a = [text, '.','What', 'is', 'the', 'aim', 'of', 'this', 'sentence', '?', 'the', '[MASK]']       #这里是设置prompt的地方
        #block_flag_a = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        string_list_a = ['The', 'intent','of', text, '.']       #这里是设置prompt的地方
        block_flag_a = [ 1, 1, 1, 0, 1]
        assert len(string_list_a) == len(block_flag_a)
        return string_list_a,block_flag_a


    def encode(self, example, priming: bool = False, labeled: bool = False) \
            -> Tuple[List[int], List[int]]:
        """
         an input example using this pattern-verbalizer pair.

        :param example: the input example to encode
        :param priming: whether to use this example for priming
        :param labeled: if ``priming=True``, whether the label should be appended to this example
        :return: A tuple, consisting of a list of input ids and a list of token type ids
        """

        tokenizer = self.tokenizer  # type: PreTrainedTokenizer

        parts_a, block_flag_a = self.get_parts(example)

        parts_a = [x if isinstance(x, tuple) else (x, False) for x in parts_a]
        parts_a = [(tokenizer.encode(x, add_special_tokens=False), s) for x, s in parts_a if x]
        origin_a = [tokenizer.decode(x) for x, s in parts_a if x]

        tokens_a = [token_id for part, _ in parts_a for token_id in part]

        ### add
        assert len(parts_a) == len(block_flag_a)

        block_flag_a = [flag for (part, _), flag in zip(parts_a, block_flag_a) for _ in part]
        assert len(tokens_a) == len(block_flag_a)

        #examples = tokenizer.pad(tokens_a, padding='longest', return_tensors='pt')

        input_ids = tokenizer.build_inputs_with_special_tokens(tokens_a)
        token_type_ids = tokenizer.create_token_type_ids_from_sequences(tokens_a)
        block_flag = tokenizer.build_inputs_with_special_tokens(block_flag_a)
        block_flag = [item if item in [0, 1] else 0 for item in block_flag]
        #print(origin_a)
        #print(block_flag)
        #print(tokens_a)

        ### return input_ids, token_type_ids
        return input_ids, block_flag, token_type_ids
    
    def get_input_features(self, label = False):
        tokens = []
        block_flags = []
        #origin_as = []
        attention = []
        token_type_ids = []

        examples = self.regeneratedata()

        for idx,example in enumerate(examples):
            token, block_flag, token_type_ids_ = self.encode(example)
            attention_mask = [1] * len(token)
            attention.append(attention_mask)
            #token = torch.tensor([t for t in token], dtype=torch.long)
            tokens.append(token)
            block_flags.append(block_flag)
            token_type_ids.append(token_type_ids_)
        
        max_length = 0
        #examples = self.tokenizer.pad(tokens, padding='longest', return_tensors='pt')
        for token in tokens:
            if len(token)>max_length:
                max_length = len(token)

        length = max_length
        #mlm_labels = []
        #print(attention)

        for idx,example in enumerate(examples):
            padding_length = length - len(block_flags[idx])
            tokens[idx] = tokens[idx] + ([self.tokenizer.pad_token_id] * padding_length)
            block_flags[idx] = block_flags[idx] + ([0] * padding_length)
            attention[idx] = attention[idx] + ([0] * padding_length)
            token_type_ids[idx] = token_type_ids[idx] + ([0] * padding_length)
            assert len(tokens[idx]) == length
            assert len(block_flags[idx]) == length
            assert len(attention[idx]) == length
            assert len(token_type_ids[idx]) == length
            #print(block_flags[idx])
            #print(attention[idx])
            #print(token_type_ids[idx])
            #print(tokens[idx])
            #label_idx = examples['input_ids'][idx].index(self.mask_id)
            #labels = [-1] * length
            #labels[label_idx] = 1
            #mlm_labels.append(labels)
        
        block_flags = torch.tensor([block_flag for block_flag in block_flags], dtype=torch.long)
        input_ids = torch.tensor([token for token in tokens], dtype=torch.long)
        attention_mask = torch.tensor([att for att in attention], dtype=torch.long)
        token_type_ids = torch.tensor([ty for ty in token_type_ids], dtype=torch.long)
        #mlm_labels = torch.tensor([f.mlm_labels for f in features], dtype=torch.long),

        if label:
            if not isinstance(label, torch.Tensor):
                label = torch.tensor(label,dtype=torch.long)
            examples = TensorDataset(label,
                                    block_flags,
                                    input_ids,
                                    token_type_ids,
                                    attention_mask)
        else:
            examples = TensorDataset(block_flags,
                                    input_ids,
                                    token_type_ids,
                                    attention_mask)

        return examples

class update_method():
    def __init__(self, utt,label,dom,tok,pattern_id = 4, seed = 42 , mlm = False):

        self.pattern_id = pattern_id
        self.rng = random.Random(seed)
        self.utt = utt
        self.label = label
        self.dom = dom
        self.tokenizer = tok
        self.mlm = mlm

    
    def mask(self) -> str:
        """Return the underlying LM's mask token"""
        return self.tokenizer.mask_token

    def regeneratedata(self):
        examples = []
        for idx,line in enumerate(self.utt):
            example = InputExample(text=line,domain=self.dom[idx], label=self.label[idx])
            examples.append(example)
        return examples

    
    def shortenable(self,s):
        """Return an instance of this string that is marked as shortenable"""
        return s, True
    
    def get_parts(self, example):

        text = self.shortenable(example.text)

        # searched patterns in fully-supervised learning
        # string_list_a = [passage, '.', 'the', 'Question:', question, '?', 'the', 'Answer:', self.mask]
        # string_list_a = [passage, '.', 'the', question, '?', 'the', self.mask]
        # string_list_a = [passage, 'the', question, '?', 'the', self.mask]
        #string_list_a = [text, '.','What', 'is', 'the', 'aim', 'of', 'this', 'sentence', '?', 'the', '[MASK]']       #这里是设置prompt的地方
        #block_flag_a = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        string_list_a = ['The', 'intent','of', text, '.']       #这里是设置prompt的地方
        block_flag_a = [ 1, 1, 1, 0, 1]
        assert len(string_list_a) == len(block_flag_a)
        return string_list_a,block_flag_a


    def encode(self, example, priming: bool = False, labeled: bool = False) \
            -> Tuple[List[int], List[int]]:
        """
         an input example using this pattern-verbalizer pair.

        :param example: the input example to encode
        :param priming: whether to use this example for priming
        :param labeled: if ``priming=True``, whether the label should be appended to this example
        :return: A tuple, consisting of a list of input ids and a list of token type ids
        """

        tokenizer = self.tokenizer  # type: PreTrainedTokenizer

        parts_a, block_flag_a = self.get_parts(example)

        parts_a = [x if isinstance(x, tuple) else (x, False) for x in parts_a]
        parts_a = [(tokenizer.encode(x, add_special_tokens=False), s) for x, s in parts_a if x]
        origin_a = [tokenizer.decode(x) for x, s in parts_a if x]

        tokens_a = [token_id for part, _ in parts_a for token_id in part]

        ### add
        assert len(parts_a) == len(block_flag_a)

        block_flag_a = [flag for (part, _), flag in zip(parts_a, block_flag_a) for _ in part]
        assert len(tokens_a) == len(block_flag_a)

        #examples = tokenizer.pad(tokens_a, padding='longest', return_tensors='pt')

        input_ids = tokenizer.build_inputs_with_special_tokens(tokens_a)
        token_type_ids = tokenizer.create_token_type_ids_from_sequences(tokens_a)
        block_flag = tokenizer.build_inputs_with_special_tokens(block_flag_a)
        block_flag = [item if item in [0, 1] else 0 for item in block_flag]
        #print(origin_a)
        #print(block_flag)
        #print(tokens_a)

        ### return input_ids, token_type_ids
        return input_ids, block_flag, token_type_ids
    
    def get_input_features(self, label = False):
        tokens = []
        block_flags = []
        #origin_as = []
        attention = []
        token_type_ids = []

        examples = self.regeneratedata()

        for idx,example in enumerate(examples):
            token, block_flag, token_type_ids_ = self.encode(example)
            attention_mask = [1] * len(token)
            attention.append(attention_mask)
            #token = torch.tensor([t for t in token], dtype=torch.long)
            tokens.append(token)
            block_flags.append(block_flag)
            token_type_ids.append(token_type_ids_)
        
        max_length = 0
        #examples = self.tokenizer.pad(tokens, padding='longest', return_tensors='pt')
        for token in tokens:
            if len(token)>max_length:
                max_length = len(token)

        length = max_length
        #mlm_labels = []
        #print(attention)

        for idx,example in enumerate(examples):
            padding_length = length - len(block_flags[idx])
            tokens[idx] = tokens[idx] + ([self.tokenizer.pad_token_id] * padding_length)
            block_flags[idx] = block_flags[idx] + ([0] * padding_length)
            attention[idx] = attention[idx] + ([0] * padding_length)
            token_type_ids[idx] = token_type_ids[idx] + ([0] * padding_length)
            assert len(tokens[idx]) == length
            assert len(block_flags[idx]) == length
            assert len(attention[idx]) == length
            assert len(token_type_ids[idx]) == length
            #print(block_flags[idx])
            #print(attention[idx])
            #print(token_type_ids[idx])
            #print(tokens[idx])
            #label_idx = examples['input_ids'][idx].index(self.mask_id)
            #labels = [-1] * length
            #labels[label_idx] = 1
            #mlm_labels.append(labels)
        
        block_flags = torch.tensor([block_flag for block_flag in block_flags], dtype=torch.long)
        input_ids = torch.tensor([token for token in tokens], dtype=torch.long)
        attention_mask = torch.tensor([att for att in attention], dtype=torch.long)
        token_type_ids = torch.tensor([ty for ty in token_type_ids], dtype=torch.long)
        #mlm_labels = torch.tensor([f.mlm_labels for f in features], dtype=torch.long),

        if label:
            if not isinstance(label, torch.Tensor):
                label = torch.tensor(label,dtype=torch.long)
            examples = TensorDataset(label,
                                    block_flags,
                                    input_ids,
                                    token_type_ids,
                                    attention_mask)
        else:
            examples = TensorDataset(block_flags,
                                    input_ids,
                                    token_type_ids,
                                    attention_mask)

        return examples
    
class InputExample(object):
    """A raw input example consisting of one or two segments of text and a label"""

    def __init__(self, text, domain=None, label=None, meta=None):
        """
        Create a new InputExample.
        :param guid: a unique textual identifier
        :param text_a: the sequence of text
        :param text_b: an optional, second sequence of text
        :param label: an optional label
        :param logits: an optional list of per-class logits
        :param meta: an optional dictionary to store arbitrary meta information
        :param idx: an optional numeric index
        """
        self.text = text
        self.label = label
        self.domain = domain
        self.meta = meta if meta else {}