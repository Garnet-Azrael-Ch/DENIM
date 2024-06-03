import json, logging
import torch
from torch.utils.data import Dataset
from collections import namedtuple
import numpy as np
import random

logger = logging.getLogger(__name__)

gen_batch_fields = ['enc_idxs', 'enc_attn', 'dec_idxs', 'dec_attn', 'lbl_idxs', 'EDU_graph', 'EDU_text_cat_idxs', 'EDU_text_cat_attn', 'EDU_attn', 'EDU_index']
GenBatch = namedtuple('GenBatch', field_names=gen_batch_fields, defaults=[None] * len(gen_batch_fields))

def get_meaningful_ARs_token(AC_num, dataset):
    # 给定AC数量，返回有意义的AR索引
    list_meaningful_index = []
    
    if dataset=="AAEC":
        if AC_num < 1:
            return list_meaningful_index
        else:
            index = 0
            for i in range(int(AC_num - 1)):
                for j in range(int(AC_num - 1 - i)):
                    list_meaningful_index.append(int(index))
                    index += 1
                index += (12 - AC_num)
    elif dataset=="AbstRCT":
        if AC_num < 1:
            return list_meaningful_index
        else:
            index = 0
            for i in range(int(AC_num - 1)):
                for j in range(int(AC_num - 1 - i)):
                    list_meaningful_index.append(int(index))
                    index += 1
                index += (11 - AC_num)
    else:
        raise ValueError
    
    return list_meaningful_index

def mapping_ARs_token_index(relation_tuple, dataset):
    
    # 以AAEC为例，把AR[0,1]看做索引0， AR[10,11]是最后一个索引
    
    relation_tgt = relation_tuple[0]
    relation_src = relation_tuple[1]
        
    if dataset == "AAEC":
        if relation_tgt < 0 or relation_tgt > 11 or relation_src < 0 or relation_src > 11:
            print("ar pair idx error")
            raise ValueError
    elif dataset == "AbstRCT":
        if relation_tgt < 0 or relation_tgt > 10 or relation_src < 0 or relation_src > 10:
            print("ar pair idx error")
            raise ValueError
    else:
        raise ValueError
        
    # 例如传入的是 (0,3)或者(3,0)，即 3 -> 0  或 0 -> 3的关系
    if relation_tgt < relation_src:
        flag_sm2big = False  # 表示(0,3)在AR[0,3]那个token，关系方向是0 <- 3
        smaller_index = relation_tgt
        bigger_index = relation_src
    elif relation_tgt > relation_src:
        flag_sm2big = True  # 表示(3,0)在AR[0,3]那个token，关系方向是0 -> 3
        smaller_index = relation_src
        bigger_index = relation_tgt
    else:
        print("ar pair idx equal")
        raise ValueError

    if bigger_index - smaller_index > 12:
        print(bigger_index)
        print(smaller_index)
        print("ar pair distance error")
        raise ValueError

    
    if dataset == "AAEC":
        ARs_token_index = (11 + 11-smaller_index+1) * smaller_index / 2 + (bigger_index - smaller_index) - 1
    elif dataset == "AbstRCT":
        ARs_token_index = (10 + 10-smaller_index+1) * smaller_index / 2 + (bigger_index - smaller_index) - 1
    else:
        raise ValueError
    
    return int(ARs_token_index), flag_sm2big

class PEDataset(Dataset):
    def __init__(self, config, tokenizer, data_name, data_list, unseen_types=[], no_bos=False):
        self.config = config
        self.tokenizer = tokenizer
        self.data_name = data_name
        self.data_list = data_list
        self.no_bos = no_bos # if you use bart, then this should be False; if you use t5, then this should be True
        self.data = []

        self.dataset = 'AAEC'
        self.num_prompt_token = 12 + 12 + 66
        self.num_span_token = 12
        self.num_AC_token = 12
        self.num_ARs_token = 66
        self.num_AC_type = 3
        self.num_ARs_type = 3

        self.load_data(unseen_types)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def load_data(self, unseen_types):
  
        
        for sample in self.data_list:
            input = sample['para_text'].split()
            para_length = len(input)
            input.insert(0, '<s>')
            input.append('</s>')
            
            # 处理label
            label_list = []

            # 第一部分是span的
            span_begin = []
            span_end = []

            num_AC = len(eval(sample['adu_spans']))
            meaningless_cnt = 0
            meaningless_num = 1
            for i in range(self.num_span_token):
                
                if i < num_AC:
                    tuple = eval(sample['adu_spans'])[i]
                    span_begin.append(tuple[0])
                    span_end.append(tuple[1])
                    
                else:
                    if meaningless_cnt < meaningless_num:
                        meaningless_cnt += 1
                        span_begin.append(para_length)
                        span_end.append(para_length)
                        
                    else:
                        span_begin.append(-1)
                        span_end.append(-1)

            # 这里是为了后面一个span标签分成两个头拼接的时候方便标签对齐
            label_list.extend(span_begin)
            label_list.extend(span_end)


        
            # 第二部分是ACs的
            mapping = {'Premise': 0, 'Claim': 1, 'MajorClaim': 2, "Support": 0, "Attack": 1, "no relation":2}
            
            ACs_label = eval(sample['AC_types'])
            
            ACs_label_mapping = [mapping[item] for item in ACs_label]
            label_list.extend(ACs_label_mapping)
            label_list.extend([-1] * (self.num_AC_token - len(ACs_label)))
            
            # 第三部分是ARs的
            ARs_types = eval(sample['AR_types'])
            ARs_pairs = eval(sample['AR_pairs'])
            ARs_index_list = []
            for pair in ARs_pairs:
                ARs_index, __ = mapping_ARs_token_index(pair, self.config.dataset)
                ARs_index_list.append(ARs_index)
            
            list_meaningful_ARs_index = get_meaningful_ARs_token(len(ACs_label), self.dataset)
            ARs_label = [-1] * int(self.num_ARs_token)  
            for i in list_meaningful_ARs_index:
                if 0 <= i < len(ARs_label):
                    ARs_label[i] = 2
            label_list.extend(ARs_label)
            for index, type in zip(ARs_index_list, ARs_types):
                if index in list_meaningful_ARs_index:
                    label_list[int(self.num_span_token*2 + self.num_AC_token + index)] = mapping[type]
                else:
                    print("error in ARs")
                    exit()
                
            # EDU的部分
            EDU_graph = sample['EDU_graph']
            EDU_text_cat = sample['EDU_text_cat']
            EDU_index = sample['EDU_index']

            self.data.append({
                'input': input,
                'target': label_list,
                'EDU_graph':EDU_graph,
                'EDU_text_cat':EDU_text_cat,
                'EDU_index':EDU_index
            })
            
        logger.info(f'Loaded {len(self)} instances from {self.data_name}')
        

    def collate_fn(self, batch):
        
        batch_size = len(batch)
        targets = [x['target'] for x in batch]
        vocab_size = self.tokenizer.vocab_size

        prompt_token = torch.arange(vocab_size + self.num_AC_type + self.num_ARs_type, vocab_size + self.num_AC_type + self.num_ARs_type + self.num_prompt_token)
        dec_attn = torch.tensor([1]).repeat(batch_size, int(self.num_prompt_token))
        dec_idxs = prompt_token.repeat(batch_size, 1)
        

        input_text = [x['input'] for x in batch]
        max_length_input = max(len(word_list) for word_list in input_text)
        pad_token = self.tokenizer.pad_token
        enc_word_list = [word_list + [pad_token] * (max_length_input - len(word_list)) for word_list in input_text]
        
                
        enc_idxs = []
        for word_list in enc_word_list:
            word_tokenized = self.tokenizer.convert_tokens_to_ids(word_list)
            enc_idxs.append(word_tokenized)

        enc_idxs = torch.tensor(enc_idxs)
        
        enc_attn = torch.zeros(enc_idxs.size(0), enc_idxs.size(1), dtype=torch.long)

        for i in range(enc_idxs.size(0)):
            for j in range(enc_idxs.size(1)):
                if enc_idxs[i][j] != 1:
                    enc_attn[i][j] = 1
        
        batch_size = enc_idxs.size(0)
        
        try:
            labels = torch.tensor(targets)
            
        except ValueError:
            print("{} : {} ".format(i, len(targets[6])))
            print(targets[6])
            print(batch[6]["input"])
            exit()
        
        
        # EDU的部分
        EDU_graph = [x['EDU_graph'] for x in batch]
        EDU_index = [x['EDU_index'] for x in batch]

        # 给连接的EDU碎片加pad和掩码
        EDU_text_cat = [x['EDU_text_cat'] for x in batch]
        max_length_EDU_text_cat = max(len(EDU_text_) for EDU_text_ in EDU_text_cat)
        enc_EDU_text_cat = [EDU_text_ + [pad_token] * (max_length_EDU_text_cat - len(EDU_text_)) for EDU_text_ in EDU_text_cat]
        EDU_text_cat_idxs = []
        for EDU_text_ in enc_EDU_text_cat:
            word_tokenized = self.tokenizer.convert_tokens_to_ids(EDU_text_)
            EDU_text_cat_idxs.append(word_tokenized)

        EDU_text_cat_idxs = torch.tensor(EDU_text_cat_idxs)
        EDU_text_cat_attn = torch.zeros(EDU_text_cat_idxs.size(0), EDU_text_cat_idxs.size(1), dtype=torch.long)

        for i in range(EDU_text_cat_idxs.size(0)):
            for j in range(EDU_text_cat_idxs.size(1)):
                if EDU_text_cat_idxs[i][j] != 1:
                    EDU_text_cat_attn[i][j] = 1


        max_batch_EDU_length = max(len(x) for x in EDU_index)
        max_batch_EDU_length_ = max(len(x[0]) for x in EDU_graph)
        assert max_batch_EDU_length == max_batch_EDU_length_
        batch_graph = []
        batch_EDU_attn = []
        for i, graph in enumerate(EDU_graph):
            graph_size = len(graph[0])
            graph = np.pad(graph, ((0, 0), (0, max_batch_EDU_length - graph_size),\
                            (0, max_batch_EDU_length - graph_size)), 'constant')
            
            EDU_attn = np.ones((graph_size))
            EDU_attn = np.pad(EDU_attn, (0, max_batch_EDU_length-graph_size), mode='constant')
            
            batch_graph.append(graph)
            batch_EDU_attn.append(EDU_attn)
        
        
        
        enc_idxs = enc_idxs.cuda()
        enc_attn = enc_attn.cuda()
        dec_idxs = dec_idxs.cuda()
        dec_attn = dec_attn.cuda()
        labels = labels.cuda()
        
        EDU_text_cat_idxs = EDU_text_cat_idxs.cuda()
        EDU_text_cat_attn = EDU_text_cat_attn.cuda()
        batch_graph = torch.tensor(batch_graph).cuda()
        batch_EDU_attn = torch.tensor(batch_EDU_attn).cuda()


        return GenBatch(
            enc_idxs=enc_idxs,
            enc_attn=enc_attn,
            dec_idxs=dec_idxs,
            dec_attn=dec_attn,
            lbl_idxs=labels,
            EDU_graph=batch_graph,
            EDU_text_cat_idxs=EDU_text_cat_idxs,
            EDU_text_cat_attn=EDU_text_cat_attn,
            EDU_attn=batch_EDU_attn,
            EDU_index=EDU_index
        )
        
class AbstRCTDataset(Dataset):
    def __init__(self, config, tokenizer, data_name, data_list, unseen_types=[], no_bos=False):
        self.config = config
        self.tokenizer = tokenizer
        self.data_name = data_name
        self.data_list = data_list
        self.no_bos = no_bos # if you use bart, then this should be False; if you use t5, then this should be True
        self.data = []
        
        self.dataset = 'AbstRCT'
        self.num_prompt_token = 11 + 11 + 55
        self.num_span_token = 11
        self.num_AC_token = 11
        self.num_ARs_token = 55
        self.num_AC_type = 3
        self.num_ARs_type = 4
        
        self.load_data(unseen_types)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def load_data(self, unseen_types):
        
        for sample in self.data_list:
            input = sample['para_text'].split()
            para_length = len(input)
            input.insert(0, '<s>')
            input.append('</s>')
            
            # 处理label
            label_list = []

            # 第一部分是span的
            span_begin = []
            span_end = []

            num_AC = len(eval(sample['adu_spans']))
            meaningless_cnt = 0
            meaningless_num = 1
            for i in range(self.num_span_token):
                
                if i < num_AC:
                    tuple = eval(sample['adu_spans'])[i]
                    span_begin.append(tuple[0])
                    span_end.append(tuple[1])
                    
                else:
                    if meaningless_cnt < meaningless_num:
                        meaningless_cnt += 1
                        span_begin.append(para_length)
                        span_end.append(para_length)
                        
                    else:
                        span_begin.append(-1)
                        span_end.append(-1)

            # 这里是为了后面一个span标签分成两个头拼接的时候方便标签对齐
            label_list.extend(span_begin)
            label_list.extend(span_end)


        
            # 第二部分是ACs的
            mapping = {'Evidence': 0, 'Claim': 1, 'MajorClaim': 2, "Support": 0, "Partial-Attack": 1, "no relation":2, "Attack":3}
            
            ACs_label = eval(sample['AC_types'])
            
            ACs_label_mapping = [mapping[item] for item in ACs_label]
            label_list.extend(ACs_label_mapping)
            label_list.extend([-1] * (self.num_AC_token - len(ACs_label)))
            
            # 第三部分是ARs的
            ARs_types = eval(sample['AR_types'])
            ARs_pairs = eval(sample['AR_pairs'])
            ARs_index_list = []
            for pair in ARs_pairs:
                ARs_index, __ = mapping_ARs_token_index(pair, self.config.dataset)
                ARs_index_list.append(ARs_index)
            
            list_meaningful_ARs_index = get_meaningful_ARs_token(len(ACs_label), self.dataset)
            ARs_label = [-1] * int(self.num_ARs_token)  
            for i in list_meaningful_ARs_index:
                if 0 <= i < len(ARs_label):
                    ARs_label[i] = 2
            label_list.extend(ARs_label)
            for index, type in zip(ARs_index_list, ARs_types):
                if index in list_meaningful_ARs_index:
                    label_list[int(self.num_span_token*2 + self.num_AC_token + index)] = mapping[type]
                else:
                    print("error in ARs")
                    exit()
                
            # EDU的部分
            EDU_graph = sample['EDU_graph']
            EDU_text_cat = sample['EDU_text_cat']
            EDU_index = sample['EDU_index']

            self.data.append({
                'input': input,
                'target': label_list,
                'EDU_graph':EDU_graph,
                'EDU_text_cat':EDU_text_cat,
                'EDU_index':EDU_index
            })
            
        logger.info(f'Loaded {len(self)} instances from {self.data_name}')
        

    def collate_fn(self, batch):
        
        batch_size = len(batch)
        targets = [x['target'] for x in batch]
        vocab_size = self.tokenizer.vocab_size

        prompt_token = torch.arange(vocab_size + self.num_AC_type+self.num_ARs_type, vocab_size + self.num_AC_type+self.num_ARs_type + self.num_prompt_token)
        dec_attn = torch.tensor([1]).repeat(batch_size, int(self.num_prompt_token))
        dec_idxs = prompt_token.repeat(batch_size, 1)
        

        input_text = [x['input'] for x in batch]
        max_length_input = max(len(word_list) for word_list in input_text)
        pad_token = self.tokenizer.pad_token
        enc_word_list = [word_list + [pad_token] * (max_length_input - len(word_list)) for word_list in input_text]
        
                
        enc_idxs = []
        for word_list in enc_word_list:
            word_tokenized = self.tokenizer.convert_tokens_to_ids(word_list)
            enc_idxs.append(word_tokenized)

        enc_idxs = torch.tensor(enc_idxs)
        enc_attn = torch.zeros(enc_idxs.size(0), enc_idxs.size(1), dtype=torch.long)

        for i in range(enc_idxs.size(0)):
            for j in range(enc_idxs.size(1)):
                if enc_idxs[i][j] != 1:
                    enc_attn[i][j] = 1
        
        batch_size = enc_idxs.size(0)
        
        try:
            labels = torch.tensor(targets)
            
        except ValueError:
            pass

        
        # EDU的部分
        EDU_graph = [x['EDU_graph'] for x in batch]
        EDU_index = [x['EDU_index'] for x in batch]

        # 给连接的EDU碎片加pad和掩码
        EDU_text_cat = [x['EDU_text_cat'] for x in batch]
        max_length_EDU_text_cat = max(len(EDU_text_) for EDU_text_ in EDU_text_cat)
        enc_EDU_text_cat = [EDU_text_ + [pad_token] * (max_length_EDU_text_cat - len(EDU_text_)) for EDU_text_ in EDU_text_cat]
        EDU_text_cat_idxs = []
        for EDU_text_ in enc_EDU_text_cat:
            word_tokenized = self.tokenizer.convert_tokens_to_ids(EDU_text_)
            EDU_text_cat_idxs.append(word_tokenized)

        EDU_text_cat_idxs = torch.tensor(EDU_text_cat_idxs)
        EDU_text_cat_attn = torch.zeros(EDU_text_cat_idxs.size(0), EDU_text_cat_idxs.size(1), dtype=torch.long)

        for i in range(EDU_text_cat_idxs.size(0)):
            for j in range(EDU_text_cat_idxs.size(1)):
                if EDU_text_cat_idxs[i][j] != 1:
                    # print(enc_word_list[i][j])
                    EDU_text_cat_attn[i][j] = 1


        max_batch_EDU_length = max(len(x) for x in EDU_index)
        max_batch_EDU_length_ = max(len(x[0]) for x in EDU_graph)
        assert max_batch_EDU_length == max_batch_EDU_length_
        batch_graph = []
        batch_EDU_attn = []
        for i, graph in enumerate(EDU_graph):
            graph_size = len(graph[0])
            graph = np.pad(graph, ((0, 0), (0, max_batch_EDU_length - graph_size),\
                            (0, max_batch_EDU_length - graph_size)), 'constant')
            
            EDU_attn = np.ones((graph_size))
            EDU_attn = np.pad(EDU_attn, (0, max_batch_EDU_length-graph_size), mode='constant')
            
            batch_graph.append(graph)
            batch_EDU_attn.append(EDU_attn)
        
        
        
        enc_idxs = enc_idxs.cuda()
        enc_attn = enc_attn.cuda()
        dec_idxs = dec_idxs.cuda()
        dec_attn = dec_attn.cuda()
        labels = labels.cuda()
        
        EDU_text_cat_idxs = EDU_text_cat_idxs.cuda()
        EDU_text_cat_attn = EDU_text_cat_attn.cuda()
        batch_graph = torch.tensor(batch_graph).cuda()
        batch_EDU_attn = torch.tensor(batch_EDU_attn).cuda()


        return GenBatch(
            enc_idxs=enc_idxs,
            enc_attn=enc_attn,
            dec_idxs=dec_idxs,
            dec_attn=dec_attn,
            lbl_idxs=labels,
            EDU_graph=batch_graph,
            EDU_text_cat_idxs=EDU_text_cat_idxs,
            EDU_text_cat_attn=EDU_text_cat_attn,
            EDU_attn=batch_EDU_attn,
            EDU_index=EDU_index
        )
        