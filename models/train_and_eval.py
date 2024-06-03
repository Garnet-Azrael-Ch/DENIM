import os, json, logging, time, pprint, tqdm
import sys
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from model import GenerativeModel
from data import PEDataset, AbstRCTDataset, mapping_ARs_token_index
from utils import Summarizer, args_metric, get_meaningful_ARs_turple, eval_component_PE, eval_edge_PE, eval_component_AbstRCT, eval_edge_AbstRCT
from argparse import ArgumentParser, Namespace
import functools
import time
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = ArgumentParser()
parser.add_argument('-c', '--config', default="./config/AAEC.json")
args = parser.parse_args()
with open(args.config) as fp:
    config = json.load(fp)

config = Namespace(**config)

def set_seed(seed):
    # seed init.
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # torch seed init.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
set_seed(config.seed)

# logger and summarizer
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
output_dir = os.path.join(config.output_dir, timestamp)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
log_path = os.path.join(output_dir, "train.log")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]', 
                    handlers=[logging.FileHandler(os.path.join(output_dir, "train.log")), logging.StreamHandler()])
logger = logging.getLogger(__name__)
logger.info(f"\n{pprint.pformat(vars(config), indent=4)}")
summarizer = Summarizer(output_dir)

# set GPU device
torch.cuda.set_device(config.gpu_device)

# output
with open(os.path.join(output_dir, 'config.json'), 'w') as fp:
    json.dump(vars(config), fp, indent=4)
best_model_path = os.path.join(output_dir, 'best_model.mdl')
dev_prediction_path = os.path.join(output_dir, 'pred.dev.json')
test_prediction_path = os.path.join(output_dir, 'pred.test.json')

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=config.cache_dir, use_fast=False, add_prefix_space=True)


no_bos = False
if config.model_name.startswith('t5') or config.model_name.startswith('google/t5'):
    no_bos = True


with open(config.path_data_dir + 'data.train.json', 'r', encoding='utf-8') as fp:
    data_train = json.load(fp)
with open(config.path_data_dir + 'data.test.json', 'r', encoding='utf-8') as fp:
    data_test = json.load(fp)
with open(config.path_data_dir + 'data.dev.json', 'r', encoding='utf-8') as fp:
    data_valid = json.load(fp)

if config.dataset == "AAEC":
    train_set = PEDataset(config, tokenizer, "train", data_train, no_bos=no_bos)
    dev_set = PEDataset(config, tokenizer, "valid", data_valid, no_bos=no_bos)
    test_set = PEDataset(config, tokenizer, "test", data_test, no_bos=no_bos)
elif config.dataset == "AbstRCT":
    train_set = AbstRCTDataset(config, tokenizer, "train", data_train, no_bos=no_bos)
    dev_set = AbstRCTDataset(config, tokenizer, "valid", data_valid, no_bos=no_bos)
    test_set = AbstRCTDataset(config, tokenizer, "test", data_test, no_bos=no_bos)


train_batch_num = len(train_set) // config.train_batch_size + (len(train_set) % config.train_batch_size != 0)
dev_batch_num = len(dev_set) // config.eval_batch_size + (len(dev_set) % config.eval_batch_size != 0)
test_batch_num = len(test_set) // config.eval_batch_size + (len(test_set) % config.eval_batch_size != 0)

# initialize the model
model = GenerativeModel(config, tokenizer)
model.cuda(device=config.gpu_device)


param_groups = [{'params': [p for n, p in model.named_parameters() if "projector" in n], 
                'lr': config.lr_prefix, 'weight_decay': 1e-5},
                {'params': [p for n, p in model.named_parameters() if "model.model.model" in n], 
                'lr': config.lr_bart, 'weight_decay': 1e-5},
                {'params': [p for n, p in model.named_parameters() if all(substring not in n for substring in ['projector','model.model.model'])], 
                'lr': config.lr_other, 'weight_decay': config.weight_decay}]

optimizer = AdamW(params=param_groups)
schedule = get_linear_schedule_with_warmup(optimizer,
                                        num_warmup_steps=train_batch_num*config.warmup_epoch,
                                        num_training_steps=train_batch_num*config.max_epoch)

# start training
logger.info("Start training ...")
summarizer_step = 0
best_dev_epoch = -1
best_dev_scores = {
    'F1_ACS': 0.0,
    'F1_ACTC': 0.0,
    'macro_ACTC': 0.0,
    'F1_ARI': 0.0,
    'F1_ARTC': 0.0,
    'macro_ARTC': 0.0,
    'acc_ACS': 0.0,
    'acc_ACTC': 0.0,
    'acc_ARI': 0.0,
    'acc_ARTC': 0.0,
}

best_dev_test_scores = {}

cnt_stop = 0

loss_result = {
    'sum':[],
    'ACS':[],
    'ACTC':[],
    'ARTC':[],
    'train_time':[],
    'predict_time_dev':[],
    'predict_time_test':[]
}

for epoch in range(1, config.max_epoch+1):
    logger.info(log_path)
    logger.info(f"Epoch {epoch}")
    # training
    progress = tqdm.tqdm(total=train_batch_num, ncols=75, desc='Train {}'.format(epoch))
    model.train()
    optimizer.zero_grad()
    losses = []
    losses_ACS = []
    losses_ACTC = []
    losses_ARTC = []
    if epoch == 50:
        cnt_stop = 0
    elif epoch > 50 and cnt_stop >= config.early_stop:  # 提前停止训练，设置最少训练50轮
        break
    

    start_time = time.time()
    for batch_idx, batch in enumerate(DataLoader(train_set, batch_size=config.train_batch_size // config.accumulate_step, 
                                                shuffle=True, drop_last=False, collate_fn=train_set.collate_fn)):
        # forard model
        loss, loss_acs, loss_actc, loss_artc = model(batch)
        # record loss
        summarizer.scalar_summary('train/loss', loss, summarizer_step)
        summarizer_step += 1
        
        loss = loss * (1 / config.accumulate_step)
        loss.backward()
        losses.append(loss)
        losses_ACS.append(loss_acs)
        losses_ACTC.append(loss_actc)
        losses_ARTC.append(loss_artc)
        if (batch_idx + 1) % config.accumulate_step == 0:
            progress.update(1)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clipping)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
        
        
        
    progress.close()
    end_time = time.time()
    time_result = end_time - start_time
    loss_result['train_time'].append(time_result)
    losses_avg = torch.mean(torch.stack(losses)).tolist()
    losses_ACS_avg = torch.mean(torch.stack(losses_ACS)).tolist()
    losses_ACTC_avg = torch.mean(torch.stack(losses_ACTC)).tolist()
    losses_ARTC_avg = torch.mean(torch.stack(losses_ARTC)).tolist()
    
    loss_result['sum'].append(losses_avg)
    loss_result['ACS'].append(losses_ACS_avg)
    loss_result['ACTC'].append(losses_ACTC_avg)
    loss_result['ARTC'].append(losses_ARTC_avg)
    
    logger.info("Average training loss : {}...".format(losses_avg))

    
    # eval dev set
    progress = tqdm.tqdm(total=dev_batch_num, ncols=75, desc='Dev {}'.format(epoch))
    model.eval()
    best_dev_flag = False
    

    if config.dataset == "AAEC":
        num_prompt_token = 12 + 12 + 66
        num_span_token = 12
        num_AC_token = 12
        num_ARs_token = 66
        num_AC_type = 3
        num_ARs_type = 3
    elif config.dataset == "AbstRCT":
        num_prompt_token = 11 + 11 + 55
        num_span_token = 11
        num_AC_token = 11
        num_ARs_token = 55
        num_AC_type = 3
        num_ARs_type = 4
    else:
        print("The dataset is doesn't exist!")
        raise ValueError
    
    
    
    label_final_ACS = []
    label_final_ACI = []
    label_final_ARI = []
    label_final_ARTC = []
    predict_final_ACS = []
    predict_final_ACI = []
    predict_final_ARI = []
    predict_final_ARTC = []
    best_outputs_predict_dev = []

    start_time = time.time()
    for batch_idx, batch in enumerate(DataLoader(dev_set, batch_size=config.eval_batch_size, 
                                                shuffle=False, collate_fn=dev_set.collate_fn)):
        progress.update(1)
        span_predict, AC_predict, ARs_predict = model.predict(batch)

        
        labels = batch.lbl_idxs
        batch_size = labels.size(0)
        
        # 所有label

        labels_para_begin = labels[:,:num_span_token]
        labels_para_end = labels[:,num_span_token:num_span_token*2]
        labels_AC = labels[:,num_span_token*2: num_span_token*2 + num_AC_token]
        labels_ARs = labels[:,num_span_token*2 + num_AC_token:]
        
        span_predict_begin = span_predict[:,:num_span_token]
        span_predict_end = span_predict[:,num_span_token:num_span_token*2]
            


        labels_AC[labels_AC == -1] = int(num_AC_type)
        
        # 构造ACS的label和prediction，先找到无意义span的索引
        # 取模型预测有意义的span和实际存在的span作为总体
        label_same_begin_end = (labels_para_begin!=labels_para_end)
        flag_label_ACS=False
        
        labels_para_begin_clone = labels_para_begin.clone()
        labels_para_end_clone = labels_para_end.clone()
        span_predict_begin_clone = span_predict_begin.clone()
        span_predict_end_clone = span_predict_end.clone()
        
        for i, row in enumerate(label_same_begin_end):
            true_indices = torch.nonzero(row)
            # 获取最后一个 True 的索引
            if true_indices.numel() > 0:
                last_true_index = true_indices[-1].item()
            else:
                last_true_index = -1
            if last_true_index + 1 < len(row):
                meaningless_span=labels_para_begin[i, last_true_index + 1]
                negative_labels_mask = (labels_para_begin[i] == -1)
                
                labels_para_begin_clone[i, negative_labels_mask] = meaningless_span
                labels_para_end_clone[i, negative_labels_mask] = meaningless_span
                
                a_labels_ACS = ~(labels_para_begin_clone[i]==meaningless_span)
                a_predict_ACS = ~((span_predict_begin[i]==meaningless_span) | (span_predict_end[i]==meaningless_span))
            else:
                # 全部都是有意义的span，都要参与计算
                a_labels_ACS = torch.ones(num_span_token, dtype=torch.bool).cuda()
                a_predict_ACS = ~(span_predict_begin[i]==span_predict_end[i])


            a_label_final_ACS = [(x, y) for x, y in zip(labels_para_begin_clone[i, a_labels_ACS].cpu().numpy().tolist(), labels_para_end_clone[i, a_labels_ACS].cpu().numpy().tolist())]
            a_label_final_ACI = [(x, y, z) for x, y, z in zip(labels_para_begin_clone[i, a_labels_ACS].cpu().numpy().tolist(), labels_para_end_clone[i, a_labels_ACS].cpu().numpy().tolist(), labels_AC[i, a_labels_ACS].cpu().numpy().tolist())]
            a_predict_final_ACS = [(x, y) for x, y in zip(span_predict_begin_clone[i, a_predict_ACS].cpu().numpy().tolist(), span_predict_end_clone[i, a_predict_ACS].cpu().numpy().tolist())]
            a_predict_final_ACI = [(x, y, z) for x, y, z in zip(span_predict_begin_clone[i, a_predict_ACS].cpu().numpy().tolist(), span_predict_end_clone[i, a_predict_ACS].cpu().numpy().tolist(), AC_predict[i, a_predict_ACS].cpu().numpy().tolist())]
            
            # AR
            label_meaningful_ARs_turple = get_meaningful_ARs_turple(len(a_label_final_ACS), dataset=config.dataset)
            predict_meaningful_ARs_turple = get_meaningful_ARs_turple(len(a_predict_final_ACS), dataset=config.dataset)
            
            mapping_ARs_token_index_ = functools.partial(mapping_ARs_token_index, dataset=config.dataset)

            
            a_label_final_ARI = []
            a_label_final_ARTC = []
            for pair in label_meaningful_ARs_turple:
                pair_index = mapping_ARs_token_index_(pair)[0]
                relation = labels_ARs[i].cpu().numpy().tolist()[pair_index]
                if relation == -1:
                    raise ValueError
                if relation != 2:
                    a_label_final_ARI.append((labels_para_begin_clone[i].cpu().numpy().tolist()[pair[0]], 
                                            labels_para_end_clone[i].cpu().numpy().tolist()[pair[0]],
                                            labels_para_begin_clone[i].cpu().numpy().tolist()[pair[1]],
                                            labels_para_end_clone[i].cpu().numpy().tolist()[pair[1]],
                                            ))
                    a_label_final_ARTC.append((labels_para_begin_clone[i].cpu().numpy().tolist()[pair[0]], 
                                            labels_para_end_clone[i].cpu().numpy().tolist()[pair[0]],
                                            labels_para_begin_clone[i].cpu().numpy().tolist()[pair[1]],
                                            labels_para_end_clone[i].cpu().numpy().tolist()[pair[1]],
                                            relation))
            a_predict_final_ARI = []
            a_predict_final_ARTC = []
            for pair in predict_meaningful_ARs_turple:
                pair_index = mapping_ARs_token_index_(pair)[0]
                relation = ARs_predict[i].cpu().numpy().tolist()[pair_index]
                if relation == -1:
                    raise ValueError
                if relation != 2:
                    a_predict_final_ARI.append((span_predict_begin_clone[i].cpu().numpy().tolist()[pair[0]], 
                                            span_predict_end_clone[i].cpu().numpy().tolist()[pair[0]],
                                            span_predict_begin_clone[i].cpu().numpy().tolist()[pair[1]],
                                            span_predict_end_clone[i].cpu().numpy().tolist()[pair[1]],
                                            ))
                    a_predict_final_ARTC.append((span_predict_begin_clone[i].cpu().numpy().tolist()[pair[0]], 
                                            span_predict_end_clone[i].cpu().numpy().tolist()[pair[0]],
                                            span_predict_begin_clone[i].cpu().numpy().tolist()[pair[1]],
                                            span_predict_end_clone[i].cpu().numpy().tolist()[pair[1]],
                                            relation
                                            ))
                
                                
            
            label_final_ACS.append(a_label_final_ACS)
            label_final_ACI.append(a_label_final_ACI)
            label_final_ARI.append(a_label_final_ARI)
            label_final_ARTC.append(a_label_final_ARTC)
            predict_final_ACS.append(a_predict_final_ACS)
            predict_final_ACI.append(a_predict_final_ACI)
            predict_final_ARI.append(a_predict_final_ARI)
            predict_final_ARTC.append(a_predict_final_ARTC)

            
    outputs_result = []
    for i, __ in enumerate(label_final_ACS):
        an_output_result = {}
        an_output_result['essay_id'] = data_valid[i]['essay_id']
        an_output_result['para_text'] = data_valid[i]['para_text']
        an_output_result['adu_spans'] = data_valid[i]['adu_spans']
        an_output_result['AC_types'] = data_valid[i]['AC_types']
        an_output_result['AR_pairs'] = data_valid[i]['AR_pairs']
        an_output_result['AR_types'] = data_valid[i]['AR_types']
        an_output_result['predict_ACS'] = repr(predict_final_ACS[i])
        an_output_result['predict_ACI'] = repr(predict_final_ACI[i])
        an_output_result['predict_ARI'] = repr(predict_final_ARI[i])
        an_output_result['predict_ARTC'] = repr(predict_final_ARTC[i])
        outputs_result.append(an_output_result)
        
    result_ACS = args_metric(label_final_ACS, predict_final_ACS)
    result_ACI = args_metric(label_final_ACI, predict_final_ACI)
    result_ARI = args_metric(label_final_ARI, predict_final_ARI)
    result_ARTC = args_metric(label_final_ARTC, predict_final_ARTC)   
    f1_ACS = result_ACS['f1']
    f1_ACI = result_ACI['f1']
    if config.dataset == "AAEC": 
        result_ACI_macro = eval_component_PE(predict_final_ACI, label_final_ACI) 
        macro_ACI = (result_ACI_macro["Premise"]['f'] + result_ACI_macro["Claim"]['f'] + result_ACI_macro["MajorClaim"]['f'])/3
    elif config.dataset == "AbstRCT": 
        result_ACI_macro = eval_component_AbstRCT(predict_final_ACI, label_final_ACI) 
        macro_ACI = (result_ACI_macro["Evidence"]['f'] + result_ACI_macro["Claim"]['f'] + result_ACI_macro["MajorClaim"]['f'])/3

    else:
        raise ValueError
    
    f1_ARI = result_ARI['f1']
    f1_ARTC = result_ARTC['f1']
    
    
    if config.dataset == "AAEC":  
        result_ARTC_macro = eval_edge_PE(predict_final_ARTC, label_final_ARTC)
        macro_ARTC = (result_ARTC_macro['Support']['f'] + result_ARTC_macro['Attack']['f'])/2
    elif config.dataset == "AbstRCT":  
        result_ARTC_macro = eval_edge_AbstRCT(predict_final_ARTC, label_final_ARTC)
        macro_ARTC = (result_ARTC_macro['Support']['f'] + result_ARTC_macro['Partial-Attack']['f'] + result_ARTC_macro['Attack']['f'])/3
    else:
        raise ValueError

    acc_ACS = result_ACS['acc']
    acc_ACI = result_ACI['acc']
    acc_ARI = result_ARI['acc']
    acc_ARTC = result_ARTC['acc']
    
    progress.close()
    end_time = time.time()
    time_result = end_time - start_time
    loss_result['predict_time_dev'].append(time_result)

    dev_scores = {
        'F1_ACS': f1_ACS,
        'F1_ACTC': f1_ACI,
        'macro_ACTC': macro_ACI,
        'F1_ARI': f1_ARI,
        'F1_ARTC': f1_ARTC,
        'macro_ARTC': macro_ARTC,
        'acc_ACS': acc_ACS,
        'acc_ACTC': acc_ACI,
        'acc_ARI': acc_ARI,
        'acc_ARTC': acc_ARTC
    }
    
    
    
    logger.info("--------------------------Dev Scores---------------------------------")
    logger.info('ACS - F: {:5.2f}'.format(
        dev_scores['F1_ACS'] * 100.0))
    logger.info('ACTC - F: {:5.2f}'.format(
        dev_scores['F1_ACTC'] * 100.0))
    logger.info('ACTC - Macro: {:5.2f}'.format(
        dev_scores['macro_ACTC'] * 100.0))
    logger.info('ARI - F: {:5.2f}'.format(
        dev_scores['F1_ARI'] * 100.0))
    logger.info('ARTC - F: {:5.2f}'.format(
        dev_scores['F1_ARTC'] * 100.0))
    logger.info('ARTC - Macro: {:5.2f}'.format(
        dev_scores['macro_ARTC'] * 100.0))
    logger.info('ACS - acc: {:5.2f}'.format(
        dev_scores['acc_ACS'] * 100.0))
    logger.info('ACTC - acc: {:5.2f}'.format(
        dev_scores['acc_ACTC'] * 100.0))
    logger.info('ARI - acc: {:5.2f}'.format(
        dev_scores['acc_ARI'] * 100.0))
    logger.info('ARTC - acc: {:5.2f}'.format(
        dev_scores['acc_ARTC'] * 100.0))
    logger.info("---------------------------------------------------------------------")
    
    # check best dev model
    dev_score_sum = dev_scores['F1_ACS'] + dev_scores['F1_ACTC'] + dev_scores['F1_ARI'] + dev_scores['F1_ARTC']
    best_dev_score_sum = best_dev_scores['F1_ACS'] + best_dev_scores['F1_ACTC'] + best_dev_scores['F1_ARI'] + best_dev_scores['F1_ARTC']
    if dev_score_sum > best_dev_score_sum:
        best_dev_flag = True
        cnt_stop = 0
    else:
        cnt_stop += 1
        
    # if best dev, save model and evaluate test set
    if best_dev_flag:    
        best_dev_scores = dev_scores
        best_dev_epoch = epoch
    
        # save best model
        logger.info('Saving best model')
        torch.save(model.state_dict(), best_model_path)
        
        # save dev result
        best_outputs_predict_dev = outputs_result
        with open(dev_prediction_path, 'w', encoding='utf-8') as fp:
            json.dump(best_outputs_predict_dev, fp, indent=4, ensure_ascii=False)
            
    if True:
        # eval test set
        start_time = time.time()
        progress = tqdm.tqdm(total=test_batch_num, ncols=75, desc='Test {}'.format(epoch))
        

        test_label_final_ACS = []
        test_label_final_ACI = []
        test_label_final_ARI = []
        test_label_final_ARTC = []
        test_predict_final_ACS = []
        test_predict_final_ACI = []
        test_predict_final_ARI = []
        test_predict_final_ARTC = []
        
        
        for batch_idx, batch in enumerate(DataLoader(test_set, batch_size=config.eval_batch_size, 
                                                    shuffle=False, collate_fn=test_set.collate_fn)):
            progress.update(1)
            
            test_span_predict, test_AC_predict, test_ARs_predict = model.predict(batch)


            test_labels = batch.lbl_idxs
            test_batch_size = test_labels.size(0)
            
            # 所有label

            test_labels_para_begin = test_labels[:,:num_span_token]
            test_labels_para_end = test_labels[:,num_span_token:num_span_token*2]
            test_labels_AC = test_labels[:,num_span_token*2: num_span_token*2 + num_AC_token]
            test_labels_ARs = test_labels[:,num_span_token*2 + num_AC_token:]
        
            test_span_predict_begin = test_span_predict[:,:num_span_token]
            test_span_predict_end = test_span_predict[:,num_span_token:num_span_token*2]
                
                
            
            test_labels_AC[test_labels_AC == -1] = int(num_AC_type)
            
            # 构造ACS的label和prediction，先找到无意义span的索引
            # 取模型预测有意义的span和实际存在的span作为总体
            test_label_same_begin_end = (test_labels_para_begin!=test_labels_para_end)
            test_flag_label_ACS=False
            
            test_labels_para_begin_clone = test_labels_para_begin.clone()
            test_labels_para_end_clone = test_labels_para_end.clone()
            test_span_predict_begin_clone = test_span_predict_begin.clone()
            test_span_predict_end_clone = test_span_predict_end.clone()
            for i, test_row in enumerate(test_label_same_begin_end):
                test_true_indices = torch.nonzero(test_row)
                # 获取最后一个 True 的索引
                if test_true_indices.numel() > 0:
                    test_last_true_index = test_true_indices[-1].item()
                else:
                    test_last_true_index = -1
                    
                if test_last_true_index + 1 < len(test_row):
                    test_meaningless_span=test_labels_para_begin[i, test_last_true_index + 1]
                    test_negative_labels_mask = (test_labels_para_begin[i] == -1)
                    
                    test_labels_para_begin_clone[i, test_negative_labels_mask] = test_meaningless_span
                    test_labels_para_end_clone[i, test_negative_labels_mask] = test_meaningless_span
                    
                    test_a_labels_ACS = ~(test_labels_para_begin_clone[i]==test_meaningless_span)
                    test_a_predict_ACS = ~((test_span_predict_begin[i]==test_meaningless_span) | (test_span_predict_end[i]==test_meaningless_span))
                else:
                    # 全部都是有意义的span，都要参与计算
                    test_a_labels_ACS = torch.ones(num_span_token, dtype=torch.bool).cuda()
                    test_a_predict_ACS = ~(test_span_predict_begin[i]==test_span_predict_end[i])
                    
                
                test_a_label_final_ACS = [(x, y) for x, y in zip(test_labels_para_begin_clone[i, test_a_labels_ACS].cpu().numpy().tolist(), test_labels_para_end_clone[i, test_a_labels_ACS].cpu().numpy().tolist())]
                test_a_label_final_ACI = [(x, y, z) for x, y, z in zip(test_labels_para_begin_clone[i, test_a_labels_ACS].cpu().numpy().tolist(), test_labels_para_end_clone[i, test_a_labels_ACS].cpu().numpy().tolist(), test_labels_AC[i, test_a_labels_ACS].cpu().numpy().tolist())]
                test_a_predict_final_ACS = [(x, y) for x, y in zip(test_span_predict_begin_clone[i, test_a_predict_ACS].cpu().numpy().tolist(), test_span_predict_end_clone[i, test_a_predict_ACS].cpu().numpy().tolist())]
                test_a_predict_final_ACI = [(x, y, z) for x, y, z in zip(test_span_predict_begin_clone[i, test_a_predict_ACS].cpu().numpy().tolist(), test_span_predict_end_clone[i, test_a_predict_ACS].cpu().numpy().tolist(), test_AC_predict[i, test_a_predict_ACS].cpu().numpy().tolist())]
                
                
                # AR
                test_label_meaningful_ARs_turple = get_meaningful_ARs_turple(len(test_a_label_final_ACS), dataset=config.dataset)
                test_predict_meaningful_ARs_turple = get_meaningful_ARs_turple(len(test_a_predict_final_ACS), dataset=config.dataset)
                
                mapping_ARs_token_index_ = functools.partial(mapping_ARs_token_index, dataset=config.dataset)
                
                test_a_label_final_ARI = []
                test_a_label_final_ARTC = []
                for test_pair in test_label_meaningful_ARs_turple:
                    test_pair_index = mapping_ARs_token_index_(test_pair)[0]
                    test_relation = test_labels_ARs[i].cpu().numpy().tolist()[test_pair_index]
                    if test_relation == -1:
                        raise ValueError
                    if test_relation != 2:
                        test_a_label_final_ARI.append((test_labels_para_begin_clone[i].cpu().numpy().tolist()[test_pair[0]], 
                                                    test_labels_para_end_clone[i].cpu().numpy().tolist()[test_pair[0]],
                                                    test_labels_para_begin_clone[i].cpu().numpy().tolist()[test_pair[1]],
                                                    test_labels_para_end_clone[i].cpu().numpy().tolist()[test_pair[1]],
                                                    ))
                        test_a_label_final_ARTC.append((test_labels_para_begin_clone[i].cpu().numpy().tolist()[test_pair[0]], 
                                                    test_labels_para_end_clone[i].cpu().numpy().tolist()[test_pair[0]],
                                                    test_labels_para_begin_clone[i].cpu().numpy().tolist()[test_pair[1]],
                                                    test_labels_para_end_clone[i].cpu().numpy().tolist()[test_pair[1]],
                                                    test_relation
                                                    ))
                test_a_predict_final_ARI = []
                test_a_predict_final_ARTC = []
                for test_pair in test_predict_meaningful_ARs_turple:
                    test_pair_index = mapping_ARs_token_index_(test_pair)[0]
                    test_relation = test_ARs_predict[i].cpu().numpy().tolist()[test_pair_index]
                    if test_relation == -1:
                        raise ValueError
                    if test_relation != 2:
                        test_a_predict_final_ARI.append((test_span_predict_begin_clone[i].cpu().numpy().tolist()[test_pair[0]], 
                                                        test_span_predict_end_clone[i].cpu().numpy().tolist()[test_pair[0]],
                                                        test_span_predict_begin_clone[i].cpu().numpy().tolist()[test_pair[1]],
                                                        test_span_predict_end_clone[i].cpu().numpy().tolist()[test_pair[1]],
                                                        ))
                        test_a_predict_final_ARTC.append((test_span_predict_begin_clone[i].cpu().numpy().tolist()[test_pair[0]], 
                                                        test_span_predict_end_clone[i].cpu().numpy().tolist()[test_pair[0]],
                                                        test_span_predict_begin_clone[i].cpu().numpy().tolist()[test_pair[1]],
                                                        test_span_predict_end_clone[i].cpu().numpy().tolist()[test_pair[1]],
                                                        test_relation
                                                        ))
                    

                test_label_final_ACS.append(test_a_label_final_ACS)
                test_label_final_ACI.append(test_a_label_final_ACI)
                test_label_final_ARI.append(test_a_label_final_ARI)
                test_label_final_ARTC.append(test_a_label_final_ARTC)
                test_predict_final_ACS.append(test_a_predict_final_ACS)
                test_predict_final_ACI.append(test_a_predict_final_ACI)
                test_predict_final_ARI.append(test_a_predict_final_ARI)
                test_predict_final_ARTC.append(test_a_predict_final_ARTC)
                

        outputs_result_test = []
        for i, __ in enumerate(test_label_final_ACS):
            an_output_result_test = {}
            an_output_result_test['essay_id'] = data_test[i]['essay_id']
            an_output_result_test['para_text'] = data_test[i]['para_text']
            an_output_result_test['adu_spans'] = data_test[i]['adu_spans']
            an_output_result_test['AC_types'] = data_test[i]['AC_types']
            an_output_result_test['AR_pairs'] = data_test[i]['AR_pairs']
            an_output_result_test['AR_types'] = data_test[i]['AR_types']
            an_output_result_test['predict_ACS'] = repr(test_predict_final_ACS[i])
            an_output_result_test['predict_ACI'] = repr(test_predict_final_ACI[i])
            an_output_result_test['predict_ARI'] = repr(test_predict_final_ARI[i])
            an_output_result_test['predict_ARTC'] = repr(test_predict_final_ARTC[i])
            outputs_result_test.append(an_output_result_test)
        
        test_result_ACS = args_metric(test_label_final_ACS, test_predict_final_ACS)
        test_result_ACI = args_metric(test_label_final_ACI, test_predict_final_ACI)
        test_result_ARI = args_metric(test_label_final_ARI, test_predict_final_ARI)
        test_result_ARTC = args_metric(test_label_final_ARTC, test_predict_final_ARTC)
        
        
        test_f1_ACS = test_result_ACS['f1']
        test_f1_ACI = test_result_ACI['f1']
        if config.dataset == "AAEC":  
            test_result_ACI_macro = eval_component_PE(test_predict_final_ACI, test_label_final_ACI)
            test_macro_ACI = (test_result_ACI_macro["Premise"]['f'] + test_result_ACI_macro["Claim"]['f'] + test_result_ACI_macro["MajorClaim"]['f'])/3
        elif config.dataset == "AbstRCT":  
            test_result_ACI_macro = eval_component_AbstRCT(test_predict_final_ACI, test_label_final_ACI)
            test_macro_ACI = (test_result_ACI_macro["Evidence"]['f'] + test_result_ACI_macro["Claim"]['f'] + test_result_ACI_macro["MajorClaim"]['f'])/3
        else:
            raise ValueError

        test_f1_ARI = test_result_ARI['f1']
        test_f1_ARTC = test_result_ARTC['f1']
        if config.dataset == "AAEC":
            test_result_ARTC_macro = eval_edge_PE(test_predict_final_ARTC, test_label_final_ARTC)
            test_macro_ARTC = (test_result_ARTC_macro['Support']['f'] + test_result_ARTC_macro['Attack']['f'])/2
        elif config.dataset == "AbstRCT":
            test_result_ARTC_macro = eval_edge_AbstRCT(test_predict_final_ARTC, test_label_final_ARTC)
            test_macro_ARTC = (test_result_ARTC_macro['Support']['f'] + test_result_ARTC_macro['Partial-Attack']['f'] + test_result_ARTC_macro['Attack']['f'])/3
        else:
            raise ValueError
        test_acc_ACS = test_result_ACS['acc']
        test_acc_ACI = test_result_ACI['acc']
        test_acc_ARI = test_result_ARI['acc']
        test_acc_ARTC = test_result_ARTC['acc']
        
        progress.close()
        end_time = time.time()
        time_result = end_time - start_time
        loss_result['predict_time_test'].append(time_result)
        
        test_scores = {
            'F1_ACS': test_f1_ACS,
            'F1_ACTC': test_f1_ACI,
            'macro_ACTC': test_macro_ACI,
            'F1_ARI': test_f1_ARI,
            'F1_ARTC': test_f1_ARTC,
            'macro_ARTC': test_macro_ARTC,
            'acc_ACS': test_acc_ACS,
            'acc_ACTC': test_acc_ACI,
            'acc_ARI': test_acc_ARI,
            'acc_ARTC': test_acc_ARTC
        }
        if best_dev_flag:
            best_dev_test_scores = test_scores
        
        # print scores
        logger.info("--------------------------TEST Scores---------------------------------")
        logger.info('ACS - F: {:5.2f}'.format(
            test_scores['F1_ACS'] * 100.0))
        logger.info('ACTC - F: {:5.2f}'.format(
            test_scores['F1_ACTC'] * 100.0))
        logger.info('ACTC - Macro: {:5.2f}'.format(
            test_scores['macro_ACTC'] * 100.0))
        logger.info('ARI - F: {:5.2f}'.format(
            test_scores['F1_ARI'] * 100.0))
        logger.info('ARTC - F: {:5.2f}'.format(
            test_scores['F1_ARTC'] * 100.0))
        logger.info('ARTC - Macro: {:5.2f}'.format(
            test_scores['macro_ARTC'] * 100.0))
        logger.info('ACS - acc: {:5.2f}'.format(
            test_scores['acc_ACS'] * 100.0))
        logger.info('ACTC - acc: {:5.2f}'.format(
            test_scores['acc_ACTC'] * 100.0))
        logger.info('ARI - acc: {:5.2f}'.format(
            test_scores['acc_ARI'] * 100.0))
        logger.info('ARTC - acc: {:5.2f}'.format(
            test_scores['acc_ARTC'] * 100.0))
        logger.info("---------------------------------------------------------------------")
            
        # save test result
    if best_dev_flag:
        with open(test_prediction_path, 'w', encoding='utf-8') as fp:
            json.dump(outputs_result_test, fp, indent=4, ensure_ascii=False)
            
    logger.info({"epoch": epoch, "dev_scores": dev_scores})
    logger.info("Current best:-----")
    logger.info({"best_dev_epoch": best_dev_epoch, "best_dev_scores": best_dev_scores})
    logger.info({"test_score_in_best_dev": best_dev_test_scores})
    logger.info("------------------\n\n")
    
    df = pd.DataFrame(loss_result)
    df.to_excel(output_dir+'/time_and_loss_'+config.dataset+'.xlsx', index=False)

logger.info("train complete!\nResults:")
logger.info({"best_dev_epoch": best_dev_epoch, "best_dev_scores": best_dev_scores})
logger.info({"test_score_in_best_dev": best_dev_test_scores})
