import os
import torch
import numpy as np
import argparse
import os
import json
import config
from transformers import AutoTokenizer, AutoModel
from model_depth import ParsingNet

torch.cuda.set_device(0)

relation_mapping = {
    "Attribution": [0, 34],
    "Background":[8, 14],
    "Cause":[2, 3, 6,],
    "Comparison":[19, 22, 37],
    "Condition":[5, 18, 23],
    "Contrast":[15, 21, 33],
    "Elaboration":[7, 10],
    "Enablement":[1, 26],
    "Evaluation":[11, 16, 38],
    "Explanation":[12, 20, 25],
    "Joint":[36],
    "Manner-Means":[30, 35],
    "Same-Unit":[31],
    "Summary":[24, 32, 41],
    "Temporal":[4, 27, 28],
    "TextualOrganization":[13],
    "Topic-Change":[39, 40],
    "Topic-Comment":[9, 17, 29],
}

def parse_args():
    parser = argparse.ArgumentParser()
    """ config the saved checkpoint """
    parser.add_argument('--ModelPath', type=str, default='depth_mode/Savings/multi_all_checkpoint.torchsave', help='pre-trained model')
    base_path = config.tree_infer_mode + "_mode/"
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--savepath', type=str, default=base_path + './Savings', help='Model save path')
    args = parser.parse_args()
    return args


def inference(model, tokenizer, input_sentences, batch_size):
    LoopNeeded = int(np.ceil(len(input_sentences) / batch_size))

    input_sentences = [tokenizer.tokenize(i, add_special_tokens=False) for i in input_sentences]
    all_segmentation_pred = []
    all_tree_parsing_pred = []
    all_logits = []
    with torch.no_grad():
        for loop in range(LoopNeeded):
            StartPosition = loop * batch_size
            EndPosition = (loop + 1) * batch_size
            if EndPosition > len(input_sentences):
                EndPosition = len(input_sentences)

            input_sen_batch = input_sentences[StartPosition:EndPosition]
            _, _, SPAN_batch, _, predict_EDU_breaks, Logits_batch = model.TestingLoss(input_sen_batch, input_EDU_breaks=None, LabelIndex=None,
                                                                        ParsingIndex=None, GenerateTree=True, use_pred_segmentation=True)
            all_segmentation_pred.extend(predict_EDU_breaks)
            all_tree_parsing_pred.extend(SPAN_batch)
            all_logits.extend(Logits_batch)
    return input_sentences, all_segmentation_pred, all_tree_parsing_pred, all_logits


if __name__ == '__main__':

    args = parse_args()
    model_path = args.ModelPath
    batch_size = args.batch_size
    save_path = args.savepath

    """ BERT tokenizer and model """
    bert_tokenizer = AutoTokenizer.from_pretrained("../pretrained_model/xlm-roberta-base", use_fast=True)
    bert_model = AutoModel.from_pretrained("../pretrained_model/xlm-roberta-base")

    bert_model = bert_model.cuda()

    for name, param in bert_model.named_parameters():
        param.requires_grad = False

    model = ParsingNet(bert_model, bert_tokenizer=bert_tokenizer)

    model = model.cuda()
    model.load_state_dict(torch.load(model_path))
    model = model.eval()

    Test_InputSentences = open("./data/abstrct_dev.txt").readlines()

    input_sentences, all_segmentation_pred, all_tree_parsing_pred, Logits_batch = inference(model, bert_tokenizer, Test_InputSentences, batch_size)

    result = []

    assert len(input_sentences)==len(all_segmentation_pred) and len(all_tree_parsing_pred)==len(all_segmentation_pred) and len(input_sentences)==len(all_tree_parsing_pred)
    for i in range(len(input_sentences)):
        a_logit_batch = Logits_batch[i]
        a_relation_logit_mapping = torch.zeros([len(a_logit_batch), 18]).cuda()  
        for j in range(len(a_logit_batch)):
            a_relation_logit = a_logit_batch[j][0]
            for k, mapping_list in enumerate(relation_mapping.values()):
                a_relation_logit_mapping[j, k] = torch.sum(a_relation_logit[[mapping_list]])
        

        result.append({"input_sentences":input_sentences[i],
                        "all_segmentation_pred":all_segmentation_pred[i],
                        "all_tree_parsing_pred":all_tree_parsing_pred[i],
                        "all_relation_logits":a_relation_logit_mapping.tolist(),
                        })

    with open('./output/abstrct_RST_logits_dev.json', 'w') as file:
        json.dump(result, file, indent=4, ensure_ascii=False)
    print("done!")