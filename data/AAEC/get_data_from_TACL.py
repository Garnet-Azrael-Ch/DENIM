import json
from transformers import BartTokenizer
import string


def get_data(dataset_type):
    mrp_file_path = './data/AAEC/aaec_para_' +dataset_type+ '.mrp'  # 请替换为你的文件路径
    output_json_file_path = './data/AAEC/aaec_' + dataset_type+ '.json'

    # 读取MRP文件并将每个字典添加到列表中
    with open(mrp_file_path, 'r', encoding='utf-8') as file:
        mrp_list = [json.loads(line.strip()) for line in file]

    bart_path = "./pretrained_model/bart-base"
    bart_tokenizer = BartTokenizer.from_pretrained(bart_path, add_prefix_space=True)

    ac_label_dict = {'AAECPARA_Premise': 'Premise',
                    'AAECPARA_MajorClaim': 'MajorClaim',
                    'AAECPARA_Claim:For': 'Claim',
                    'AAECPARA_Claim:Against': 'Claim',
    }

    ar_label_dict = {'AAECPARA_supports': 'Support',
                    'AAECPARA_attacks': 'Attack',
    }

    def is_punctuation(char):
        return char in string.punctuation


    dataset = []
    for sample in mrp_list:
        a_data = {}
        try:
            essay_id = sample['id'][5:8]
            para_id = sample['id'][9]
            a_data['essay_id'] = str(int(essay_id))
            a_data['para_id'] = str(int(para_id))
        except:
            raise ValueError
        
        text = sample['input']
        len_text = len(text)
        ACs = sample['nodes']
        ARs = sample['edges']
        token_id = 0
        str_id = 0
        para_text = ""
        adu_spans = []
        AC_types = []
        AR_pairs = []
        AR_types = []
        for a_AC in ACs:

            span = a_AC['anchors']
            assert len(span) == 1
            span_s = span[0]['from']
            span_e = span[0]['to']
            if span_s != str_id:
                if is_punctuation(text[str_id]):
                    tokens = bart_tokenizer.tokenize(text[str_id:span_s].replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"').strip(), add_prefix_space=False)
                else:
                    tokens = bart_tokenizer.tokenize(text[str_id:span_s].replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"').strip(), add_prefix_space=True)
                token_id += len(tokens)
                para_text += ' '.join(tokens) + ' '
                str_id += len(text[str_id:span_s])
            tokens = bart_tokenizer.tokenize(text[span_s:span_e].replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"').strip(), add_prefix_space=True)
            adu_spans.append([token_id, token_id + len(tokens)-1])
            token_id += len(tokens)
            para_text += ' '.join(tokens) + ' '
            str_id += len(text[span_s:span_e])

            try:
                AC_types.append(ac_label_dict[a_AC['label']])
            except:
                raise ValueError
        
        if str_id < len_text:
            if is_punctuation(text[str_id]):
                tokens = bart_tokenizer.tokenize(text[str_id:].replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"').strip(), add_prefix_space=False)
            else:
                tokens = bart_tokenizer.tokenize(text[str_id:].replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"').strip(), add_prefix_space=True)
            para_text += ' '.join(tokens) + ' '
        elif str_id == len_text:
            pass
        else:
            raise ValueError
        
        for a_AR in ARs:
            ar_src = a_AR['source']
            ar_tgt = a_AR['target']
            AR_pairs.append([ar_src, ar_tgt])
            AR_types.append(ar_label_dict[a_AR['label']])
            
        
        a_data['para_text'] = para_text.strip()
        a_data['adu_spans'] = adu_spans
        a_data['AC_types'] = AC_types
        a_data['AR_pairs'] = AR_pairs
        a_data['AR_types'] = AR_types
        dataset.append(a_data)


    dropped_list = []
    dropped_ac = 0
    dropped_rel = 0

    max_len_for_bart = 1024 - 2 # <s> </s>
    for a_data in dataset:
        para_id = a_data['para_id']
        para_text = a_data['para_text']
        adu_spans= a_data['adu_spans']
        ac_types= a_data['AC_types']
        ac_rel_pairs= a_data['AR_pairs']
        ac_rel_types= a_data['AR_types']
        
        a_data['para_text'] = para_text
        a_data['adu_spans'] = repr(adu_spans)
        a_data['AC_types'] = repr(ac_types)
        a_data['AR_pairs'] = repr(ac_rel_pairs)
        a_data['AR_types'] = repr(ac_rel_types)
        
        a_data['orig_para_text'] = para_text
        a_data['orig_adu_spans'] = repr(adu_spans)
        a_data['orig_AC_types'] = repr(ac_types)
        a_data['orig_AR_pairs'] = repr(ac_rel_pairs)
        a_data['orig_AR_types'] = repr(ac_rel_types)
        
        dropped_id = {}
        if len(para_text.split(' ')) > max_len_for_bart or len(adu_spans) >= 12: 
            exceeded_idx = len(adu_spans)
            for i, span in enumerate(adu_spans):
                if span[0] > max_len_for_bart or span[1] > max_len_for_bart or i >= 12:
                    dropped_ac += (len(adu_spans) - i)
                    dropped_id['para_id'] = para_id
                    dropped_id['num_dropped_ac'] = (len(adu_spans) - i)
                    exceeded_idx = i
                    break
                
            new_para_text = para_text
            new_adu_spans = adu_spans[:exceeded_idx]
            new_ac_types = ac_types[:exceeded_idx]
            new_ac_rel_pairs = []
            new_ac_rel_types = []

            new_para_text = " ".join(new_para_text.split()[:new_adu_spans[-1][1]+1])
            
            dropped_id['num_dropped_rel'] = 0
            for (pairs, types) in zip(ac_rel_pairs, ac_rel_types):
                if pairs[0] < exceeded_idx and pairs[1] < exceeded_idx:
                    new_ac_rel_pairs.append(pairs)
                    new_ac_rel_types.append(types)
                else:
                    dropped_rel += 1
                    dropped_id['num_dropped_rel'] += 1
            
            
            a_data['para_text'] = new_para_text
            a_data['adu_spans'] = repr(new_adu_spans)
            a_data['AC_types'] = repr(new_ac_types)
            a_data['AR_pairs'] = repr(new_ac_rel_pairs)
            a_data['AR_types'] = repr(new_ac_rel_types)
            dropped_list.append(dropped_id)
            
            

    with open(output_json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(dataset, json_file, indent=4, ensure_ascii=False)
    print(dataset_type)
    print("total_dropped_ac: "+ str(dropped_ac))
    print("total_dropped_rel: "+ str(dropped_rel))
    print("detail_dropped: " + repr(dropped_list))
    print('\n\n')

if __name__ == '__main__':
    get_data(dataset_type='dev')
    get_data(dataset_type='test')
    get_data(dataset_type='train')