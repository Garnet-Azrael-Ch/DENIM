import json
import numpy as np
from collections import OrderedDict
import re
from transformers import BartTokenizer


def get_data(data_kind):
    with open('./data/AAEC/aaec_'+data_kind+'.json', 'r') as file:
        data_pe = json.load(file)

    with open('./data/AAEC/aaec_RST_logits_' + data_kind + '.json', 'r') as file:
        data = json.load(file)

    relation_dict={
        'Association':['Joint', "Contrast", "Temporal", "TextualOrganization", "Topic-Change"],
        'cause':['Cause', 'Explanation'],
        'Expansion': ['Background', 'Elaboration', 'Evaluation', 'Summary', "Topic-Comment"],
    }

    relation_mapping = {
        'Association': [5, 10, 14, 15, 16],
        'cause': [2, 9],
        "Expansion": [1, 6, 8, 13, 17]
    }

    def find_spans(numbers):
        # 找到可以合并EDU_text的索引
        spans = []
        start = numbers[0]
        end = numbers[0]

        for num in numbers[1:]:
            if num - end == 1:
                end = num
            else:
                spans.append((start, end+1))
                start = end = num

        spans.append((start, end+1))
        return spans

    def get_RST_category(chip1, chip2):
        centrality1 = chip1[1]
        centrality2 = chip2[1]
        assert centrality1 in ['Nucleus','Satellite']
        assert centrality2 in ['Nucleus','Satellite']
        if centrality1 == 'Nucleus' and centrality2 == 'Nucleus':
            category = chip1[2]
            if chip1[2] != chip2[2]:
                raise ValueError
        elif centrality1 == 'Nucleus' and centrality2 == 'Satellite':
            category = chip2[2]
            if chip1[2] != 'span':
                raise ValueError
        elif centrality1 == 'Satellite' and centrality2 == 'Nucleus':
            category = chip1[2]
            if chip2[2] != 'span':
                raise ValueError
        else:
            raise ValueError
        return category

    def update_RST_tree_span(to_update_index, merged_index):
        # 更新合并后的RST_tree元组的值
        if to_update_index >= merged_index:
            return to_update_index - 1
        else:
            return to_update_index


    def get_relation(category):
        flag_if_category = False
        cate = 0
        for key_index, value in enumerate(relation_dict.values()):
            if category in value:
                flag_if_category = True
                cate = key_index
                break
        return cate, flag_if_category
            
    data_pe_new = []

    paragraphs = data
    max_num = 0
    # EDU_graph_all = []
    # EDU_list_all = []
    for id, paragraph in enumerate(paragraphs):
        a_pe_data = data_pe[id]
        
        sentence_tokens = paragraph['input_sentences']
        segmentation_pred = paragraph['all_segmentation_pred']
        assert len(paragraph['all_tree_parsing_pred'])==1
        tree_parsing_pred = paragraph['all_tree_parsing_pred'][0]
        logits = paragraph['all_relation_logits']

        # 处理句子tokens列表，分成句子段落
        start = 0
        EDU_text = []
        for end in segmentation_pred:
            sentence = sentence_tokens[start:end + 1]
            EDU_text.append("".join(sentence).replace("▁", " ").strip())
            start = end + 1

        # Add the remaining words as the last sentence
        len_input_sentences = len(sentence_tokens)
        if start < len_input_sentences:
            remaining_words = sentence_tokens[start:]
            EDU_text.append("".join(remaining_words).replace("▁", " ").strip())


        # 处理句子的关系树
        contents = re.findall(r'\(([^)]+)\)', tree_parsing_pred)
        RST_tree = []
        for item in contents:
            sub_items = item.split(',')
            sub_result = []
            for sub_item in sub_items:
                sub_parts = re.split(r'[:=]', sub_item)
                sub_result.append(tuple(sub_parts))
            RST_tree.append(tuple(sub_result))


        # 先进行一个合并操作，从底层往上合并，重复迭代
        if RST_tree:
            flag_update_over = False # 判断是不是没有可以合并的项了
            while(not flag_update_over):
                if RST_tree:
                    for idx, relation in enumerate(RST_tree):
                        flag_update_over = True
                        chip1 = relation[0]
                        chip2 = relation[1]
                        if chip1[0] != chip1[3] or chip2[0] != chip2[3]:
                            continue
                        category = get_RST_category(chip1, chip2)
                        __, flag_if_category = get_relation(category) 
                        
                        if not flag_if_category:
                            RST_tree.pop(idx)
                            logits.pop(idx)
                            # 先合并text
                            flag_update_over = False
                            text1 = EDU_text.pop(int(chip1[0])-1)
                            text2 = EDU_text.pop(int(chip1[0])-1)
                            text_merged = text1 + " " +text2
                            EDU_text.insert(int(chip1[0])-1, text_merged)
                            new_RST_tree = []
                            new_logits = []
                            for i, to_update_relation in enumerate(RST_tree):
                                to_update_chip1 = to_update_relation[0]
                                to_update_chip2 = to_update_relation[1]
                                
                                to_update_chip1_start = str(update_RST_tree_span(int(to_update_chip1[0]), int(chip2[0])))
                                to_update_chip1_end = str(update_RST_tree_span(int(to_update_chip1[3]), int(chip2[0])))
                                new_chip1 = (to_update_chip1_start, to_update_chip1[1], to_update_chip1[2], to_update_chip1_end)
                                to_update_chip2_start = str(update_RST_tree_span(int(to_update_chip2[0]), int(chip2[0])))
                                to_update_chip2_end = str(update_RST_tree_span(int(to_update_chip2[3]), int(chip2[0])))
                                new_chip2 = (to_update_chip2_start, to_update_chip2[1], to_update_chip2[2], to_update_chip2_end)
                                new_RST_tree.append((new_chip1, new_chip2))
                                new_logits.append(logits[i])
                            RST_tree = new_RST_tree
                            logits = new_logits
                            break
                else:
                    break
            logits = np.array(logits)     

        EDU_graph = np.zeros((3, len(EDU_text),len(EDU_text))).astype(float)
        for i in range(len(EDU_text)):
            EDU_graph[0, i, i] = 1
        for k, relation in enumerate(RST_tree):
            chip1 = relation[0]
            chip2 = relation[1] 
            # category = get_RST_category(chip1, chip2)
            # cate, flag_if_category = get_relation(category) # cate是对应某一类的索引
            # if flag_if_category:
            for i in range(int(chip1[0])-1, int(chip1[3])):
                for j in range(int(chip2[0])-1, int(chip2[3])):
                    for rel_i, rel_list in enumerate(relation_mapping.values()):
                        EDU_graph[rel_i,i,j] = np.sum(logits[k, rel_list])
                        EDU_graph[rel_i,j,i] = np.sum(logits[k, rel_list])

        
        text_tokens_all = ['<s>']
        text_index_all = []
        index_pointer = 1
        flag = False
        num_EDU_valid = 0
        for text in EDU_text:
            text_tokens = tokenizer.tokenize(text.replace('\\n', '\n'))
            new_text_tokens = []
            for s in text_tokens:
                if s == 'ĊĊ':
                    new_text_tokens.extend(['Ċ', 'Ċ'])
                else:
                    new_text_tokens.append(s)
            text_tokens = new_text_tokens
            text_len = len(text_tokens)
            if len(text_tokens_all) + text_len >= 1024:
                print("出现截断：")
                text_tokens = text_tokens[:-(len(text_tokens_all) + text_len - 1023)]
                flag = True
            text_len = len(text_tokens)
            text_tokens_all.extend(text_tokens)
            text_tokens_all.append('</s>')
            text_index_all.append([index_pointer, index_pointer + text_len - 1])
            index_pointer += (text_len + 1)
            num_EDU_valid += 1
            if flag:
                break
        
        EDU_graph = EDU_graph[:, :num_EDU_valid, :num_EDU_valid]
        a_pe_data['EDU_graph'] = EDU_graph.tolist()
        a_pe_data['EDU_text_cat'] = text_tokens_all
        a_pe_data['EDU_index'] = text_index_all

        data_pe_new.append(a_pe_data)
        if len(EDU_text) > max_num:
            max_num = len(EDU_text)
        # print(a_pe_data)
        # break

    print(data_kind)
    print("max_num:" + str(max_num))
        
    with open('./data/AAEC/aaec_EDU_logits_'+data_kind+'.json', 'w') as file:
        json.dump(data_pe_new, file, indent=4, ensure_ascii=False)
    
    print('done!\n\n')

if __name__ == '__main__':
    tokenizer = BartTokenizer.from_pretrained("./pretrained_model/bart-base",add_prefix_space=True)
    get_data(data_kind='dev')
    get_data(data_kind='test')
    get_data(data_kind='train')
