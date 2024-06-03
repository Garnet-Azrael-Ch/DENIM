import torch
import torch.nn as nn
import torch.nn.functional as F
from prefix_gen_bart import PrefixGenBartForConditionalGeneration
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers import BartTokenizer
from utils import mapping_index_to_ARs_token
import random
from graph import RGCN

class BartWithPointer(PrefixGenBartForConditionalGeneration):
    def __init__(self, config, config_outside):
        super().__init__(config)

        self.config_outside = config_outside
        self.dropout = self.config_outside.mlp_dropout

        self.multi_head_begin = nn.Linear(self.config.d_model, self.config.d_model)
        self.multi_head_end = nn.Linear(self.config.d_model, self.config.d_model)

        self.ac_aggre = nn.Linear(self.config.d_model, self.config.d_model)
        self.ar_aggre = nn.Linear(4 * self.config.d_model, self.config.d_model)


    def forward(
        self,
        input_ids=None,
        prefix=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        cross_attention_mask=None,
        predict_result=None,
        decoder_inputs_embeds=None,
        labels=None,
        mode=None,
        return_dict=None,
        embedding_ACs=None,
        embedding_ARs=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        if self.config_outside.dataset == "AAEC":
            num_span_token = 12
            num_ACs_token = 12
            num_ARs_token = 66
        elif self.config_outside.dataset == "AbstRCT":
            num_span_token = 11
            num_ACs_token = 11
            num_ARs_token = 55
        else:
            print("The dataset is doesn't exist!")
            raise ValueError
        

        context_outputs = self.model(
            input_ids,
            prefix=prefix,
            attention_mask=attention_mask,
            return_dict=True,
        )
        context_encoder_outputs = context_outputs.encoder_last_hidden_state
        # context_decoder_outputs = context_outputs.last_hidden_state
        
        context_outputs = context_outputs.last_hidden_state

        if predict_result is not None:
            labels = predict_result

        context_encoder_embeddings = context_encoder_outputs[:, 1:, :]
            
        if labels is not None:

            labels_para = labels[:,:num_span_token*2]
            labels_ACs = labels[:,num_span_token*2: num_span_token*2 + num_ACs_token]
            prompt_context_rep = []
            
            for batch_i, row_label_para in enumerate(labels_para):
                batch_i_rep = []
                meaningful_span_num = 0
                
                batch_i_rep.extend([torch.zeros(context_encoder_embeddings.size(2)).cuda()] * num_span_token)
                
                batch_i_ac_rep = []
                for label_pair_i in range(int(len(row_label_para)/2)):
                    pair_index_begin = row_label_para[label_pair_i]
                    pair_index_end = row_label_para[label_pair_i + num_span_token]
                    # AC = AC + span
                    if pair_index_begin != pair_index_end:
                        batch_i_ac_rep.append(self.ac_aggre(context_encoder_embeddings[batch_i, pair_index_begin:pair_index_end + 1].mean(0)))
                        meaningful_span_num += 1
                    else:
                        batch_i_ac_rep.append(torch.zeros(context_encoder_embeddings.size(2)).cuda())
                batch_i_rep.extend(batch_i_ac_rep)

                     
                for AR_token_i in range(int(num_ARs_token)):
                    pair_src, pair_tgt = mapping_index_to_ARs_token(AR_token_i, self.config_outside.dataset)
                    if pair_src < meaningful_span_num and pair_tgt < meaningful_span_num:
                        ar_repre = torch.cat([batch_i_ac_rep[pair_src], batch_i_ac_rep[pair_tgt], embedding_ACs[0, labels_ACs[batch_i, pair_src]], embedding_ACs[0, labels_ACs[batch_i, pair_tgt]]], dim=0)
                        batch_i_rep.append(self.ar_aggre(ar_repre))
                    else:
                        batch_i_rep.append(torch.zeros(context_encoder_embeddings.size(2)).cuda())
                       
                batch_i_rep = torch.stack(batch_i_rep, 0)  
                prompt_context_rep.append(batch_i_rep)
                
            prompt_context_rep = torch.stack(prompt_context_rep, 0)  # [batch_size, AC_num, dim]

            decoder_inputs_embeds = self.model.shared(decoder_input_ids) + prompt_context_rep
            
        else:
            decoder_inputs_embeds = self.model.shared(decoder_input_ids)
        
        if self.config_outside.use_decoder_prefix == True and self.config_outside.use_cross_prefix == True:
            outputs = self.model.decoder(
                inputs_embeds=decoder_inputs_embeds,
                decoder_prefix=prefix['decoder_prefix'],
                cross_prefix=prefix['cross_prefix'],
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=context_encoder_outputs,
                encoder_attention_mask=cross_attention_mask,
                return_dict=return_dict,
            )
        elif self.config_outside.use_decoder_prefix == False and self.config_outside.use_cross_prefix == True:
            outputs = self.model.decoder(
                inputs_embeds=decoder_inputs_embeds,
                decoder_prefix=None,
                cross_prefix=prefix['cross_prefix'],
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=context_encoder_outputs,
                encoder_attention_mask=cross_attention_mask,
                return_dict=return_dict,
            )
        elif self.config_outside.use_decoder_prefix == True and self.config_outside.use_cross_prefix == False:
            outputs = self.model.decoder(
                inputs_embeds=decoder_inputs_embeds,
                decoder_prefix=prefix['decoder_prefix'],
                cross_prefix=None,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=context_encoder_outputs,
                encoder_attention_mask=cross_attention_mask,
                return_dict=return_dict,
            )
        else:
            outputs = self.model.decoder(
                inputs_embeds=decoder_inputs_embeds,
                decoder_prefix=None,
                cross_prefix=None,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=context_encoder_outputs,
                encoder_attention_mask=cross_attention_mask,
                return_dict=return_dict,
            )
           
        pointer_representation = context_outputs[:, 1:, :]
        
        decoder_outputs_ = outputs.last_hidden_state # batch x dec_sequence_length x hidden_size
        
        decoder_outputs_span = decoder_outputs_[:,:num_span_token,:]
        decoder_outputs_ACs = decoder_outputs_[:,num_span_token:num_span_token + num_ACs_token,:]
        decoder_outputs_ARs = decoder_outputs_[:,num_span_token + num_ACs_token:,:]
        

        decoder_outputs_span_begin = self.multi_head_begin(decoder_outputs_span)
        decoder_outputs_span_end = self.multi_head_end(decoder_outputs_span)
        decoder_outputs_span_new = torch.cat([decoder_outputs_span_begin, decoder_outputs_span_end], dim=1)

        
        logits_para = torch.matmul(decoder_outputs_span_new, pointer_representation.transpose(1,2)) 
        logits_ACs = torch.matmul(decoder_outputs_ACs, embedding_ACs.transpose(1,2))
        logits_ARs = torch.matmul(decoder_outputs_ARs, embedding_ARs.transpose(1,2))
        
        
    

        lm_loss = None
        if labels is not None and mode=="train":
            
            flag_loss_AC = True
            flag_loss_AR = True
            
            labels_para = labels[:,:num_span_token*2]
            labels_ACs = labels[:,num_span_token*2: num_span_token*2 + num_ACs_token]
            labels_ARs = labels[:,num_span_token*2 + num_ACs_token:]
            loss_fct_para = nn.CrossEntropyLoss()
            loss_fct_ACs = nn.CrossEntropyLoss()
            loss_fct_ARs = nn.CrossEntropyLoss()
        
        
            # span-----
            mask_pad = attention_mask.unsqueeze(1).repeat(1, logits_para.size(1), 1)[:,:,1:].bool()
            logits_para[~mask_pad] = torch.tensor(float("-inf")).cuda()
            logits_para_merge_batch = logits_para.reshape(-1, logits_para.size(2))
            labels_para_merge_batch = labels_para.reshape(-1)
            mask = (labels_para_merge_batch != -1)
            labels_para_merge_batch = labels_para_merge_batch[mask]
            logits_para_merge_batch = logits_para_merge_batch[mask]
            lm_loss_para = loss_fct_para(logits_para_merge_batch, labels_para_merge_batch)
            # -----
            
            
            # AC-----
            logits_ACs_merge_batch = logits_ACs.reshape(-1, logits_ACs.size(2))
            labels_ACs_merge_batch = labels_ACs.reshape(-1)
            mask = (labels_ACs_merge_batch != -1)
            labels_ACs_merge_batch = labels_ACs_merge_batch[mask]
            logits_ACs_merge_batch = logits_ACs_merge_batch[mask]

            if labels_ACs_merge_batch.tolist():
                lm_loss_ACs = loss_fct_ACs(logits_ACs_merge_batch,labels_ACs_merge_batch)
            else:
                lm_loss_ACs = torch.tensor(0, device=decoder_outputs_.device)
                flag_loss_AC = False
            # -----
            
            # AR-----
            
            logits_ARs_merge_batch = logits_ARs.reshape(-1, logits_ARs.size(2))
            labels_ARs_merge_batch = labels_ARs.reshape(-1)
            # # 创建一个掩码，标识哪些位置的元素是要保留的
            # 将标签中不为-1的项的掩码设为True以确保它们保留
            mask = (labels_ARs_merge_batch != -1)
            # 使用掩码过滤label和logits张量
            labels_ARs_merge_batch = labels_ARs_merge_batch[mask]
            logits_ARs_merge_batch = logits_ARs_merge_batch[mask]
            if labels_ARs_merge_batch.tolist():    
                # 不训练所有的no relation的边，因为效果会变得很差
                a_AR = self.config_outside.a_AR
                relation_count = ((labels_ARs_merge_batch==1) | (labels_ARs_merge_batch==0)).sum().item()
                if relation_count:
                    mask = labels_ARs_merge_batch != 2
                    mask_True = (mask == False).nonzero()
                    mask_True = [item.item() for sublist in mask_True for item in sublist]
                    if(len(mask_True)) > int((relation_count)*a_AR):
                        mask_True = random.sample(range(len(mask_True)), int((relation_count)*a_AR))
                    mask[mask_True] = True
                    labels_ARs_merge_batch = labels_ARs_merge_batch[mask]
                    logits_ARs_merge_batch = logits_ARs_merge_batch[mask]
                else:
                    # 完全没有relation，保留一个，防止出现NaN
                    labels_ARs_merge_batch = labels_ARs_merge_batch[0].unsqueeze(0)
                    logits_ARs_merge_batch = logits_ARs_merge_batch[0].unsqueeze(0)
                
                lm_loss_ARs = loss_fct_ARs(logits_ARs_merge_batch, labels_ARs_merge_batch)
                
            else:
                lm_loss_ARs = torch.tensor(0, device=decoder_outputs_.device)
                flag_loss_AR = False
             

            if flag_loss_AC and flag_loss_AR:
                lm_loss = lm_loss_para + lm_loss_ACs + lm_loss_ARs
            elif flag_loss_AC and not flag_loss_AR:
                lm_loss = lm_loss_para + lm_loss_ACs
            elif not flag_loss_AC and flag_loss_AR:
                raise ValueError
            elif not flag_loss_AC and not flag_loss_AR:
                lm_loss = lm_loss_para
            else:
                raise ValueError

        if not return_dict:
            output = (lm_loss,) + outputs[1:]
            return ((lm_loss,) + output) if lm_loss is not None else output


        return lm_loss, lm_loss_para, lm_loss_ACs, lm_loss_ARs

    def predict(
        self,
        input_ids=None,
        prefix=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        cross_attention_mask=None,
        predict_result=None,
        decoder_inputs_embeds=None,
        labels=None,
        mode=None,
        return_dict=None,
        embedding_ACs=None,
        embedding_ARs=None,
        context_rep=None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        if self.config_outside.dataset == "AAEC":
            num_span_token = 12
            num_ACs_token = 12
            num_ARs_token = 66
        elif self.config_outside.dataset == "AbstRCT":
            num_span_token = 11
            num_ACs_token = 11
            num_ARs_token = 55
        else:
            print("The dataset is doesn't exist!")
            raise ValueError
        
        if mode == "ACS":

            context_outputs = self.model(
                input_ids,
                prefix=prefix,
                attention_mask=attention_mask,
                return_dict=True,
            )
            context_encoder_outputs = context_outputs.encoder_last_hidden_state
            # context_decoder_outputs = context_outputs.last_hidden_state
            
            context_outputs = context_outputs.last_hidden_state
            context_rep_return = context_encoder_outputs

        else:
            context_encoder_outputs = context_rep
            context_rep_return = None
                
        if predict_result is not None:
            labels = predict_result

        context_encoder_embeddings = context_encoder_outputs[:, 1:, :]
            
        if labels is not None:
            labels_para = labels[:,:num_span_token*2]
            labels_ACs = labels[:,num_span_token*2: num_span_token*2 + num_ACs_token]
            prompt_context_rep = []
            
            for batch_i, row_label_para in enumerate(labels_para):
                batch_i_rep = []
                meaningful_span_num = 0
                
                batch_i_rep.extend([torch.zeros(context_encoder_embeddings.size(2)).cuda()] * num_span_token)
                
                batch_i_ac_rep = []
                for label_pair_i in range(int(len(row_label_para)/2)):
                    pair_index_begin = row_label_para[label_pair_i]
                    pair_index_end = row_label_para[label_pair_i + num_span_token]
                    # AC = AC + span
                    if pair_index_begin != pair_index_end:
                        batch_i_ac_rep.append(self.ac_aggre(context_encoder_embeddings[batch_i, pair_index_begin:pair_index_end + 1].mean(0)))
                        meaningful_span_num += 1
                    else:
                        batch_i_ac_rep.append(torch.zeros(context_encoder_embeddings.size(2)).cuda())
                batch_i_rep.extend(batch_i_ac_rep)
                     
                for AR_token_i in range(int(num_ARs_token)):
                    pair_src, pair_tgt = mapping_index_to_ARs_token(AR_token_i, self.config_outside.dataset)
                    if pair_src < meaningful_span_num and pair_tgt < meaningful_span_num:
                        ar_repre = torch.cat([batch_i_ac_rep[pair_src], batch_i_ac_rep[pair_tgt], embedding_ACs[0, labels_ACs[batch_i, pair_src]], embedding_ACs[0, labels_ACs[batch_i, pair_tgt]]], dim=0)
                        batch_i_rep.append(self.ar_aggre(ar_repre))
                    else:
                        batch_i_rep.append(torch.zeros(context_encoder_embeddings.size(2)).cuda())
                       
                batch_i_rep = torch.stack(batch_i_rep, 0)
                prompt_context_rep.append(batch_i_rep)
                
            prompt_context_rep = torch.stack(prompt_context_rep, 0)  # [batch_size, AC_num, dim]

            decoder_inputs_embeds = self.model.shared(decoder_input_ids) + prompt_context_rep
            
        else:
            decoder_inputs_embeds = self.model.shared(decoder_input_ids)
        
        if self.config_outside.use_decoder_prefix == True and self.config_outside.use_cross_prefix == True:
            outputs = self.model.decoder(
                inputs_embeds=decoder_inputs_embeds,
                decoder_prefix=prefix['decoder_prefix'],
                cross_prefix=prefix['cross_prefix'],
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=context_encoder_outputs,
                encoder_attention_mask=cross_attention_mask,
                return_dict=return_dict,
            )
        elif self.config_outside.use_decoder_prefix == False and self.config_outside.use_cross_prefix == True:
            outputs = self.model.decoder(
                inputs_embeds=decoder_inputs_embeds,
                decoder_prefix=None,
                cross_prefix=prefix['cross_prefix'],
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=context_encoder_outputs,
                encoder_attention_mask=cross_attention_mask,
                return_dict=return_dict,
            )
        elif self.config_outside.use_decoder_prefix == True and self.config_outside.use_cross_prefix == False:
            outputs = self.model.decoder(
                inputs_embeds=decoder_inputs_embeds,
                decoder_prefix=prefix['decoder_prefix'],
                cross_prefix=None,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=context_encoder_outputs,
                encoder_attention_mask=cross_attention_mask,
                return_dict=return_dict,
            )
        else:
            outputs = self.model.decoder(
                inputs_embeds=decoder_inputs_embeds,
                decoder_prefix=None,
                cross_prefix=None,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=context_encoder_outputs,
                encoder_attention_mask=cross_attention_mask,
                return_dict=return_dict,
            )
        
    
        
        decoder_outputs_ = outputs.last_hidden_state # batch x dec_sequence_length x hidden_size
        if mode == "ACS":
            pointer_representation = context_outputs[:, 1:, :]
            decoder_outputs_span = decoder_outputs_[:,:num_span_token,:]
            decoder_outputs_span_begin = self.multi_head_begin(decoder_outputs_span)
            decoder_outputs_span_end = self.multi_head_end(decoder_outputs_span)
            decoder_outputs_span_new = torch.cat([decoder_outputs_span_begin, decoder_outputs_span_end], dim=1)
            logits = torch.matmul(decoder_outputs_span_new, pointer_representation.transpose(1,2)) 
            
        elif mode == "ACTC":
            decoder_outputs_ACs = decoder_outputs_[:,num_span_token:num_span_token + num_ACs_token,:]
            logits = torch.matmul(decoder_outputs_ACs, embedding_ACs.transpose(1,2))
        else:
            decoder_outputs_ARs = decoder_outputs_[:,num_span_token + num_ACs_token:,:]
            logits = torch.matmul(decoder_outputs_ARs, embedding_ARs.transpose(1,2))
        
        
        return context_rep_return, logits



class EDURoberta(nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.rel_num = config.rel_num
        self.graph_layer = config.graph_layer
        self.dropout = config.graph_dropout
        self.edge_norm = config.edge_norm
        self.residual = config.residual
        self.plm_output_size = config.latent_dim
        
        self.tokenizer = BartTokenizer.from_pretrained(config.model_name, add_prefix_space=True)
        self.model = model.model.encoder
        self.max_len = 1024

        self.graph = RGCN(self.rel_num, self.plm_output_size, self.plm_output_size, self.graph_layer, self.dropout,
                            edge_norm=self.edge_norm, residual=self.residual)


    def get_encoder_output(self, EDU_graph, EDU_text_cat_idxs, EDU_text_cat_attn, EDU_attn, EDU_index):


        bth_size = EDU_text_cat_idxs.size(0)
        max_batch_EDU_length = max(len(x) for x in EDU_index)
        
        rep_EDU_batch = []
        outputs = self.model(input_ids=EDU_text_cat_idxs, attention_mask=EDU_text_cat_attn).last_hidden_state
        
        for i, a_EDU_index in enumerate(EDU_index):
            rep_EDU = []
            for EDU_span in a_EDU_index:
                rep_EDU.append(torch.mean(outputs[i,EDU_span[0]:EDU_span[1]+1,:], dim=0))
            rep_EDU = torch.stack(rep_EDU, dim=0)
            EDU_pad = torch.zeros((max_batch_EDU_length - rep_EDU.size(0), rep_EDU.size(1)), dtype=rep_EDU.dtype, device=rep_EDU.device)
            rep_EDU = torch.cat((rep_EDU, EDU_pad), dim=0)

            rep_EDU_batch.append(rep_EDU)
        
        rep_EDU_batch = torch.stack(rep_EDU_batch, dim=0)        

        EDU_graph = EDU_graph.to(dtype=rep_EDU.dtype)

        rep_EDU_RGCN = self.graph(rep_EDU_batch, EDU_graph)
        return rep_EDU_RGCN, EDU_attn
        