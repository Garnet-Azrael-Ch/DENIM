import logging, os
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig
from model_copyutils import  EDURoberta, BartWithPointer
from projector import Projector
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class GenerativeModel(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.model = EDUPrefixGenCopyReg(config, tokenizer)


    def forward(self, batch):
        return self.model(batch)
        
    def predict(self, batch):
        return self.model.predict(batch)
    

    def save_model(self, save_path):
        """
        This save model is created mainly in case we need partial save and load. Such as cases with pretraining.
        """
        self.model.save_model(save_path)

    def load_model(self, load_path):
        """
        This load model is created mainly in case we need partial save and load. Such as cases with pretraining.
        """
        self.model.load_model(load_path)


class EDUPrefixGenBase(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        """
        Need to init by class
        """

    def get_EDU_embedding(self, EDU_graph, EDU_text_cat_idxs, EDU_text_cat_attn, EDU_attn, EDU_index):
        return self.EDU_model.get_encoder_output(EDU_graph, EDU_text_cat_idxs, EDU_text_cat_attn, EDU_attn, EDU_index)
    
    def set_output_embeddings(self, new_embeddings):
        self.model.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.model.lm_head

    def init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def init_embedding(self):

        if self.config.dataset == "AAEC":
            label_dict = {0: 'premise', 1: 'claim', 2: 'major claim', 3: 'support', 4: 'attack', 5: "no relation"}
    
        elif self.config.dataset == "AbstRCT":
            label_dict = {0: 'evidence', 1: 'claim', 2: 'major claim', 3: 'support', 4: 'partial attack', 5: "no relation", 6: "attack"}
        else:
            raise ValueError

        label_dict = {i: self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(v)) for i, v in label_dict.items()}
        label_emb = []
        input_embeds = self.model.get_input_embeddings()
        # print("input_embeds", type(input_embeds))
        for i in range(len(label_dict)):
            label_emb.append(
                torch.index_select(input_embeds.weight, 0, torch.tensor(label_dict[i])).mean(dim=0))
        label_emb = torch.stack(label_emb)


        if self.config.dataset == "AAEC":
            prompt_embedding = nn.Embedding(int(12 + 12 + 66), input_embeds.weight.size(1), 0)
            self.init_weights(prompt_embedding)
        elif self.config.dataset == "AbstRCT":
            prompt_embedding = nn.Embedding(int(11 + 11 + 55), input_embeds.weight.size(1), 0)
            self.init_weights(prompt_embedding)
        else:
            raise ValueError

        # label prompt mask
        prompt_emb = prompt_embedding.weight[:, :]
        self.embedding = prompt_Embedding(self.config, input_embeds, label_emb, prompt_emb, self.tokenizer)
        self.model.set_input_embeddings(self.embedding)

        output_embeddings = OutputEmbedding(self.get_output_embeddings().bias)
        output_embeddings.weight = self.embedding.weight
        vocab_size = output_embeddings.bias.size(0)
        assert self.vocab_size == vocab_size, print("self.vocab_size", self.vocab_size, "vocab_size", vocab_size)
        output_embeddings.bias.data = F.pad(output_embeddings.bias.data,
                                            (0, self.embedding.size - output_embeddings.bias.shape[0],), "constant", 0)
        self.set_output_embeddings(output_embeddings)

    def get_prefix(self, EDU_embedding):
        batch_size = EDU_embedding[0].size()[0]
        input_tokens = torch.arange(self.config.prefix_length).long().cuda()
        input_tokens = input_tokens.unsqueeze(0).expand(batch_size, -1)
        input_embeds = self.wte(input_tokens) # bz, prefix_len, dim
        prefix = {}
        if self.model_config.use_prefix:
            prefix['encoder_prefix'] = self.enc_prefix_projector.project(EDU_embedding, input_embeds)
            prefix['cross_prefix'] = self.cross_prefix_projector.project(EDU_embedding, input_embeds)
            prefix['decoder_prefix'] = self.dec_prefix_projector.project(EDU_embedding, input_embeds)
        return prefix


    # 为了获取不在词表中表示的embedding
    def get_extra_embeddings(self, input_ids, batch_size):
        embeds = self.model.get_input_embeddings()

        embedding_vacab_size = embeds.size

        if self.config.dataset == "AAEC":
            prompt_token_size = 12 + 12 + 66
            label_token_size = 6
        elif self.config.dataset == "AbstRCT":
            prompt_token_size = 11 + 11 + 55
            label_token_size = 7
        else:
            raise ValueError
        
        
        para_embeddings = embeds(input_ids)[:, 1:, :]

        label_index = torch.arange(label_token_size).long().cuda() + embedding_vacab_size - prompt_token_size - label_token_size 
        embeds_weight = embeds.weight()
        label_embeddings = torch.index_select(embeds_weight, 0, label_index.int())

        if self.config.dataset == "AAEC":
            ACs_embeddings = label_embeddings[:3].repeat(batch_size, 1, 1)
            ARs_embeddings = label_embeddings[3:].repeat(batch_size, 1, 1)
        elif self.config.dataset == "AbstRCT":
            ACs_embeddings = label_embeddings[:3].repeat(batch_size, 1, 1)
            ARs_embeddings = label_embeddings[3:].repeat(batch_size, 1, 1)
        
        return para_embeddings, ACs_embeddings, ARs_embeddings
        


    def forward(self, batch):
        batch_size = batch.lbl_idxs.size(0)
        EDU_graph = batch.EDU_graph
        EDU_text_cat_idxs = batch.EDU_text_cat_idxs
        EDU_text_cat_attn = batch.EDU_text_cat_attn
        EDU_attn = batch.EDU_attn
        EDU_index = batch.EDU_index
        EDU_embedding = self.get_EDU_embedding(EDU_graph, EDU_text_cat_idxs, EDU_text_cat_attn, EDU_attn, EDU_index)
        prefix = self.get_prefix(EDU_embedding)

        __, ACs_embeddings, ARs_embeddings = self.get_extra_embeddings(batch.enc_idxs, batch_size)
        # if self.mask_type == "full":
		
        masks = batch.enc_attn

        loss, loss_acs, loss_actc, loss_artc = self.model(input_ids=batch.enc_idxs,
                             prefix=prefix,
                             attention_mask=batch.enc_attn, 
                             decoder_input_ids=batch.dec_idxs, 
                             decoder_attention_mask=batch.dec_attn, 
                             cross_attention_mask=masks,
                             labels=batch.lbl_idxs, 
                             mode="train",
                             return_dict=True,
                             embedding_ACs=ACs_embeddings,
                             embedding_ARs=ARs_embeddings
                             )
        
        
        return loss, loss_acs, loss_actc, loss_artc
        
    def predict(self, batch):
        
        if self.config.dataset == "AAEC":
            num_span_label = 12
            num_ac_label = 12
        elif self.config.dataset == "AbstRCT":
            num_span_label = 11
            num_ac_label = 11
        else:
            raise ValueError
        
        self.eval()
        
        with torch.no_grad():
            batch_size = batch.lbl_idxs.size(0)
            EDU_graph = batch.EDU_graph
            EDU_text_cat_idxs = batch.EDU_text_cat_idxs
            EDU_text_cat_attn = batch.EDU_text_cat_attn
            EDU_attn = batch.EDU_attn
            EDU_index = batch.EDU_index
            EDU_embedding = self.get_EDU_embedding(EDU_graph, EDU_text_cat_idxs, EDU_text_cat_attn, EDU_attn, EDU_index)
            prefix = self.get_prefix(EDU_embedding)

            __, ACs_embeddings, ARs_embeddings = self.get_extra_embeddings(batch.enc_idxs, batch_size)


            context_rep_return, logits_para= self.model.predict(input_ids=batch.enc_idxs,
                                                            prefix=prefix,
                                                            attention_mask=batch.enc_attn, 
                                                            decoder_input_ids=batch.dec_idxs, 
                                                            decoder_attention_mask=batch.dec_attn, 
                                                            labels=None, 
                                                            mode="ACS",
                                                            predict_result=None,
                                                            return_dict=True,
                                                            embedding_ACs=ACs_embeddings,
                                                            embedding_ARs=ARs_embeddings,
                                                            context_rep=None
                                                            )
        
        # 先预测出span，再重新构建decoder_embedding输入模型中
        
            mask_pad = batch.enc_attn.unsqueeze(1).repeat(1, logits_para.size(1), 1)[:,:,1:].bool()
            logits_para[~mask_pad] = torch.tensor(float("-inf")).cuda()
            span_predict_mask = []
            
            for bth in  range(logits_para.size(0)):

                a_span_predict_mask_begin = []
                a_span_predict_mask_end = []
                last_meaningful_token_index = (mask_pad[bth, 0] == True).nonzero()[-1].item()
                
                flag_first_predict = True
                
                if self.config.dataset == "AAEC" or self.config.dataset == "AbstRCT":  # 补充预测
                    if self.config.predict_supplement == True:  # 不连续+补充预测
                        for span in range(int(logits_para.size(1)/2)):
                            if flag_first_predict:
                                last_time_end = -1
                                flag_first_predict = False
                            else:
                                last_time_end = int(a_span_predict_mask_end[-1])
                                
                            if last_time_end < last_meaningful_token_index:
                                last_time_end += 1
                            a_span_predict_mask_begin.append(torch.argmax(logits_para[bth, span, last_time_end:]) + last_time_end)
                            
                            this_time_begin = int(a_span_predict_mask_begin[-1])
                            if this_time_begin < last_meaningful_token_index:
                                this_time_begin += 1
                            this_time_end = torch.argmax(logits_para[bth, span + num_span_label, this_time_begin:]) + this_time_begin
                            if this_time_begin != last_meaningful_token_index and this_time_end == last_meaningful_token_index:
                                __, this_time_end = torch.topk(logits_para[bth, span + num_span_label, this_time_begin:], 2)
                                this_time_end = this_time_end[1] + this_time_begin
                            a_span_predict_mask_end.append(this_time_end)
                    
                        a_span_predict_mask_begin = torch.stack(a_span_predict_mask_begin, dim=0)  
                        a_span_predict_mask_end = torch.stack(a_span_predict_mask_end, dim=0)  
                    else:
                        for span in range(int(logits_para.size(1)/2)):
                            if flag_first_predict:
                                last_time_end = -1
                                flag_first_predict = False
                            else:
                                last_time_end = int(a_span_predict_mask_end[-1])
                                
                            if last_time_end < last_meaningful_token_index:
                                last_time_end += 1
                            a_span_predict_mask_begin.append(torch.argmax(logits_para[bth, span, last_time_end:]) + last_time_end)
                            
                            this_time_begin = int(a_span_predict_mask_begin[-1])
                            if this_time_begin < last_meaningful_token_index:
                                this_time_begin += 1
                            a_span_predict_mask_end.append(torch.argmax(logits_para[bth, span + num_span_label, this_time_begin:])+this_time_begin)
                        
                        a_span_predict_mask_begin = torch.stack(a_span_predict_mask_begin, dim=0)  
                        a_span_predict_mask_end = torch.stack(a_span_predict_mask_end, dim=0)  
                
                    
                span_predict_mask.append(torch.cat([a_span_predict_mask_begin, a_span_predict_mask_end]))
                
            span_predict_mask = torch.stack(span_predict_mask, dim=0)    
                
            padding_tensor = torch.zeros(batch_size, num_ac_label, dtype=torch.int64).cuda()
            predict_result = torch.cat((span_predict_mask, padding_tensor), dim=1)
            
            __, logits_ACs = self.model.predict(input_ids=batch.enc_idxs,
                                                            prefix=prefix,
                                                            attention_mask=batch.enc_attn, 
                                                            decoder_input_ids=batch.dec_idxs, 
                                                            decoder_attention_mask=batch.dec_attn, 
                                                            labels=None, 
                                                            mode="ACTC",
                                                            predict_result=predict_result,
                                                            return_dict=True,
                                                            embedding_ACs=ACs_embeddings,
                                                            embedding_ARs=ARs_embeddings,
                                                            context_rep=context_rep_return
                                                            )

            ACs_predict = torch.argmax(logits_ACs,dim=2)
            predict_result = torch.cat((span_predict_mask, ACs_predict), dim=1)
            __, logits_ARs = self.model.predict(input_ids=batch.enc_idxs,
                                                            prefix=prefix,
                                                            attention_mask=batch.enc_attn, 
                                                            decoder_input_ids=batch.dec_idxs, 
                                                            decoder_attention_mask=batch.dec_attn, 
                                                            labels=None, 
                                                            mode="ARTC",
                                                            predict_result=predict_result,
                                                            return_dict=True,
                                                            embedding_ACs=ACs_embeddings,
                                                            embedding_ARs=ARs_embeddings,
                                                            context_rep=context_rep_return
                                                            )
            ARs_predict = torch.argmax(logits_ARs,dim=2)
        
        return span_predict_mask, ACs_predict, ARs_predict
    
    
    def save_model(self, save_path):
        self.model.save_pretrained(os.path.join(save_path, "checkpoint-bart"))
        torch.save(self.wte.state_dict(), os.path.join(save_path, "wte.mdl"))
        torch.save(self.EDU_model.state_dict(), os.path.join(save_path, "edumodel.mdl"))
        if self.model_config.use_prefix:
            self.enc_prefix_projector.save(os.path.join(save_path, "enc_prefix_projector.mdl"))
            self.cross_prefix_projector.save(os.path.join(save_path, "cross_prefix_projector.mdl"))
            self.dec_prefix_projector.save(os.path.join(save_path, "dec_prefix_projector.mdl"))
    
    def load_model(self, load_path):
        logger.info(f"Loading model from {load_path}")
        self.model.from_pretrained(os.path.join(load_path, "checkpoint-bart"))
        self.wte.load_state_dict(torch.load(os.path.join(load_path, "wte.mdl"), map_location=f'cuda:{self.config.gpu_device}'))
        self.EDU_model.load_state_dict(torch.load(os.path.join(load_path, "edumodel.mdl"), map_location=f'cuda:{self.config.gpu_device}'))
        if self.model_config.use_prefix:
            self.enc_prefix_projector.load(os.path.join(load_path, "enc_prefix_projector.mdl"))
            self.cross_prefix_projector.load(os.path.join(load_path, "cross_prefix_projector.mdl"))
            self.dec_prefix_projector.load(os.path.join(load_path, "dec_prefix_projector.mdl"))
            


class EDUPrefixGenCopyReg(EDUPrefixGenBase):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.config = config
        self.tokenizer = tokenizer
        self.vocab_size = len(self.tokenizer)
        logger.info(f'Using model {self.__class__.__name__}')
        logger.info(f'Loading pre-trained model {config.model_name}')

        if config.model_name.endswith('bart-base'):
            # main model
            self.model_config = AutoConfig.from_pretrained(config.model_name, cache_dir=config.cache_dir)
            self.model_config.output_attentions = True
            self.model_config.use_prefix = config.use_prefix
            self.model_config.prefix_length = config.prefix_length
            
            self.model = BartWithPointer.from_pretrained(config.model_name, cache_dir=config.cache_dir, config=self.model_config, config_outside=self.config)
            
            self.EDU_model = EDURoberta(self.config, self.model)
        
            ## Prefix Generator
            self.wte = nn.Embedding(config.prefix_length, config.latent_dim)

            if self.model_config.use_prefix:
                self.enc_prefix_projector =  Projector(self.config, self.model_config, "AttIndep")
                self.cross_prefix_projector =  Projector(self.config, self.model_config, "AttIndep")
                self.dec_prefix_projector =  Projector(self.config, self.model_config, "AttIndep")

            self.init_embedding()

        else:
            raise ValueError("Model does not support yet.")



class prompt_Embedding(nn.Module):
    def __init__(self, config, embedding, label_embedding, prompt_embedding, tokenizer=None):
        super(prompt_Embedding, self).__init__()
        
        self.padding_idx = tokenizer.pad_token_id
        self.original_embedding = embedding
        new_embedding = torch.cat(
            [torch.zeros(1, label_embedding.size(-1), device=label_embedding.device, dtype=label_embedding.dtype),
             label_embedding, prompt_embedding], dim=0)
        self.new_embedding = nn.Embedding.from_pretrained(new_embedding, False, 0)
        self.size = self.original_embedding.num_embeddings + self.new_embedding.num_embeddings - 1
        self.prompt_idx = self.original_embedding.num_embeddings + label_embedding.size(0)


    @property
    def weight(self):
        def foo():
            return torch.cat([self.original_embedding.weight, self.new_embedding.weight[1:, :]], dim=0)

        return foo

    def forward(self, x, adjs=None, span_num=None, context_rep=None):
        # print("self.weight()[-1]", self.weight()[-1])  # see whether the parameters have been updated or not

        if adjs == None and span_num == None:
            y = F.embedding(x.long(), self.weight(), self.padding_idx)

        return y

class OutputEmbedding(nn.Module):
    def __init__(self, bias):
        super(OutputEmbedding, self).__init__()
        self.weight = None
        self.bias = bias

    def forward(self, x):
        return F.linear(x, self.weight(), self.bias)