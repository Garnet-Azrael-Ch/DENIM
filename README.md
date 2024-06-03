# Discourse Structure-Aware Prefix for Generation-Based End-to-End Argumentation Mining
Code for our ACL-2024 paper [Discourse Structure-Aware Prefix for Generation-Based End-to-End Argumentation Mining] 

## Environment
- ipdb==0.13.9
- nltk==3.8.1
- numpy==1.21.5
- pandas==1.3.5
- pattern==3.6
- scikit_learn==1.0.2
- tensorboardX==2.5.1
- torch==1.9.1
- torch_geometric==2.3.1
- tqdm==4.66.1
- transformers==4.18.0

The other packages needed is shown in requirement.txt .

### DSG Parser Setup
We follow the instruction in https://github.com/seq-to-mind/DMRST_Parser to get the DSG information with minor modifications.


## Data
We support `AAEC`, and `AbstRCT`.

### Original data
Our preprocessing mainly adapts https://github.com/hitachi-nlp/graph_parser released scripts. We deeply thank the contribution from the authors of the paper.

1. Get original data by https://github.com/hitachi-nlp/graph_parser, and arrange according to the following path:

data
├── AAEC
│     ├── aaec_para_dev.mrp
│     ├── aaec_para_test.mrp
│     └── aaec_para_train.mrp
└── AbstRCT
      ├── abstRCT_dev.mrp
      ├── abstRCT_test.mrp
      └── abstRCT_train.mrp


2. Prepare DSG data by https://github.com/seq-to-mind/DMRST_Parser. Please replace the original project file with our modified file in folder `./DSG/replacement/` and the txt files can help obtain DSG files. Arrange according to the following path:

data
├── AAEC
│     ├── aaec_para_dev.mrp
│     ├── aaec_para_test.mrp
│     ├── aaec_para_train.mrp
│     ├── aaec_RST_logits_dev.json
│     ├── aaec_RST_logits_test.json
│     └── aaec_RST_logits_train.json
└── AbstRCT
      ├── abstRCT_dev.mrp
      ├── abstRCT_test.mrp
      ├── abstRCT_train.mrp
      ├── abstrct_RST_logits_dev.json
      ├── abstrct_RST_logits_test.json
      └── abstrct_RST_logits_train.json

3. 
Download the pretrained model bart-base and place it in folder `./pretrained_model/`.
Run `sh ./data/AbstRCT/get_data.sh` and `sh ./data/AAEC/get_data.sh`.

The above scripts will generate processed data in `.data/AbstRCT/cleaned_data/` and `.data/AAEC/cleaned_data/`.



## Training and Evaluating
Run `python3 ./models/train_and_eval.py`

### Outputs files
1. You can see the final result in `train.log`.
2. You can get the predicted results of the model in `pred.dev.json` and `pred.test.json`.
3. You can get the time of training or inference and losses during every epochs in `time_result_3_stage_pe.xlsx`.

## Citation

If you find that the code is useful in your research, please consider citing our paper.

