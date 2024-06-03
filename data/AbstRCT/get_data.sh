
python3 ./data/AbstRCT/get_data_from_TACL.py
python3 ./data/AbstRCT/get_EDU_merge_logits.py


if [ ! -d "./data/AbstRCT/cleaned_data/" ]; then
    mkdir ./data/AbstRCT/cleaned_data/
    echo "make new folder ./data/AbstRCT/cleaned_data/"
fi


if [ -e "./data/AbstRCT/abstrct_EDU_logits_dev.json" ] && \
   [ -e "./data/AbstRCT/abstrct_EDU_logits_test.json" ] && \
   [ -e "./data/AbstRCT/abstrct_EDU_logits_train.json" ]; then
    mv ./data/AbstRCT/abstrct_EDU_logits_dev.json ./data/AbstRCT/cleaned_data/data.dev.json
    mv ./data/AbstRCT/abstrct_EDU_logits_train.json ./data/AbstRCT/cleaned_data/data.train.json
    mv ./data/AbstRCT/abstrct_EDU_logits_test.json ./data/AbstRCT/cleaned_data/data.test.json
else
    echo "some files does not exist!"
fi

