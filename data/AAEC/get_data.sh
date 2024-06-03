
python3 ./data/AAEC/get_data_from_TACL.py
python3 ./data/AAEC/get_EDU_merge_logits.py


if [ ! -d "./data/AAEC/cleaned_data/" ]; then
    mkdir ./data/AAEC/cleaned_data/
    echo "make new folder ./data/AAEC/cleaned_data/"
fi


if [ -e "./data/AAEC/aaec_EDU_logits_dev.json" ] && \
   [ -e "./data/AAEC/aaec_EDU_logits_test.json" ] && \
   [ -e "./data/AAEC/aaec_EDU_logits_train.json" ]; then
    mv ./data/AAEC/aaec_EDU_logits_dev.json ./data/AAEC/cleaned_data/data.dev.json
    mv ./data/AAEC/aaec_EDU_logits_train.json ./data/AAEC/cleaned_data/data.train.json
    mv ./data/AAEC/aaec_EDU_logits_test.json ./data/AAEC/cleaned_data/data.test.json
else
    echo "some files does not exist!"
fi

