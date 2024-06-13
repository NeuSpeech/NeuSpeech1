python process_dataset/combine_jsonls.py --jsonl \
"datasets/schoffelen2019n/preprocess6/ZINNEN/train.jsonl"\
 "datasets/gwilliams2023/preprocess5/split1/train.jsonl" \
--output_jsonl="datasets/gwilliams_schoffelen/train.jsonl"

python process_dataset/combine_jsonls.py --jsonl \
"datasets/schoffelen2019n/preprocess6/ZINNEN/val.jsonl"\
 "datasets/gwilliams2023/preprocess5/split1/val.jsonl" \
--output_jsonl="datasets/gwilliams_schoffelen/val.jsonl"

python process_dataset/combine_jsonls.py --jsonl \
"datasets/schoffelen2019n/preprocess6/ZINNEN/test.jsonl"\
 "datasets/gwilliams2023/preprocess5/split1/test.jsonl" \
--output_jsonl="datasets/gwilliams_schoffelen/test.jsonl"


python process_dataset/add_language.py --jsonl \
"datasets/gwilliams2023/preprocess5/info.jsonl" \
"datasets/gwilliams2023/preprocess5/split1/train.jsonl" \
"datasets/gwilliams2023/preprocess5/split1/val.jsonl" \
"datasets/gwilliams2023/preprocess5/split1/test.jsonl" \
"datasets/gwilliams2023/preprocess5/split2/train.jsonl" \
"datasets/gwilliams2023/preprocess5/split2/val.jsonl" \
"datasets/gwilliams2023/preprocess5/split2/test.jsonl" \
 --language="English"