# Prerequisites:
Need to install sentence-transformers:

`pip install sentence-transformers`
# To pretrain TSDAE with domain data
`python run_tsade.py  --model_name_or_path bert-base-uncased --do_train --per_device_train_batch_size 4 --learning_rate 2e-5 --num_train_epochs 3 --output_dir output --overwrite_output_dir --train_file aic_nlp_bert.txt --dataloader_pin_memory False --fp16 --save_steps 500 --delete_ratio 0.6`

# To load the trained model into a sentencetransformer for later downstream tasks:
```
from sentence_transformers import SentenceTransformer
sbert = SentenceTransformer("/home/kangwenj/aic/fork/transformers/examples/pytorch/unsupervised_learning/tsdae/output/tsdae-2022-06-06_19-11-48/checkpoint-500")
```