# Seq2seq with ALONE to reproduce summarization 

## Requirements

- PyTorch version 0.4
- Python version >= 3.6

## Training

For binary mask with D_{inter} = 1024 using 4GPUs

```bash
python -u train.py \
    pre-processed-data-dir \
    --arch transformer_wmt_en_de --optimizer adam --adam-betas '(0.9, 0.98)' \
    --clip-norm 1.0 --lr 0.001 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --warmup-init-lr 1e-07 --dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 --max-tokens 3584 --min-lr 1e-09 --update-freq 16 --log-interval 100 --max-epoch 100 \
    --one-emb binary --one-emb-relu-dropout 0.5 \
    --one-emb-layernum 2 --one-emb-inter-dim 1024 \
    --share-all-embeddings --stop-relu-dropout-update 500 \
    --represent-length-by-lrpe --ordinary-sinpos --save-dir model-save-dir
```

## Test (decoding)

Averaging latest 10 checkpoints.

```bash
python scripts/average_checkpoints.py --inputs model-save-dir --num-epoch-checkpoints 10 --output model-save-dir/averaged.pt
```

Decoding with the averaged checkpoint.

```bash
python generate.py pre-processed-data-dir --path model-save-dir/averaged.pt  --beam 5 --desired-length 75
```

For comparison with the reported scores, use reranking following [this procedure](https://github.com/takase/control-length/tree/master/encdec).

## Acknowledgements

A large portion of this repo is borrowed from [fairseq](https://github.com/pytorch/fairseq).