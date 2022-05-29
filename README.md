# Reproducing "Self-Supervised Quality Estimation for Machine Translation" by Zheng et al.

## Introduction

Adapted and annotated the original author's code from https://github.com/THUNLP-MT/SelfSupervisedQE as part of
the reproducibility project for CSE 481 (NLP Capstone) at University of Washington during spring quarter of
2022.

Team Name: Team TBD

List of Members: Chahyon Ku, Daniel Cheng, Sherry Zhao, Shubhkarman Singh

## Steps

1. Clone repo

2. Install Dependencies (Ours / Author's)

python (3.9.12 / 3.6)

pytorch (1.11.0 / >=1.4.0)

transformers (4.11.3 / >=4.4.2)

pandas (1.4.2 / >=1.0.5)

3. Download pre-trained multilingual BERT from Hugging Face (https://huggingface.co/bert-base-multilingual-cased).

4. Fine-tune the model by running python train.py

    ```
    python train.py \
        --train-src', type=str, default='data/en-de/train/train.en')
        parser.add_argument('--train-tgt', type=str, default='data/en-de/train/train.de')

        parser.add_argument('--dev-src', type=str, default='data/en-de/dev/dev.src')
        parser.add_argument('--dev-tgt', type=str, default='data/en-de/dev/dev.mt')
        parser.add_argument('--dev-hter', type=str, default='data/en-de/dev/dev.hter')
        parser.add_argument('--dev-tags', type=str, default='data/en-de/dev/dev.tags')

        parser.add_argument('--block-size', type=int, default=256)
        parser.add_argument('--eval-block-size', type=int, default=512)
        parser.add_argument('--wwm', action='store_true', default=True)
        parser.add_argument('--mlm-probability', type=float, default=0.15)

        parser.add_argument('--batch-size', type=int, default=8)
        parser.add_argument('--update-cycle', type=int, default=8)
        parser.add_argument('--eval-batch-size', type=int, default=8)
        parser.add_argument('--train-steps', type=int, default=100000)
        parser.add_argument('--eval-steps', type=int, default=1000)
        parser.add_argument('--learning-rate', type=float, default=5e-5)

        parser.add_argument('--pretrained-model-path', type=str, default='bert-base-multilingual-cased')
        parser.add_argument('--save-model-path', type=str, default='models')

        parser.add_argument('--seed', type=int, default=42)
    ```
