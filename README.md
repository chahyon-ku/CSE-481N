# Reproducing "Self-Supervised Quality Estimation for Machine Translation" by Zheng et al.

## Introduction

Adapted and annotated the original author's code from https://github.com/THUNLP-MT/SelfSupervisedQE as part of
the reproducibility project for CSE 481 (NLP Capstone) at University of Washington during spring quarter of
2022.

Team Name: Team TBD

List of Members: Chahyon Ku, Daniel Cheng, Sherry Zhao, Shubhkarman Singh

## Instructions

1. Clone repo


2. Install Dependencies (Ours / Author's)

   ```
   python (3.9.12 / 3.6)
   
   pytorch (1.11.0 / >=1.4.0)
   
   transformers (4.11.3 / >=4.4.2)
   
   pandas (1.4.2 / >=1.0.5)
   ```
3. Download pre-trained multilingual BERT from Hugging Face (https://huggingface.co/bert-base-multilingual-cased)
   and save it under './bert-base-multilingual-cased/'


4. Fine-tune the model by running train.py

    ```
    python -u train.py \
        --train-src 'data/en-de/train/train.en' \
        --train-tgt 'data/en-de/train/train.de' \

        --dev-src 'data/en-de/dev/dev.src' \
        --dev-tgt 'data/en-de/dev/dev.mt' \
        --dev-hter 'data/en-de/dev/dev.hter' \
        --dev-tags 'data/en-de/dev/dev.tags' \

        --block-size 256 \
        --eval-block-size 512 \
        --wwm True \
        --mlm-probability 0.15 \

        --batch-size 128 \
        --update-cycle 1 \
        --eval-batch-size 8 \
        --train-steps 100000
        --eval-steps 1000 \
        --learning-rate 5e-5 \

        --pretrained-model-path './bert-base-multilingual-cased/' \
        --save-model-path './models/en-de/' \

        --seed 42
    ```
   - Above parameters are what the authors of the original paper used to produce results.
   - To reduce batch-size, multiply the update-cycle by the factor to maintain consistent effective batch size.
     - For example, a batch-size of 16 and update-cycle of 8 would accumulate gradients for 8 batches, and
       update the weights every 128 data points, simulating a batch size of 128.


5. Save predictions and thresholds by running predict.py on dev set

   ```
   python -u predict.py \
      --test-src './data/en-de/dev/dev.src' \
      --test-tgt './data/en-de/dev/dev.mt' \
      --threshold-tune './data/en-de/dev/dev.tags' \
   
      --block-size', type=int, default=512 \
      --wwm', action='store_true', default=True \
      --predict-n', type=int, default=40 \
      --predict-m', type=int, default=6 \
      --batch-size', type=int, default=20 \
      --mc-dropout', action='store_true', default=True \
   
      --checkpoint './models/en-de/checkpoint_best' \
   
      --seed', type=int, default=42 \
      --output-dir './models/en-de/dev/'
   
   ```
   
6. Save predictions by running predict.py on test set

   ```
   python -u predict.py \
      --test-src './data/en-de/test/test.src' \
      --test-tgt './data/en-de/test/test.mt' \
      --threshold' './models/en-de/dev/threshold.txt' \
   
      --block-size', type=int, default=512 \
      --wwm', action='store_true', default=True \
      --predict-n', type=int, default=40 \
      --predict-m', type=int, default=6 \
      --batch-size', type=int, default=20 \
      --mc-dropout', action='store_true', default=True \
   
      --checkpoint './models/en-de/checkpoint_best' \
   
      --seed', type=int, default=42 \
      --output-dir './models/en-de/test/'   
   ```

7. Compute scores for the predictions by running compute_scores.py
   
   ```
   python -u compute_scores.py \
      --test-hter './data/en-de/dev/dev.hter' \
      --test-tags './data/en-de/dev/dev.tags' \
   
      --sent-output './models/en-de/dev/sent_output.txt' \
      --word-output './models/en-de/dev/word_output.txt'
   
   python -u compute_scores.py \
      --test-hter './data/en-de/test/test.hter' \
      --test-tags './data/en-de/test/test.tags' \
   
      --sent-output './models/en-de/test/sent_output.txt' \
      --word-output './models/en-de/test/word_output.txt'
   ```
   - 'Pearson' is the sentence-level metric, F1-OK, F1-BAD, and F1-MUL are the word-level metrics.
