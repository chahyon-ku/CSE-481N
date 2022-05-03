# PREDICT ON DEV & TEST SET

# these values may be set larger for better performance
PATH_OF_SOURCE_SIDE_OF_DEV_SET=data/en-de/dev/dev.src
PATH_OF_TARGET_SIDE_OF_DEV_SET=data/en-de/dev/dev.mt
PATH_OF_WORD_LEVEL_TAGS_OF_DEV_SET=data/en-de/dev/dev.tags
PREDICT_N=40
PREDICT_M=6
PATH_OF_SAVED_CHECKPOINT=../daniel/pytorch_model.bin
PATH_OF_DEV_OUTPUT_OF_WORD_LEVEL_TAGS=word_level_tags.txt
PATH_OF_DEV_OUTPUT_OF_SENT_LEVEL_SCORE=sent_level_score.txt
PATH_OF_DEV_OUTPUT_OF_WORD_LEVEL_SCORE=word_level_score.txt
PATH_OF_THRESHOLD=threshold.txt

python -u predict.py \
    --test-src=$PATH_OF_SOURCE_SIDE_OF_DEV_SET \
    --test-tgt=$PATH_OF_TARGET_SIDE_OF_DEV_SET \
    --threshold-tune=$PATH_OF_WORD_LEVEL_TAGS_OF_DEV_SET \
    --wwm \
    --mc-dropout \
    --predict-n=$PREDICT_N \
    --predict-m=$PREDICT_M \
    --checkpoint=$PATH_OF_SAVED_CHECKPOINT \
    --word-output=$PATH_OF_DEV_OUTPUT_OF_WORD_LEVEL_TAGS \
    --sent-output=$PATH_OF_DEV_OUTPUT_OF_SENT_LEVEL_SCORE \
    --score-output=$PATH_OF_DEV_OUTPUT_OF_WORD_LEVEL_SCORE \
    --threshold-output=$PATH_OF_THRESHOLD

python -u predict.py \
    --test-src=$PATH_OF_SOURCE_SIDE_OF_TEST_SET \
    --test-tgt=$PATH_OF_TARGET_SIDE_OF_DEV_SET \
    --wwm \
    --mc-dropout \
    --predict-n=$PREDICT_N \
    --predict-m=$PREDICT_M \
    --checkpoint=$PATH_OF_SAVED_CHECKPOINT \
    --word-output=$PATH_OF_TEST_OUTPUT_OF_WORD_LEVEL_TAGS \
    --sent-output=$PATH_OF_TEST_OUTPUT_OF_SENT_LEVEL_SCORE \
    --score-output=$PATH_OF_TEST_OUTPUT_OF_WORD_LEVEL_SCORE \
    --threshold=$PATH_OF_THRESHOLD
