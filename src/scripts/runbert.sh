#!/usr/bin/env bash
# pip install torch==1.6.0 torchvision==0.7.0
#SRC=/root/topic_classification_augmentation/TransformersDataAugmentation/src

source /root/topic_classification_augmentation/cbert_aug/env_cbert/bin/activate
CACHE=/root/topic_classification_augmentation/TransformersDataAugmentation/CACHE
TASK=${1}
MAXLEN=100

python ./utils/create_fsl_dataset.py -datadir ./utils/datasets/${TASK} -num_train 100 -num_dev 500 -sim 20 


for i in $(seq 0 20);
#for i in 7
    do
    RAWDATADIR=./utils/datasets/${TASK}/exp_${i}_100
    echo "===>"
    echo ${RAWDATADIR}

   # Baseline classifier
    python ./bert_aug/bert_classifier.py --task $TASK  --data_dir $RAWDATADIR --seed ${i}  --max_seq_length ${MAXLEN} --cache $CACHE > ${TASK}_${i}_bert_baseline.log

  ##############
  ## EDA
  ##############

  EDADIR=$RAWDATADIR/eda
  mkdir $EDADIR
  python ./bert_aug/eda.py --input $RAWDATADIR/train.tsv --output $EDADIR/eda_aug.tsv --num_aug=1 --alpha=0.1 --seed ${i}
  cat $RAWDATADIR/train.tsv $EDADIR/eda_aug.tsv > $EDADIR/train.tsv
  cp $RAWDATADIR/test.tsv $EDADIR/test.tsv
  cp $RAWDATADIR/dev.tsv $EDADIR/dev.tsv
  python ./bert_aug/bert_classifier.py --task $TASK --data_dir $EDADIR --seed ${i}  --cache $CACHE  --max_seq_length ${MAXLEN}  > ${TASK}_${i}_bert_eda.log


    #######################
    # GPT2 Classifier
    #######################

    GPT2DIR=$RAWDATADIR/gpt2
    mkdir $GPT2DIR
    python ./bert_aug/cgpt2.py --data_dir $RAWDATADIR --output_dir $GPT2DIR --task_name $TASK  --num_train_epochs 25 --seed ${i} --top_p 0.9 --temp 1.0 --cache $CACHE
    cat $RAWDATADIR/train.tsv $GPT2DIR/cmodgpt2_aug_3.tsv > $GPT2DIR/train.tsv
    cp $RAWDATADIR/test.tsv $GPT2DIR/test.tsv
    cp $RAWDATADIR/dev.tsv $GPT2DIR/dev.tsv
    python ./bert_aug/bert_classifier.py --task $TASK --data_dir $GPT2DIR --seed ${i} --cache $CACHE --max_seq_length ${MAXLEN}  > ${TASK}_${i}_bert_gpt2_3.log

    #    #######################
    #    # Backtranslation DA Classifier
    #    #######################

    BTDIR=$RAWDATADIR/bt
    mkdir $BTDIR
    python ./bert_aug/backtranslation.py --data_dir $RAWDATADIR --output_dir $BTDIR --task_name $TASK  --seed ${i} --cache $CACHE
    cat $RAWDATADIR/train.tsv $BTDIR/bt_aug.tsv > $BTDIR/train.tsv
    cp $RAWDATADIR/test.tsv $BTDIR/test.tsv
    cp $RAWDATADIR/dev.tsv $BTDIR/dev.tsv
    python ./bert_aug/bert_classifier.py --task $TASK --data_dir $BTDIR --seed ${i} --cache $CACHE  --max_seq_length ${MAXLEN}  > ${TASK}_${i}_bert_bt.log

    # #######################
    # # CBERT Classifier
    # #######################

    CBERTDIR=$RAWDATADIR/cbert
    mkdir $CBERTDIR
    python ./bert_aug/cbert.py --data_dir $RAWDATADIR --output_dir $CBERTDIR --task_name $TASK  --num_train_epochs 10 --seed ${i}  --cache $CACHE > $RAWDATADIR/cbert.log
    cat $RAWDATADIR/train.tsv $CBERTDIR/cbert_aug.tsv > $CBERTDIR/train.tsv
    cp $RAWDATADIR/test.tsv $CBERTDIR/test.tsv
    cp $RAWDATADIR/dev.tsv $CBERTDIR/dev.tsv
    python ./bert_aug/bert_classifier.py --task $TASK --data_dir $CBERTDIR --seed ${i} --cache $CACHE --max_seq_length ${MAXLEN} > ${TASK}_${i}_bert_cbert.log

    # #######################
    # # CMODBERT Classifier
    # ######################

    CMODBERTDIR=$RAWDATADIR/cmodbert
    mkdir $CMODBERTDIR
    python ./bert_aug/cmodbert.py --data_dir $RAWDATADIR --output_dir $CMODBERTDIR --task_name $TASK  --num_train_epochs 150 --learning_rate 0.00015 --seed ${i} --cache $CACHE > $RAWDATADIR/cmodbert.log
    cat $RAWDATADIR/train.tsv $CMODBERTDIR/cmodbert_aug.tsv > $CMODBERTDIR/train.tsv
    cp $RAWDATADIR/test.tsv $CMODBERTDIR/test.tsv
    cp $RAWDATADIR/dev.tsv $CMODBERTDIR/dev.tsv
    python ./bert_aug/bert_classifier.py --task $TASK --data_dir $CMODBERTDIR --seed ${i} --cache $CACHE --max_seq_length ${MAXLEN}  > ${TASK}_${i}_bert_cmodbert.log

    # #######################
    # # CMODBERTP Classifier
    # ######################

    CMODBERTPDIR=$RAWDATADIR/cmodbertp
    mkdir $CMODBERTPDIR
    python ./bert_aug/cmodbertp.py --data_dir $RAWDATADIR --output_dir $CMODBERTPDIR --task_name $TASK  --num_train_epochs 10 --seed ${i} --cache $CACHE > $RAWDATADIR/cmodbertp.log
    cat $RAWDATADIR/train.tsv $CMODBERTPDIR/cmodbertp_aug.tsv > $CMODBERTPDIR/train.tsv
    cp $RAWDATADIR/test.tsv $CMODBERTPDIR/test.tsv
    cp $RAWDATADIR/dev.tsv $CMODBERTPDIR/dev.tsv
    python ./bert_aug/bert_classifier.py --task $TASK --data_dir $CMODBERTPDIR --seed ${i}  --cache $CACHE --max_seq_length ${MAXLEN}  > ${TASK}_${i}_bert_cmodbertp.log

done



