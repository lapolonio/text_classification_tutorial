# Enterprise Text Classification using BERT Tutorial

Step 0:
    * Explore BERT Repo to understand steps

Step 1:
    * Gather Data and modify BERT Repo for training needs
    * Download model and upload to google cloud bucket
    * Give account read and write access to bucket

Step 2:
    * Add code to support exporting model
    * Write serving_input_fn
    * Export built model to bucket


Run script:
```
export BERT_BASE_DIR=gs://bert_model_demo/uncased_L-12_H-768_A-12
export IMDB_DIR=NOT_USED
export TPU_NAME=grpc://10.92.118.162:8470
export OUTPUT_DIR=gs://bert_model_demo/imdb_v1/output/
export EXPORT_DIR=gs://bert_model_demo/imdb_v1/export/

python bert_repo/step_1/bert/run_classifier.py \
  --task_name=IMDB \
  --do_train=true \
  --do_predict=true \
  --data_dir=$IMDB_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT_DIR \
  --use_tpu=True \
  --tpu_name=$TPU_NAME \
  --do_serve=true \
  --export_dir=$EXPORT_DIR
```

### Results on Text Classification from https://github.com/zihangdai/xlnet

Model | IMDB | Yelp-2 | Yelp-5 | DBpedia | Amazon-2 | Amazon-5
--- | --- | --- | --- | --- | --- | ---
BERT-Large | 4.51 | 1.89 | 29.32 | 0.64 | 2.63 | 34.17
XLNet-Large | **3.79** | **1.55** | **27.80** | **0.62** | **2.40** | **32.26**

The above numbers are error rates.


Resources: 
    * https://ai.stanford.edu/~amaas/data/sentiment/
    * https://www.tensorflow.org/hub/tutorials/text_classification_with_tf_hub
    * https://colab.research.google.com/github/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb#scrollTo=SCZWZtKxObjh