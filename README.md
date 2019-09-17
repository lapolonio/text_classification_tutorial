# Enterprise Text Classification using BERT Tutorial

## Setup:

* Download Intellij Community (for diffing folders): <https://www.jetbrains.com/idea/download/>

## Step 0:

* Explore BERT Repo to understand steps

## Step 1: Add code to process data

* Download data and inspect format
* Modify BERT Repo for training needs
* Download base model and upload to google cloud bucket
* Give account read and write access to bucket

## Step 2: Add code to support export

* Add export flags
* Write serving_input_fn

## Step 3: Build and Export Model

* Train model

```{bash}
import sys
!test -d bert_repo || git clone https://github.com/lapolonio/text_classification_tutorial bert_repo
if not 'bert_repo' in sys.path:
  sys.path += ['bert_repo/step_3/bert']


export BERT_BASE_DIR=gs://bert_model_demo/uncased_L-12_H-768_A-12
export IMDB_DIR=NOT_USED
export TPU_NAME=grpc://10.92.118.162:8470
export OUTPUT_DIR=gs://bert_model_demo/imdb_v1/output/
export EXPORT_DIR=gs://bert_model_demo/imdb_v1/export/

python bert_repo/step_3/bert/run_classifier.py \
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
* Export built model to bucket
* Test exported model

```{bash}
!saved_model_cli show --dir gs://bert_model_demo/imdb_v1/export/1567569486 --tag_set serve --signature_def serving_default

# Execute model to understand model input format
# Batch 1
!saved_model_cli run --dir gs://bert_model_demo/imdb_v1/export/1567569486 --tag_set serve --signature_def serving_default \
--input_examples 'examples=[{"input_ids":np.zeros((128), dtype=int).tolist(),"input_mask":np.zeros((128), dtype=int).tolist(),"label_ids":[0],"segment_ids":np.zeros((128), dtype=int).tolist()}]'

# Batch 3
!saved_model_cli run --dir gs://bert_model_demo/imdb_v1/export/1567569486 --tag_set serve --signature_def serving_default \
--input_examples 'examples=[{"input_ids":np.zeros((128), dtype=int).tolist(),"input_mask":np.zeros((128), dtype=int).tolist(),"label_ids":[0],"segment_ids":np.zeros((128), dtype=int).tolist()},{"input_ids":np.zeros((128), dtype=int).tolist(),"input_mask":np.zeros((128), dtype=int).tolist(),"label_ids":[0],"segment_ids":np.zeros((128), dtype=int).tolist()},{"input_ids":np.zeros((128), dtype=int).tolist(),"input_mask":np.zeros((128), dtype=int).tolist(),"label_ids":[0],"segment_ids":np.zeros((128), dtype=int).tolist()}]'
```

##Step 4: Create serving image and test with python client

* Create tensorflow_serving image with model
* Deploy image to dockerhub or GCR
* Create python client to call tf serving image

```{bash}
./make_tensorflow_serving_container.sh
docker run -p 8500:8500 --name bert_serving lapolonio/tf_serving_bert_imdb:1567569486
cd bert
APP_CONFIG_FILE=config/development.py pipenv run python run_app.py

curl -X POST \
  http://localhost:5000/ \
  -H 'Content-Type: application/json' \
  -d '{"sentences":["Tainted look at kibbutz life<br /><br />This film is less a cultural story about a boy'\''s life in a kibbutz, but the deliberate demonization of kibbutz life in general. In the first two minutes of the movie, the milk man in charge of the cows rapes one of his calves. And it'\''s all downhill from there in terms of the characters representing typical '\''kibbutznikim'\''. Besides the two main characters, a clinically depressed woman and her young son, every one else in the kibbutz is a gross caricature of wellevil.", 
"A great story a young Aussie bloke travels to england to claim his inheritance and meets up with his mates, who are just as loveable and innocent as he is.",
"i hate the movie it was racist",
"i loved the movie it was inspiring."]}'
```

##Step 5: Create client image and deploy

* Create container for client
* Create k8s deployment
* Create k8s cluster
* Deploy to k8s cluster
* Test deployment

## Resources

### Results on Text Classification from https://github.com/zihangdai/xlnet

Model | IMDB | Yelp-2 | Yelp-5 | DBpedia | Amazon-2 | Amazon-5
--- | --- | --- | --- | --- | --- | ---
BERT-Large | 4.51 | 1.89 | 29.32 | 0.64 | 2.63 | 34.17
XLNet-Large | **3.79** | **1.55** | **27.80** | **0.62** | **2.40** | **32.26**

The above numbers are error rates.

### Links

* https://martinfowler.com/articles/cd4ml.html

* Learn Production-Level Deep Learning from Top Practitioners: <https://fullstackdeeplearning.com/>

* Nuts and Bolts of Applying Deep Learning (Andrew Ng): <https://www.youtube.com/watch?v=F1ka6a13S9I>

* http://jalammar.github.io/illustrated-bert/

* BERT Repo Classification Example: <https://github.com/google-research/bert#sentence-and-sentence-pair-classification-tasks>

* IMDB Data: <https://ai.stanford.edu/~amaas/data/sentiment/>

* Predicting Movie Reviews with BERT on TF Hub.ipynb: <https://colab.research.google.com/github/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb>

* BERT End to End (Fine-tuning + Predicting) in 5 minutes with Cloud TPU: <https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb>

* Saved Models Tutorial: <https://colab.research.google.com/github/tensorflow/docs/blob/r2.0rc/site/en/r2/guide/saved_model.ipynb#scrollTo=Dk5wWyuMpuHx>

* Use TensorFlow Serving with Kubernetes: <https://www.tensorflow.org/tfx/serving/serving_kubernetes#query_the_server>

* Example client communicating with Tensorflow Serving: <https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/resnet_client_grpc.py#L53>

* TensorFlow Serving with a variable batch size: https://www.damienpontifex.com/2018/05/10/tensorflow-serving-with-a-variable-batch-size/