
import grpc
import tensorflow as tf
import numpy as np

import run_classifier as classifiers
import tokenization

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import types_pb2

from flask import Flask
from flask import request
import time

import pandas as pd
import random

app = Flask(__name__)
app.config.from_envvar('APP_CONFIG_FILE')

@app.route("/", methods = ['GET'])
def hello():
  return "Hello BERT predicting IMDB! Try posting a string to this url"


@app.route("/", methods = ['POST'])
def predict():
  app.logger.info("Entering Predict Method")
  # MODEL PARAMS
  max_seq_length = 128

  channel = grpc.insecure_channel(app.config["TF_SERVING_HOST"])
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

  # Parse Description
  tokenizer = tokenization.FullTokenizer(
    vocab_file="asset/vocab.txt", do_lower_case=True)
  processor = classifiers.ImdbProcessor()
  label_list = processor.get_labels()
  content = request.get_json()
  request_id = str(random.randint(1, 9223372036854775807))
  
  data = {}
  data["id"] = []
  data["sentence"] = []
  for sentence in content['sentences']:
    data["id"].append(request_id)
    data["sentence"].append(sentence)

  inputExamples = processor._create_examples(pd.DataFrame.from_dict(data), 'test')

  t0 = time.time()
  model_input = classifiers.memory_based_convert_examples_to_features(inputExamples, label_list, max_seq_length, tokenizer)
  t1 = time.time()
  total = t1-t0
  app.logger.info("Convert Examples Time Taken: %d", total)

  app.logger.info(model_input)
  # Send request
  # See prediction_service.proto for gRPC request/response details.
  model_request = predict_pb2.PredictRequest()
  model_request.model_spec.name = 'bert'
  model_request.model_spec.signature_name = 'serving_default'

  model_request.inputs['examples'].CopyFrom(
    tf.contrib.util.make_tensor_proto(model_input, shape=[len(model_input)])
  )
  app.logger.info(model_request)

  t0 = time.time()
  result = stub.Predict(model_request, 60.0)  # 20 secs timeout
  t1 = time.time()
  total = t1-t0

  app.logger.info(result)
  result = tf.make_ndarray(result.outputs["probabilities"])
  app.logger.info(result)
  pretty_result = "Predicted Label: " + str(np.take(processor.get_labels(), result.argmax(axis=1)))
  app.logger.info("Predicted Label: %s", str(np.take(processor.get_labels(), result.argmax(axis=1))))
  app.logger.info("Predict Time Taken: %f", total)

  return pretty_result


if __name__ == '__main__':
  app.run(debug=app.config["DEBUG"])
