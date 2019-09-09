# Create Tensorflow Serving Container and host on Dockerhub
MODEL_LOCATION=gs://bert_model_demo/imdb_v1/export/1567569486
IMAGE_NAME=tf_serving_bert_imdb
VER=1567569486
MODEL_NAME=bert
DOCKER_USER=lapolonio

docker login docker.io

cd ~
docker run -d --name $IMAGE_NAME tensorflow/serving
mkdir ~/models
gsutil cp -r  $MODEL_LOCATION ~/models
docker cp ~/models/ $IMAGE_NAME:/models/$MODEL_NAME/
docker commit --change "ENV MODEL_NAME $MODEL_NAME" $IMAGE_NAME $USER/$IMAGE_NAME
docker tag $USER/$IMAGE_NAME $DOCKER_USER/$IMAGE_NAME:$VER
docker push $DOCKER_USER/$IMAGE_NAME:$VER