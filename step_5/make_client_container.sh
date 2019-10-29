git clone https://github.com/lapolonio/bert.git
cd ~/bert

CLIENT_IMAGE_NAME=imdb_client
CLIENT_VER=v3
DOCKER_USER=lapolonio
mkdir asset
gsutil cp gs://cloud-tpu-checkpoints/bert/uncased_L-12_H-768_A-12/vocab.txt asset/
docker build -t $USER/$CLIENT_IMAGE_NAME .
docker tag $USER/$CLIENT_IMAGE_NAME $DOCKER_USER/$CLIENT_IMAGE_NAME:$CLIENT_VER
docker push $DOCKER_USER/$CLIENT_IMAGE_NAME:$CLIENT_VER
