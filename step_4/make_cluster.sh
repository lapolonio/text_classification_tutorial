export PROJECT_ID=analog-subset-256304
export CLUSTER_NAME=bert-cluster1

gcloud container clusters create $CLUSTER_NAME --zone us-central1-a --num-nodes 1 --machine-type n1-standard-8 --accelerator type=nvidia-tesla-v100
gcloud config set container/cluster $CLUSTER_NAME
gcloud container clusters get-credentials $CLUSTER_NAME --project $PROJECT_ID
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml

