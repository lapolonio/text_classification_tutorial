# Storage for models
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-repo-storage
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
EOF

cat <<EOF | kubectl apply -f -
kind: Pod
apiVersion: v1
metadata:
  name: model-load-data
spec:
  volumes:
    - name: model-repo
      persistentVolumeClaim:
        claimName: model-repo-storage
  containers:
    - name: model-load-data
      image: ubuntu
      command: ["/bin/bash", "-ecx", "while :; do printf '.'; sleep 5 ; done"]
      volumeMounts:
      - name: model-repo
        mountPath: /models
EOF

MODEL_LOCATION=gs://bert_model_demo/imdb_v1/export/1567569486
MODEL_NAME=bert
mkdir ~/models
gsutil cp -r  $MODEL_LOCATION ~/models
kubectl cp ~/models/ model-load-data:/models/$MODEL_NAME/