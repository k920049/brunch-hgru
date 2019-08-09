gcloud compute --project=gcp-tensorflow-222205 instances create brunch-gensim --zone=asia-east1-a --machine-type=n1-highcpu-64 --subnet=default --network-tier=PREMIUM --maintenance-policy=MIGRATE --service-account=brunch@gcp-tensorflow-222205.iam.gserviceaccount.com --scopes=https://www.googleapis.com/auth/cloud-platform --tags=network-gcp-spark,http-server,https-server --image=ubuntu-minimal-1604-xenial-v20190628 --image-project=ubuntu-os-cloud --boot-disk-size=128GB --boot-disk-type=pd-ssd --boot-disk-device-name=brunch-gensim

gcloud dataproc clusters create brunch-spark \
    --bucket dataproc-7e10897a-5391-4ea0-b815-f6e72cf284f7-asia-east1 \
    --region asia-east1\
    --subnet default\
    --zone asia-east1-c\
    --master-machine-type n1-highmem-16\
    --master-boot-disk-size 500 \
    --num-workers 2 \
    --worker-machine-type n1-highmem-16 \
    --worker-boot-disk-size 500 \
    --image-version 1.3-deb9 \
    --scopes 'https://www.googleapis.com/auth/cloud-platform' \
    --tags network-gcp-spark \
    --project gcp-tensorflow-222205 \
    --properties capacity-scheduler:yarn.scheduler.capacity.resource-calculator=org.apache.hadoop.yarn.util.resource.DominantResourceCalculator \
    --initialization-actions 'gs://dataproc-initialization-actions/jupyter/jupyter.sh'