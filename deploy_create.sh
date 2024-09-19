#/bin/sh
# set environment valiables
PROJECT_ID=uragasumi25-llm-app-pj
REGION=asia-northeast1
AR_REPO=llm-meeting-batch
SERVICE_NAME=llm-meeting-batch
SA_NAME=sa-llm-app
DB_USER=root
DB_PASS=1234abcd!
DB_NAME=llm-app-db
INSTANCE_NAME=llm-app-db
INSTANCE_CONNECTION_NAME='uragasumi25-llm-app-pj:asia-northeast1:llm-app-db'
GOOGLE_API_KEY='AIzaSyAl57m8ECZw-JfuWzec_EI8ft1IszppBPs'

# PUSH to Artifact Registry
gcloud builds submit --tag $REGION-docker.pkg.dev/$PROJECT_ID/$AR_REPO/$SERVICE_NAME \
  --project=$PROJECT_ID

# create to Cloud Run
gcloud run jobs create llm-meeting-batch \
 --image $REGION-docker.pkg.dev/$PROJECT_ID/$AR_REPO/$SERVICE_NAME \
 --service-account=$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com \
 --region=$REGION \
 --set-env-vars=PROJECT_ID=$PROJECT_ID,LOCATION=$REGION \
 --project=$PROJECT_ID \
 --tasks=1 \
 --cpu=1 \
 --max-retries=0 \
 --memory=512Mi \
 --parallelism=1 \
 --task-timeout=100 \
 --set-env-vars INSTANCE_CONNECTION_NAME=$INSTANCE_CONNECTION_NAME \
 --set-env-vars DB_USER=$DB_USER \
 --set-env-vars DB_PASS=$DB_PASS \
 --set-env-vars DB_NAME=$DB_NAME \
 --set-env-vars GOOGLE_API_KEY=$GOOGLE_API_KEY 

gcloud run jobs execute llm-meeting-batch --region=asia-northeast1