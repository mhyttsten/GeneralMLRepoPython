JOB_NAME=worker  # ps or worker
TASK_INDEX=0  # start with 0

python CDiscount_Retrain_Distributed.py \
     --ps_hosts=35.224.208.176:2222 \
     --worker_hosts=35.192.148.52:2222,35.203.160.176:2222,35.199.35.144:2222,35.227.120.91:2222,35.227.120.91:2222,35.197.218.233:2222 \
     --job_name=$JOB_NAME --task_index=$TASK_INDEX
