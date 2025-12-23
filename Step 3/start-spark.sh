#!/bin/bash

$SPARK_HOME/sbin/start-master.sh \

sleep 2

MASTER_URL="spark://$(hostname):7077"

$SPARK_HOME/sbin/start-worker.sh \
  $MASTER_URL

echo "Spark Master URL: $MASTER_URL"

echo "Spark REST is in port 6066."

tail -f /dev/null