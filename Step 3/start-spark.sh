#!/bin/bash

set -e

$SPARK_HOME/sbin/start-master.sh

sleep 5

MASTER_URL="spark://$(hostname):7077"

$SPARK_HOME/sbin/start-worker.sh "$MASTER_URL"

echo "Spark Master URL: $MASTER_URL"
echo "Spark REST is in port 6066."
echo "Starting Spark job spark_analysis.py..."

$SPARK_HOME/bin/spark-submit \
  --master "$MASTER_URL" \
  /app/spark_analysis.py

echo "Spark job finished. Shutting down Spark services..."

$SPARK_HOME/sbin/stop-worker.sh || true
$SPARK_HOME/sbin/stop-master.sh || true

echo "All done. Exiting container."