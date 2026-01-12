FROM eclipse-temurin:11-jdk

ENV SPARK_VERSION=3.4.2
ENV HADOOP_VERSION=3
ENV SPARK_HOME=/app/spark
ENV PATH="$SPARK_HOME/bin:$PATH"
ENV PYTHONPATH=/app/spark/python:/app/spark/python/lib
ENV PYSPARK_PYTHON=/usr/bin/python3
ENV PYSPARK_DRIVER_PYTHON=/usr/bin/python3

RUN mkdir -p /app

RUN apt-get update && apt-get install -y curl python3 python3-pip procps && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir --break-system-packages -r /app/requirements.txt

RUN curl -L https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    -o spark.tgz && \
    tar xzf spark.tgz -C /app && \
    mv /app/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} ${SPARK_HOME} && \
    rm spark.tgz

COPY spark-defaults.conf $SPARK_HOME/conf/spark-defaults.conf

COPY start-spark.sh /app/start-spark.sh
RUN chmod +x /app/start-spark.sh

COPY spark_analysis.py /app/spark_analysis.py

EXPOSE 8080 8081 6066 7077

WORKDIR /app
CMD ["/app/start-spark.sh"]	