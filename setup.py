#!/usr/bin/env python
import os, sys
from distutils.core import setup

DOCKER_FOLDER_PATH = "./docker"
PULSAR_CONTAINER = "docker_pulsar_1"
PULSAR_ADMIN = "./bin/pulsar-admin"

def docker_exec(
    title,
    command,
):
    print("".join(["\n", title, "\n"]))
    print(command)
    os.system(command)

def verify_jdbc_connector_exists():
    if not os.path.isfile('./docker/connectors/pulsar-io-jdbc-2.5.0.nar'):
        sys.exit("'./docker/connectors/pulsar-io-jdbc-2.5.0.nar' is missing. Download it from https://archive.apache.org/dist/pulsar/pulsar-2.5.0/connectors/pulsar-io-jdbc-2.5.0.nar then run this script again")

def main():
    verify_jdbc_connector_exists()

    source_config = os.popen('source ./docker/twitter_conf && echo "\'{"consumerKey":${CONSUMER_KEY},"consumerSecret":${CONSUMER_SECRET},"token":${TOKEN},"tokenSecret":${TOKEN_SECRET}}\'"').read()

    docker_exec(
        "Creating Pulsar twitter source",
        f"docker exec -it {PULSAR_CONTAINER} {PULSAR_ADMIN} source create --name twitter --source-type twitter --destinationTopicName tweets2 --source-config {source_config}",
    )

    docker_exec(
        "Creating tweet record function",
        f"docker exec -it {PULSAR_CONTAINER} {PULSAR_ADMIN} functions create --py functions/step2.py --classname step2.TweetToRecord --inputs tweets2 --output persistent://public/default/tweet_records --tenant public --namespace default --name step2",
    )

    docker_exec(
        "Create tweet_records schema",
        f"docker exec -it {PULSAR_CONTAINER} {PULSAR_ADMIN} schemas upload -f ./schemas/tweet_schema tweet_records",
    )

    docker_exec(
        "Creating MySQL sink",
        f"docker exec -it {PULSAR_CONTAINER} {PULSAR_ADMIN} sinks create  --archive ./connectors/pulsar-io-jdbc-2.5.0.nar --inputs tweet_records --name pulsar-mysql-jdbc-sink --sink-config-file ./connectors/pulsar-mysql-jdbc-sink.yaml",
    )

if __name__ == "__main__":
    main()



setup(
    name='Streamer',
    version='1.0',
    description='Using pulsar and tensorflow as the backend of a twitter sentiment app',
    install_requires=[
        'tensorflow',
        'keras',
        'keras_preprocessing',
    ],
)
