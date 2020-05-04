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

def main():
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
        "Creating MongoDB sink",
        f"docker exec -it {PULSAR_CONTAINER} {PULSAR_ADMIN} sinks create --sink-type mongo --inputs tweet_records --name pulsar-mongodb-sink --sink-config-file ./connectors/pulsar-mongodb-sink.yaml",
    )

if __name__ == "__main__":
    main()
