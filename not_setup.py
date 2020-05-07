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
        f"docker exec -it {PULSAR_CONTAINER} {PULSAR_ADMIN} source create --name twitter --source-type twitter --destinationTopicName tweets --source-config {source_config}",
    )

    docker_exec(
        "Create english filter function",
        f"docker exec -it {PULSAR_CONTAINER} {PULSAR_ADMIN} functions create --py functions/filter_english.py --classname filter_english.FilterEnglish --inputs tweets --output persistent://public/default/english_tweets --tenant public --namespace default --name filter_english",
    )

    docker_exec(
        "Create sentiment hydration function",
        f"docker exec -it {PULSAR_CONTAINER} {PULSAR_ADMIN} functions create --py functions/add_sentiment.py --classname add_sentiment.AddSentiment --inputs english_tweets --output persistent://public/default/tweets_with_sentiment --tenant public --namespace default --name add_sentiment",
    )

    docker_exec(
        "Create db writer function",
        f"docker exec -it {PULSAR_CONTAINER} {PULSAR_ADMIN} functions create --py functions/write_to_db.py --classname write_to_db.TweetToRecord --inputs tweets_with_sentiment --tenant public --namespace default --name write_to_db",
    )

    docker_exec(
        "Create influxdb writer function",
        f"docker exec -it {PULSAR_CONTAINER} {PULSAR_ADMIN} functions create --py functions/write_to_influxdb.py --classname write_to_influxdb.TweetToInfluxdb --inputs tweets_with_sentiment --tenant public --namespace default --name write_to_influxdb",
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
