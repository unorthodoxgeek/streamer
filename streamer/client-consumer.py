import pulsar
import json

client = pulsar.Client('pulsar://localhost:6650')
consumer = client.subscribe('tweets',
                            subscription_name='tweets')

while True:
    msg = consumer.receive()
    print("Received message: '%s'" % msg.data())
    json_data = json.loads(msg.data())
    print(json_data)
    consumer.acknowledge(msg)

client.close()