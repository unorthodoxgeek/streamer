import pulsar

client = pulsar.Client('pulsar://localhost:6650')
consumer = client.subscribe('tweets',
                            subscription_name='twitter_feed')

while True:
    msg = consumer.receive()
    print("Received message: '%s'" % msg.data())
    consumer.acknowledge(msg)

client.close()