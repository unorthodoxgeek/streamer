from pulsar import Function

class Step1(Function):
    def __init__(self):
        self.twitter_topic = "persistent://public/default/twitter"

    def process(self, item, context):
        warning = "The item {0}".format(item)
        context.get_logger().warn(warning)
