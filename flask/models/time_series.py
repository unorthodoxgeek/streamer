from influxdb import InfluxDBClient

client = InfluxDBClient('influx_db', 8086, '', '', 'tweets')

def get_time_series():
    return client.query("""
        SELECT mean("value") AS "mean_sentiment"
        FROM tweets
        WHERE time > now() - 1h AND time < now() - 2m
        GROUP BY time(10s)
        ORDER BY time DESC
        LIMIT 200;""")
