from influxdb import InfluxDBClient

client = InfluxDBClient('influx_db', 8086, 'root', 'root', 'example')

def get_time_series():
    result = client.query("""
        SELECT mean("sentiment") AS "mean_sentiment"
        FROM tweets
        WHERE time > now() - 1h
        GROUP BY time(10s)
        ORDER BY time DESC
        LIMIT 200;""")
    return result
