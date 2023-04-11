from pyspark.sql import SparkSession
from pyspark.sql.functions import col
# create SparkSession
spark = SparkSession.builder.appName("ComboData").getOrCreate()


fulltext_df = spark.read.csv("/home/rblaha/FullText.csv", header=True)
fulltext_df = fulltext_df.withColumnRenamed("_c0", "fulltext_c0")  # Renames _c0 column to fulltext_c0

metricdata_df = spark.read.csv("/home/rblaha/MetricData.csv", header=True)
metricdata_df = metricdata_df.withColumnRenamed("_c0", "metricdata_c0")  # Renames _c0 column to metricdata_c0

joined_df = fulltext_df.join(metricdata_df, (fulltext_df["Work ID"] == metricdata_df["ID"]))

# Dropping the duplicate column
joined_df = joined_df.drop('metricdata_c0')

# Write output file
joined_df.write.csv("/home/rblaha/JoinedData.csv", header=True)
