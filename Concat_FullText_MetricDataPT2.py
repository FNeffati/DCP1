from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Combine Data").getOrCreate()

# read in all CSV files in directory and create dataframe
df = spark.read.option("header", "true").csv("/home/rblaha/JoinedData.csv/*")

# coalesce all partitions into one
df = df.coalesce(1)

# write out as a single CSV file
df.write.mode("overwrite").option("header", "true").csv("/home/rblaha/ConcatenatedData.csv")
