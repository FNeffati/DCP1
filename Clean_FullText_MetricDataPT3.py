from pyspark.sql.functions import min, max, avg
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("COMBOCSV").getOrCreate()

data = spark.read.csv("COMBODATAAAA.csv", header=True, inferSchema=True)

# drop unwanted columns
data = data.drop("Date_updated", "Rating", "Pairing", "Warning", "Language", "Num_comments", "Num_kudos", "Num_bookmarks", "Num_hits")

# group by author and create new columns
agg_cols = [
    min("Word_count").alias("Min_Word_count"), 
    max("Word_count").alias("Max_Word_count"), 
    avg("Word_count").alias("Avg_Word_count"), 
    min("Num_chapters").alias("Min_Paragraph_Count"), 
    max("Num_chapters").alias("Max_Paragraph_Count"), 
    avg("Num_chapters").alias("Avg_Paragraph_Count")
]
updated_data = data.groupBy("Author").agg(*agg_cols)

updated_data.write.csv("UPDATEDCOMBODATAAAA.csv", header=True)

# show the results
updated_data.show()
