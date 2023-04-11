import pyspark as spark
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegressionModel , LogisticRegression
from pyspark.ml import Pipeline
from pyspark.sql import Row
from pyspark.sql.types import *
from pyspark.sql.functions import lit
from pyspark.sql.functions import col, struct, concat_ws

spark = spark.sql.SparkSession.builder.appName("LogisticRegression").getOrCreate()

# Load up our data and convert it to the format MLLib uses
def parseInput(line):
    fields = line.split(',')
    if len(fields) != 2 or not fields[0] or not fields[1]:
        return None
    return Row(label=fields[0], text=fields[1].lower())

# Read a sample of the input data
dataset = (spark.read.text("/home/rblaha/FullText.csv") # Replace with your own path. 
           .sample(False, 0.003)  # Sample 0.3% of the data because anything more will crash 
           .rdd.map(lambda r: r[0])
           .map(parseInput)
           .filter(lambda x: x is not None)
           .toDF()
           .repartition(100))



# Tokenize each sentence into words
tokenizer = Tokenizer(inputCol="text", outputCol="words")
wordsData = tokenizer.transform(dataset)

# Convert the string labels to numerical labels
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(dataset)
indexedData = labelIndexer.transform(wordsData)

# Remove stop words
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
wordsData = remover.transform(indexedData)

# Count up the occurrences of each word
cv = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=10000, minDF=5)
model = cv.fit(wordsData)
result = model.transform(wordsData)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = result.randomSplit([0.7, 0.3])

# Train a LogisticRegression model
lr = LogisticRegression(maxIter=100, regParam=0.5, elasticNetParam=0, labelCol="indexedLabel")
lrModel = lr.fit(trainingData)

# Make predictions on test data
predictions = lrModel.transform(trainingData)

# Select example rows to display.
predictions.select("text","probability","label","prediction").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g " % (1.0 - accuracy))

# Print the coefficients and intercept for logistic regression
print("Coefficients\n" + str(lrModel.coefficientMatrix))
print("Intercept: " + str(lrModel.interceptVector))

# Overwrite the model
lrModel.write().overwrite().save("/home/rblaha/logistic_regression_model") # Replace with your own path


#Load the trained model
lrModel = LogisticRegressionModel.load("/home/rblaha/logistic_regression_model") # Replace with your own path

# Make predictions on test data
predictions = lrModel.transform(testData)

# Select example rows to display.
predictions.select("text","probability","label","prediction").show(5)


# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g " % (1.0 - accuracy))

# Print the coefficients and intercept for logistic regression
print("Coefficients\n" + str(lrModel.coefficientMatrix))
print("Intercept: " + str(lrModel.interceptVector))
