import pyspark

sc = pyspark.SparkContext() 

def add(x, y):
    return x + y

# Some EDA on the FullText.csv:

# What is the average word length in the FullText.csv

def wordlength(word):
    return len(word)

set1 = set1=sc.textFile("/home/rblaha/FullText.csv")
set2 = set1.flatMap(lambda x: x.split(" "))
set3 = set2.map(wordlength)
total_words = set3.count()
total_length = set3.reduce(lambda x, y: x + y)
answer2 = total_length / total_words
print(answer2)

# Average word Len is 4.566

# Most common words:

def wordcount(word):
    return (word, 1)

set1 = set1=sc.textFile("/home/rblaha/FullText.csv")
set2 = set1.flatMap(lambda x: x.split(" "))
set3 = set2.map(wordcount)
set4 = set3.reduceByKey(lambda x, y: x + y)
set5 = set4.map(lambda x: (x[1], x[0]))
set6 = set5.sortByKey(False)
answer3 = set6.first()
print(answer3)

# The most common word (6563, 'the')

# Least common words:

def wordcount(word):
    return (word, 1)

set1 = set1=sc.textFile("/home/rblaha/FullText.csv")
set2 = set1.flatMap(lambda x: x.split(" "))
set3 = set2.map(wordcount)
set4 = set3.reduceByKey(lambda x, y: x + y)
set5 = set4.map(lambda x: (x[1], x[0]))
set6 = set5.sortByKey(True)
answer4 = set6.first()
print(answer4)

#

# (1, '45663400,"No') which seems like a typo

# What is the average number of words per line in the FullText.csv

def wordcount(word):
    return (word, 1)
    
set1 = set1=sc.textFile("/home/rblaha/FullText.csv")
set2 = set1.map(lambda x: x.split(" "))
set3 = set2.map(lambda x: len(x))
total_lines = set3.count()
total_words = set3.reduce(lambda x, y: x + y)
answer5 = total_words / total_lines 
print(answer5)

# 54.65760111576011


# What is the longest sentence in the FullText.csv

def wordcount(word):
    return (word, 1)

set1 = set1=sc.textFile("/home/rblaha/FullText.csv")
set2 = set1.map(lambda x: x.split(" "))
set3 = set2.map(lambda x: len(x))
answer6 = set3.max()
print(answer6)

# 7081

# What is the shortest sentence in the FullText.csv

def wordcount(word):
    return (word, 1)

set1 = set1=sc.textFile("/home/rblaha/FullText.csv")
set2 = set1.map(lambda x: x.split(" "))
set3 = set2.map(lambda x: len(x))
answer7 = set3.min()
print(answer7)

# 1

# What is the average number of words per sentence in the FullText.csv

def wordcount(word):
    return (word, 1)

set1 = set1=sc.textFile("/home/rblaha/FullText.csv")
set2 = set1.map(lambda x: x.split(" "))
set3 = set2.map(lambda x: len(x))
total_lines = set3.count()
total_words = set3.reduce(lambda x, y: x + y)
answer8 = total_words / total_lines
print(answer8)

