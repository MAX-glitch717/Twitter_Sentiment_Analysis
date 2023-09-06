#!/usr/bin/env python
# coding: utf-8

# ---
# ---
# 
# <center><h1>Spark ML Assignment</h1></center>
# <center><h1>PRACHI PATEL </h1></center>
#     
#     
# ---
# 
# ---

# In[1]:


from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext
from pyspark.ml.feature import Tokenizer, HashingTF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.sql import Row


# In[2]:


spark = SparkSession.builder \
    .appName("Python Spark") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()


# In[3]:


spark = SparkSession.builder.appName("TwitterSentimentAnalysis").getOrCreate()


# In[4]:


# Create SparkContext
sc = SparkContext.getOrCreate()
sc.setLogLevel("ERROR")

# Create SQLContext
sqlContext = SQLContext(sc)


# In[5]:


# Load the training data
dataframe = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('train.csv')



# In[6]:


dataframe.show()


# In[7]:


# Perform basic analysis on the data
print(f'Number of rows in the dataset: {dataframe.count()}')
print('Percentage of each class:')
dataframe.groupBy('label').count().show()


# In[8]:


# Tokenizer and HashingTF are transformers
tokenizer = Tokenizer(inputCol="tweet", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")

# LogisticRegression is an estimator
lr = LogisticRegression(maxIter=10, regParam=0.001)

# Create Pipeline
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])

# Fit the pipeline to training data.
model = pipeline.fit(dataframe)


# In[9]:


# Create StreamingContext with batch interval of 3 seconds
ssc = StreamingContext(sc, 3)

# Specify the host and port to read data from
lines = ssc.socketTextStream("localhost", 9999)

words = lines.flatMap(lambda line: line.split(" "))


# In[10]:


def process_rdd(time, rdd):
    print("========= %s =========" % str(time))
    try:
        # Convert RDD to DataFrame
        row_rdd = rdd.map(lambda word: Row(tweet=word))
        words_data_frame = spark.createDataFrame(row_rdd)

        # Make predictions
        prediction = model.transform(words_data_frame)

        # Show predictions
        prediction.show()

    except Exception as e:
        pass  # You could print the exception here if needed

# Apply this function to each RDD in the DStream
words.foreachRDD(process_rdd)

ssc.start()
ssc.awaitTermination()


# ---
# <center><h1>END</h1></center>
# 
# ---
# 
