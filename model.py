from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Heart Attack Model") \
    .getOrCreate()

# Load Data
data = spark.read.csv("heart.csv", header=True, inferSchema=True)
data.show()

# Data Preprocessing
feature_columns = ["age", "sex", "cp", "trtbps", "chol", "fbs", "restecg", "thalachh", "exng", "oldpeak", "slp", "caa", "thall"]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
assembled_data = assembler.transform(data)

# Select only the features and the label
final_data = assembled_data.select("features", "output")

# Split Data into Training and Testing Sets
train_data, test_data = final_data.randomSplit([0.8, 0.2], seed=1234)

# Train Model
lr = LogisticRegression(labelCol="output", featuresCol="features")
lr_model = lr.fit(train_data)

# Make Predictions
predictions = lr_model.transform(test_data)
predictions.show()

# Evaluate the Model
evaluator = MulticlassClassificationEvaluator(labelCol="output", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy}")



# For clarity Age is age of patient, sex is gender of patient (1 = male, 0 = female), cp is Chest pain type (0: asymptomatic, 1: atypical anngina, 2: non-anginal pain, 3: typical angina,
# trtbps is resting blood pressure, chol is Serum cholesterol in mg-dl, fbs is fasting blood suger > 120 mg/dl (1 = true, 0 = false),
# restecg is resting electrocardiographic results (0 = normal, 1 = ST-T wave abnormality,  2 = left ventricular hypertrophy by the criteria of Estes),
# thalachh is maximum heart rate achieved, exng is exercise induce angina (1 = yes, 0 = no), oldpeak is ST depression induced by exercive relative to rest,
# slp is slope of the peak exercise ST segment, caa is number of major vessels (0-3) colored by fluoroscopy, thall is Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect),
# output is diagnosis of heart disease (1 = presence; 0 = absence)




