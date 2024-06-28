from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import os

# Correctly links pyspark to python so the program doesn't get confused and shut down thinking there is no python
os.environ['PYSPARK_PYTHON'] = 'C:\\Users\\<<user>>\\AppData\\Local\\Programs\\Python\\Python311\\python.exe'
os.environ['PYSPARK_DRIVER_PYTHON'] = 'C:\\Users\\<<user>>\\AppData\\Local\\Programs\\Python\\Python311\\python.exe'

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Heart Attack Model") \
    .getOrCreate()

# Load Data
data = spark.read.csv("heart.csv", header=True, inferSchema=True)

# Data Preprocessing
feature_columns = ["age", "sex", "cp", "trtbps", "chol", "fbs", "restecg", "thalachh", "exng", "oldpeak", "slp", "caa",
                   "thall"]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
assembled_data = assembler.transform(data)

# Feature Scaling
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
scaler_model = scaler.fit(assembled_data)
scaled_data = scaler_model.transform(assembled_data)

# Select only the scaled features and the label
final_data = scaled_data.select(col("scaledFeatures").alias("features"), col("output").cast("double"))

# Split Data into Training and Testing Sets
train_data, test_data = final_data.randomSplit([0.8, 0.2], seed=1234)

# Train a Model
lr = LogisticRegression(labelCol="output", featuresCol="features")
lr_model = lr.fit(train_data)


# Function to predict a new patient
def predict_new_patient(patient_data):
    # Create a DataFrame for the new patient
    new_patient_df = spark.createDataFrame([Row(**patient_data)])

    # Preprocess the new patient data
    assembled_new_patient = assembler.transform(new_patient_df)
    scaled_new_patient = scaler_model.transform(assembled_new_patient)
    final_new_patient = scaled_new_patient.select(col("scaledFeatures").alias("features"))

    # Make predictions
    predictions = lr_model.transform(final_new_patient)
    predictions.show()


# Prompt the user for new patient data
new_patient = {
    "age": float(input("Enter age: ")),
    "sex": int(input("Enter sex (1 for male, 0 for female): ")),
    "cp": int(input("Enter chest pain type (0-3): ")),
    "trtbps": float(input("Enter resting blood pressure: ")),
    "chol": float(input("Enter serum cholesterol: ")),
    "fbs": int(input("Enter fasting blood sugar (1 if > 120 mg/dl, 0 otherwise): ")),
    "restecg": int(input("Enter resting electrocardiographic results (0-2): ")),
    "thalachh": float(input("Enter maximum heart rate achieved: ")),
    "exng": int(input("Enter exercise induced angina (1 for yes, 0 for no): ")),
    "oldpeak": float(input("Enter ST depression induced by exercise relative to rest: ")),
    "slp": int(input("Enter the slope of the peak exercise ST segment (0-2): ")),
    "caa": int(input("Enter number of major vessels (0-3) colored by fluoroscopy: ")),
    "thall": int(input("Enter thalassemia (0-3): "))
}

# Prediction for the new patient
predict_new_patient(new_patient)