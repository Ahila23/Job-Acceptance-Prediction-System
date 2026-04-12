import pandas as pd
import pymysql

import sqlalchemy as sqlalchemy 
from sqlalchemy import create_engine


engine = create_engine("mysql+pymysql://root:12345@localhost:3306/Job Acceptance Prediction System")

# CSV file path
file_path = r"C:\Users\User\Desktop\Mini project 3\cleaned_dataset.csv"

# Load CSV
df = pd.read_csv(file_path)

print("Dataset loaded")
print("Shape:", df.shape)

# MySQL connection
engine = create_engine("mysql+pymysql://root:12345@localhost:3306/Job_Acceptance_Prediction_System")

# Upload dataframe to MySQL
df.to_sql("job_acceptance_prediction", engine, if_exists="replace", index=False)

print("Data uploaded to MySQL successfully!")







