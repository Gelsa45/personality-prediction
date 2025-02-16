import pandas as pd

data = pd.read_csv("datasets/Training_Essay_Data.csv")
print(data['generated'].value_counts())  # Check class distribution
