import pandas as pd

# Load the dataset
df = pd.read_csv("C:/Users/gelsa/Personality_project/datasets/Training_Essay_Data.csv")

# Check the first few rows to see the structure
print(df.head())

# Check unique values in the 'generated' column
print(df['generated'].unique())

# Count occurrences of each label (0 = Human, 1 = AI)
print(df['generated'].value_counts())
