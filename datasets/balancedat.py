import pandas as pd
from sklearn.utils import resample

# Load dataset
file_path = "C:/Users/gelsa/Personality_project/datasets/Training_Essay_Data.csv"
df = pd.read_csv(file_path)

# Separate AI-generated (label=0) and Human-written (label=1) samples
df_ai = df[df['generated'] == 0]  # AI-Generated
df_human = df[df['generated'] == 1]  # Human-Written

# Downsample AI-Generated to match Human-Written count
df_ai_downsampled = resample(df_ai, replace=False, n_samples=len(df_human), random_state=42)

# Combine and shuffle
df_balanced = pd.concat([df_ai_downsampled, df_human]).sample(frac=1, random_state=42)

# Save the balanced dataset
balanced_file_path = "C:/Users/gelsa/Personality_project/datasets/Balanced_Training_Essay_Data.csv"
df_balanced.to_csv(balanced_file_path, index=False)

print(f"Balanced dataset saved at: {balanced_file_path}")
