import pandas as pd
df = pd.read_csv('data.csv')
print(df['label'].value_counts())
