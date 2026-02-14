import pandas as pd
import numpy as np

df = pd.read_csv('temp_data.csv')

content_rating_counts = df['Content Rating'].value_counts().reset_index()
content_rating_counts.columns = ['Content Rating', 'Count']
print(content_rating_counts)