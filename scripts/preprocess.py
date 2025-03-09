import pandas as pd
from datetime import datetime

# import data
def parse(x):
    # parse only the relevant features

df = pd.read_csv("diabetic_data.csv", parser = parse)

# handle duplicates and empty rows
df.replace("?", pd.NA, inplace=True)
df.dropna(inplace=True)
# standardize format
# select data columns
# remove outliers

# summarize first 5 rows
print(df.head(5))

# save cleaned data
df.to_csv('cleaned_data.csv')
