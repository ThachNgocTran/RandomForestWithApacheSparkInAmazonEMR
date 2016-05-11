import pandas as pd

df = pd.read_csv('covtype.data', header=None)

print df[54].unique()
df[54] = df[54].apply(lambda x : x - 1)
print df[54].unique()

df.to_csv("abc.csv", header=False, index=False)
