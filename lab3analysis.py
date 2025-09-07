import pandas as pd

df = pd.read_csv(r"C:\Users\user\Desktop\final.csv")

print(df.info())
print("____")
num_agree = df[df['Classes_Agree'] == 'YES'].shape[0]
print("Number of lines with code similarity = YES:", num_agree)
print("____")
print(df['Semantic_class'].value_counts())
print(df['Token_class'].value_counts())
print(df['Classes_Agree'].value_counts())
print("____")
print(df.groupby('Semantic_class')[['MI_Change', 'CC_Change', 'LOC_Change']].mean())
print("____")
print(df.groupby('Token_class')[['MI_Change', 'CC_Change', 'LOC_Change']].mean())
print("____")
print(df[['Semantic_Similarity', 'Token_Similarity', 'MI_Change', 'CC_Change', 'LOC_Change']].corr())
import matplotlib.pyplot as plt
df['Semantic_Similarity'].hist(bins=30)
plt.title("Semantic Similarity Distribution")
plt.show()

df['Token_Similarity'].hist(bins=30)
plt.title("Token Similarity Distribution")
plt.show()