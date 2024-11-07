import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn import datasets, preprocessing
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
import pandas as pd


df =pd.read_csv('1101.csv')
#print(df)

sns.boxplot(data=df)
plt.show()