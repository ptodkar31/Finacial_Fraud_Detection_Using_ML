
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
 
df=pd.read_csv("C:/Users/Documents/Online_Fraud.csv")

df.head()

df.isnull().sum()

df.shape
#counts the occurrences of each unique value in the 'type' column.
df.type.value_counts()
 
type=df['type'].value_counts()
 
transactions=type.index
transactions
quantity=type.values
quantity 

import plotly.express as px
fig=px.pie(df,values=quantity,names=transactions,hole=0.4,title="Distribution of Transaction Type")
fig.show()  