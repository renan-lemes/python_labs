# %%

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, scale
#from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# %%
df = pd.read_csv('stomach_cancer_new_cases_per_100000_men.csv')


# %%
df

# %%
df.isnull().sum().head()

# %%
df.drop(['1952', '1953', '1954', '1955', '1956'], axis=1, inplace=True)
# %%
df.drop(['1957', '1958', '1959', '1960', '1961',
        '1962', '1963', '1964'], axis=1, inplace=True)
# %%
df
# %%
df.isnull().sum().head()
# %%
df.drop(['1965', '1966', '1967', '1968', '1969', '1970'], axis=1, inplace=True)
# %%
df

# %%
df.drop(['1978', '1977', '1976', '1975', '1974',
        '1973', '1972', '1971'], axis=1, inplace=True)
# %%
df
# %%
df.isnull().sum()
# %%
df.drop(['1979', '1980', '1981', '1982', '1983', '1984',
        '1985', '1986', '1987', '1988'], axis=1, inplace=True)
# %%
df
# %%
df.isnull().sum()

# %%
df.fillna(np.mean(df), inplace=True)
# %%
df
# %%
df.isnull().sum()
# %%
df.to_csv('data_clear.csv')
#----------------------------------------------------------------#
# %%
df['country']

# %%
fig = px.histogram(df, x='country', y='2015')
fig.show()

# %%
df.iloc[25]

# %%
df_brazil = df.iloc[25]
# %%
df_brazil
# %%
df_brazil = pd.DataFrame(df_brazil)
# %%
df_brazil
# %%
fig_br = px.histogram(df_brazil)
fig_br.show()

# %%
df_brazil

# %%
df_brazil.columns
# %%
df_brazil.T
# %%
# %%
df_brazil
# %%
df_brazil.reset_index(drop=True)
# %%

df['country'][1]
# %%

# %%
steps = [('scaler', StandardScaler()),
         ('LogisticRegression', LogisticRegression())]
# %%
pipeline = Pipeline(steps)
# %%
df['country'].values
# %%
X = df.drop(['country', '2015'], axis=1).values
y = df['2015'].values
# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42)
# %%
X.shape
# %%
y.shape

# %%
logreg_pipe = pipeline.fit(X_train, y_train)
logreg = LogisticRegression().fit(X_train, y_train)
# %%
df_origin = pd.get_dummies(df)
# %%
df_origin
# %%
y
# %%
