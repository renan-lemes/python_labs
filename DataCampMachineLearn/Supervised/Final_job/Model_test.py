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
