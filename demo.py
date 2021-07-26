import streamlit as st
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


data = load_digits()
np.random.seed(1450)
X = np.random.randn(1000,10)
y = np.random.randint(0,2, 1000,)


#X = data['data']
#y = data['target']


l1_ratio = st.slider(min_value = 0.0, max_value = 1.0, step=0.00001, label='L1 Ratio.  Set to 0 for L2, Set to 1 for L1')






#idx_show = np.where(y==1)[0][20]
#digit_show =data['images'][idx_show]
#plt.imshow(digit_show, cmap = plt.cm.gray_r)

reg = LogisticRegression( fit_intercept= False,l1_ratio = l1_ratio,C = 0.1, penalty='elasticnet',max_iter = 1000, solver='saga').fit(X,y)
coefs = reg.coef_[0,:]

# %%

fig2,ax2 = plt.subplots()
ax2.bar(x = [i for i in range(10)], height =sorted(abs(coefs)))
#ax2.imshow(coefs.reshape(1,10), cmap=plt.cm.gray_r)



st.pyplot(fig2)

