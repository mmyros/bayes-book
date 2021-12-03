#!/usr/bin/env python
# coding: utf-8

# # LFP example with stim strength

# In[1]:


import numpy as np
import pandas as pd
from bayes_window import BayesWindow, BayesRegression
from bayes_window.generative_models import generate_fake_lfp


# ## ISI

# In[2]:


df = []
for slope in np.linspace(4, 40, 4):
    df1 = generate_fake_lfp(mouse_response_slope=slope)[0]
    df1['stim_strength'] = slope
    df.append(df1)
df = pd.concat(df)

BayesWindow(df=df, y='isi', treatment='stim_strength', group='mouse', detail='i_trial').data_box_detail()


# In[3]:


window = BayesRegression(df=df, y='isi', treatment='stim_strength', group='mouse', detail='i_trial')
window.fit(add_group_slope=False, dist_y='gamma')
window.chart


# In[4]:


window.explore_model_kinds()


# In[5]:


## Power


# ## Power

# In[6]:


df = []
for slope in np.linspace(4, 400, 4):
    df1 = generate_fake_lfp(mouse_response_slope=slope)[0]
    df1['stim_strength'] = slope
    df.append(df1)
df = pd.concat(df)

BayesWindow(df=df, y='Power', treatment='stim_strength', group='mouse', detail='i_trial').data_box_detail()


# In[7]:


window = BayesRegression(df=df, y='Power', treatment='stim_strength', group='mouse', detail='i_trial')
window.fit(add_group_slope=False, dist_y='gamma')
window.chart


# In[8]:


window = BayesRegression(df=df, y='Power', treatment='stim_strength', condition='mouse', detail='i_trial')
window.fit(add_condition_slope=True, center_intercept=True, dist_y='gamma')
window.chart


# In[9]:


window.explore_model_kinds()


# In[10]:


window.explore_models()


# In[11]:


window.explore_models()


# In[12]:


window.explore_models()

