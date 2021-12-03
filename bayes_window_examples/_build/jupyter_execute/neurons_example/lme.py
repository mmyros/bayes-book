#!/usr/bin/env python
# coding: utf-8

# # Linear mixed effects

# In[1]:


from bayes_window import models, BayesWindow, LMERegression
from bayes_window.generative_models import generate_fake_spikes, generate_fake_lfp
import numpy as np


# In[2]:


df, df_monster, index_cols, _ = generate_fake_lfp(mouse_response_slope=8,
                                                 n_trials=40)


# ## LFP 
# Without data overlay

# In[3]:


bw = LMERegression(df=df, y='Log power', treatment='stim', group='mouse')
bw.fit(add_data=False)
bw.plot().display()


# In[4]:


bw.data_and_posterior


# ## With data overlay

# In[5]:


bw = LMERegression(df=df, y='Log power', treatment='stim', group='mouse')
try:
    bw.fit(add_data=True, do_make_change='subtract');
    bw.plot()    
except NotImplementedError:
    print('\n Data addition to LME is not implemented')


# ## Spikes

# In[6]:


df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=20,
                                                                n_neurons=6,
                                                                n_mice=3,
                                                                dur=5,
                                                               mouse_response_slope=40,
                                                               overall_stim_response_strength=5)
df['log_isi']=np.log10(df['isi'])


# In[7]:


bw = LMERegression(df=df, y='log_isi', treatment='stim', condition=['neuron_x_mouse'], group='mouse',)
bw.fit(add_data=False,add_group_intercept=True, add_group_slope=False);


# In[8]:


bw.chart


# ### Group slope

# In[9]:


bw = LMERegression(df=df, y='log_isi', treatment='stim', condition=['neuron_x_mouse'], group='mouse',)
bw.fit(add_data=False,add_group_intercept=True, add_group_slope=True)


# In[10]:


bw.chart


# In[11]:


bw.plot(x='neuron_x_mouse:O').display()


# ### Categorical 

# In[12]:


bw.fit(formula='log_isi ~ (1|mouse) + C(stim| neuron_x_mouse)')


# In[13]:


bw.plot(x='neuron_x_mouse:O').display()


# ### Nested

# In[14]:


bw = LMERegression(df=df, y='log_isi', treatment='stim', condition=['neuron_x_mouse'], group='mouse',)
try:
    bw.fit(add_data=False,add_group_intercept=True, add_group_slope=True, add_nested_group=True)
except Exception as e:
    print(e)

