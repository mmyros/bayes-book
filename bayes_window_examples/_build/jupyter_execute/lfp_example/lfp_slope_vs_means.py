#!/usr/bin/env python
# coding: utf-8

# # LFP example: regression vs estimation of difference
# ## Make and visualize model oscillation power
# 40 trials of "theta power" is generated for every animal. It is drawn randomly as a poisson process.
# 
# This is repeated for "stimulation" trials, but poisson rate is higher.

# In[1]:


import altair as alt
from bayes_window import BayesWindow, models, BayesRegression, LMERegression, BayesConditions
from bayes_window.generative_models import generate_fake_lfp

try:
    alt.renderers.enable('altair_saver', fmts=['png'])
except Exception:
    pass


# In[ ]:





# In[2]:


# Draw some fake data:
df, df_monster, index_cols, _ = generate_fake_lfp(mouse_response_slope=15, n_trials=30)


# ## Estimate posteriors for treatments

# In[3]:


# Initialize:
window = BayesWindow(df=df, y='Power', #condition='stim',
                     group='mouse', 
                     treatment='stim')
# Fit:
estimation = BayesConditions(window).fit(dist_y='gamma')


# In[4]:


estimation.plot(color=':O')#.facet(column='mouse')


# In[5]:


estimation.plot_BEST()


# ## Compare with regression approach

# In[6]:


# Initialize:
# window = BayesWindow(df=df, y='Power', treatment='stim', group='mouse')
# Fit:
regression = BayesRegression(window).fit(model=models.model_hierarchical, add_group_intercept=True,
           add_group_slope=False, robust_slopes=False,
           do_make_change='subtract', dist_y='gamma')
(regression.chart + regression.chart_posterior_kde).properties(title='Regression')


# In[7]:


regression.plot_BEST()


# ## Evaluate sensitivity: CM

# In[8]:



import numpy as np

from bayes_window import model_comparison, BayesWindow
from bayes_window.generative_models import generate_fake_lfp


# ### y=Power

# In[9]:


# NBVAL_SKIP
# Note: Only works with single ys and single true_slopes
res = model_comparison.run_conditions(true_slopes=np.hstack([np.zeros(15),
                                                             np.tile(10, 15)]),
                                      n_trials=np.linspace(10, 70, 3).astype(int),
                                      ys=('Power',),
                                      methods=('bc_gamma','bw_gamma',),
                                      parallel=True)


# In[10]:


# NBVAL_SKIP
model_comparison.plot_confusion(
    model_comparison.make_confusion_matrix(res[res['y'] == 'Power'], ('method', 'y', 'randomness', 'n_trials')
                                           )).properties(width=140).facet(row='method', column='n_trials')


# In[11]:


df = model_comparison.make_roc_auc(res, binary=False, groups=('method', 'y', 'n_trials'))

bars, roc = model_comparison.plot_roc(df)
bars.facet(column='n_trials', row='y').properties().display()
roc.facet(column='n_trials', row='y').properties()
# NBVAL_SKIP


# ### y=Log power

# In[12]:


# NBVAL_SKIP
# Note: Only works with single ys and single true_slopes
res = model_comparison.run_conditions(true_slopes=np.hstack([np.zeros(15),
                                                             np.tile(10, 15)]),
                                      n_trials=np.linspace(10, 70, 3).astype(int),
                                      ys=('Log power',),
                                      methods=('bc_normal','bc_gamma','bc_student','bw_gamma',),
                                      parallel=True)


# In[13]:


# NBVAL_SKIP
model_comparison.plot_confusion(
    model_comparison.make_confusion_matrix(res[res['y'] == 'Log power'], ('method', 'y', 'randomness', 'n_trials')
                                           )).properties(width=140).facet(row='method', column='n_trials')


# In[14]:


df = model_comparison.make_roc_auc(res, binary=False, groups=('method', 'y', 'n_trials'))

bars, roc = model_comparison.plot_roc(df)
bars.facet(column='n_trials', row='y').properties().display()
roc.facet(column='n_trials', row='y').properties()
# NBVAL_SKIP

