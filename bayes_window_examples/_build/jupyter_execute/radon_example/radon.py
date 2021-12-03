#!/usr/bin/env python
# coding: utf-8

# # Radon example from Gelman and Hill (2006)
# This is a reworking of the radon example from pymc3 https://docs.pymc.io/notebooks/multilevel_modeling.html
# 
# 
# Other implementations in: 
# 
# - tensorflow https://www.tensorflow.org/probability/examples/Multilevel_Modeling_Primer - pymc3 https://docs.pymc.io/notebooks/multilevel_modeling.html 
# 
# - stan https://mc-stan.org/users/documentation/case-studies/radon.html 
# 
# - pyro https://github.com/pyro-ppl/pyro-models/blob/master/pyro_models/arm/radon.py 
# 
# - numpyro fibrosis dataset http://num.pyro.ai/en/stable/tutorials/bayesian_hierarchical_linear_regression.html

# In[7]:


import altair as alt
try:
    alt.renderers.enable('altair_saver', fmts=['png'])
except Exception:
    pass
from bayes_window import BayesWindow, BayesConditions, LMERegression, BayesRegression
from utils import load_radon
df = load_radon()

df


# In[3]:


df.set_index(['county','floor']).hist(bins=100);


# In[4]:


window=BayesWindow(df.reset_index(), y='radon', treatment='floor',group='county')


# ## Plot data

# In[5]:


window.plot(x='county').facet(row='floor')


# ## Fit LME

# In[8]:


lme=LMERegression(window)#formula='radon ~ floor + ( 1 | county)')

window.plot()


# ## Fit Bayesian hierarchical with and without county-specific intercept

# In[10]:


window1=BayesRegression(df=df.reset_index(), y='radon', treatment='floor',group='county')
window1.fit(add_group_intercept=True);
window1.plot()


# In[11]:


window1.plot(x=':O')


# ### Inspect intercepts (horizontal ticks)

# In[12]:


window1.plot_intercepts()


# In[13]:


window2=BayesRegression(df=df.reset_index(), y='radon', treatment='floor',group='county')
window2.fit(add_group_intercept=False, add_group_slope=False, do_make_change='subtract');
window2.plot()


# In[14]:


(window.plot().properties(title='LME')|
 window1.plot().properties(title='Partially pooled Bayesian')|
 window2.plot().properties(title='Unpooled  Bayesian'))


# ## Compare the two models

# In[15]:


import arviz as az
datasets = {'unpooled' : window2.trace.posterior,
           'hierarchical': window1.trace.posterior} 

az.plot_forest(data=list(datasets.values()), model_names=list(datasets.keys()), 
               #backend='bokeh',
               #kind='ridgeplot',
               #ridgeplot_overlap=1.6,
               combined=True);


# For leave-one-out, let's remove any counties that did not contain both floors. This drops about 250 rows

# In[16]:


import pandas as pd
df_clean = pd.concat([ddf for i, ddf in df.groupby(['county']) 
                      if (ddf.floor.unique().size>1) 
                      and (ddf[ddf['floor']==0].shape[0]>1)
                      and (ddf[ddf['floor']==1].shape[0]>1)
                     ])


df_clean


# In[17]:


window1.data=df_clean
window1.explore_models()


# It looks like using including intercept actually hurts leave-one-out posterior predictive. Actually, so does including floor in the model. To bring this home, let's only use the models that did not have a warning above:

# In[ ]:


from bayes_window.model_comparison import compare_models

compare_models(df_clean,y=window1.y,parallel=True,
    models = {
                'full_normal': window1.model,
                'no_condition_or_treatment': window1.model,
                'no-treatment': window1.model,
                'no_group': window1.model,
            },
            extra_model_args = [
                {'treatment': window1.treatment, 'group': window1.group},
                {'treatment': None, 'condition': None},
                {'treatment': None, 'condition': window1.condition},
                {'treatment': window1.treatment, 'group': None},
            ])


# Keep in mind though that we had to remove some data that had too few labels in order to make LOO work. 

# ## References
# 
#   -  Gelman, A., & Hill, J. (2006), Data Analysis Using Regression and Multilevel/Hierarchical Models (1st ed.), Cambridge University Press.
#   -  Gelman, A. (2006), Multilevel (Hierarchical) modeling: what it can and cannot do, Technometrics, 48(3), 432–435.
#   -  McElreath, R. (2020), Statistical Rethinking - A Bayesian Course with Examples in R and Stan (2nd ed.), CRC Press.
# 
# 
