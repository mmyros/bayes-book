#!/usr/bin/env python
# coding: utf-8

# # Neurons example, pt. 2: large datasets

# In[1]:


import numpyro
from bayes_window import models, fake_spikes_explore, BayesWindow, BayesConditions, BayesRegression
from bayes_window.generative_models import generate_fake_spikes
import numpy as np
from importlib import reload
import altair as alt
alt.data_transformers.disable_max_rows()

try:
    alt.renderers.enable('altair_saver', fmts=['png'])
except Exception:
    pass


# In[2]:



df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=20,
                                                                n_neurons=6,
                                                                n_mice=3,
                                                                dur=5,
                                                               mouse_response_slope=40,
                                                               overall_stim_response_strength=5)


# ## Step by step

# ### 1. Firing rate

# In[4]:


bw1 = BayesConditions(df=df_monster, y='isi',treatment='stim', condition=['neuron_x_mouse','i_trial'], group='mouse')
bw1.fit(dist_y='gamma');
# bw.plot_model_quality()


# ### 2. Regression

# In[5]:


bw2 = BayesRegression(df=bw1.posterior['mu_per_condition'],
                 y='center interval', treatment='stim', condition=['neuron_x_mouse'], group='mouse')
bw2.fit(model=models.model_hierarchical,
               do_make_change='subtract',
               dist_y='student',
               robust_slopes=False,
               add_group_intercept=False,
               add_group_slope=False,
               fold_change_index_cols=('stim', 'mouse', 'neuron_x_mouse'))

bw2.plot_model_quality()


# In[ ]:


bw2.chart


# ## Each neuron separately via SVI
# ### 1. Firing rate

# In[ ]:


from tqdm import tqdm

gb='neuron_x_mouse'
step1_res=[]
for i, df_m_n in tqdm(df_monster.groupby(gb)):
    bw1 = BayesConditions(df_m_n, y='isi',treatment='stim', condition=['i_trial'], group='mouse'
                     ).fit(dist_y='gamma',
#                                       fit_fn=fitting.fit_svi
                                     )
    posterior=bw1.posterior['mu_per_condition'].copy()
    posterior[gb] = i
    step1_res.append(posterior)


# ### 2. Regression
# TODO add sigma to step 2 inputs
# 

# In[ ]:


import pandas as pd
bw2 = BayesRegression(pd.concat(step1_res),
                 y='center interval', treatment='stim', condition=['neuron_x_mouse'], group='mouse',
                  detail='i_trial')
bw2.fit(model=models.model_hierarchical,
        do_make_change='subtract',
        dist_y='student',
        robust_slopes=False,
        add_group_intercept=False,
        add_group_slope=False)
bw2.chart


# ## NUTS 1-step GLM

# In[ ]:


# Gamma GLM
bw = BayesRegression(df_monster, y='isi', treatment='stim', condition=['neuron', 'mouse'], group='mouse')
bw.fit(model=models.model_hierarchical,
       progress_bar=True,
       do_make_change='subtract',
       dist_y='gamma',
       robust_slopes=False,
       add_group_intercept=False,
       add_group_slope=False,
       fold_change_index_cols=('stim', 'mouse', 'neuron','neuron_x_mouse'))

bw.plot_model_quality()


# In[ ]:


import altair as alt
alt.data_transformers.disable_max_rows()
bw.plot(x='neuron', color='mouse', independent_axes=True, finalize=True)


bw.facet(column='mouse',width=200,height=200).display()


# ## NUTS student

# In[ ]:


bw = BayesRegression(df=df_monster, y='isi', treatment='stim', condition=['neuron', 'mouse'], group='mouse')
bw.fit(model=models.model_hierarchical,
              progress_bar=True,
              do_make_change='subtract',
              dist_y='student',
              robust_slopes=False,
              add_group_intercept=False,
              add_group_slope=False,
              fold_change_index_cols=('stim', 'mouse', 'neuron','neuron_x_mouse'))

bw.plot_model_quality()


# In[ ]:



bw.plot(x='neuron', color='mouse', independent_axes=True, finalize=True)


bw.facet(column='mouse',width=200,height=200).display()


# In[ ]:



bw = BayesRegression(df_monster, y='isi', treatment='stim', condition='neuron_x_mouse', group='mouse')
bw.fit(model=models.model_hierarchical, do_make_change='subtract',
              progress_bar=True,
              dist_y='student',
              use_gpu=True,
              num_chains=1,
              num_warmup=500,
              add_group_slope=True, add_group_intercept=False,
              fold_change_index_cols=('stim', 'mouse', 'neuron','neuron_x_mouse'))

bw.plot(x='neuron', color='mouse', independent_axes=True, finalize=True)
bw.facet(column='mouse',width=200,height=200).display()

#bw.explore_models(use_gpu=True)


# ## NUTS Lognormal

# In[ ]:


bw = BayesRegression(df=df_monster, y='isi', treatment='stim', condition=['neuron', 'mouse'], group='mouse')
bw.fit(model=models.model_hierarchical, do_make_change='subtract',
              progress_bar=True,
              use_gpu=True, num_chains=1, n_draws=1500, num_warmup=1500,
              dist_y='lognormal',
              add_group_slope=True, add_group_intercept=True,
              fold_change_index_cols=('stim', 'mouse', 'neuron'))

bw.plot(x='neuron', color='mouse', independent_axes=True, finalize=True)


bw.facet(column='mouse',width=200,height=200).display()


# In[ ]:


bw.explore_models(add_group_slope=True)


# ## BarkerMH

# In[ ]:


get_ipython().run_cell_magic('time', '', "from bayes_window import fitting\n\nfrom importlib import reload\nreload(fitting)\n\nbw = BayesRegression(df=df_monster, y='isi', treatment='stim', condition=['neuron', 'mouse'], group='mouse')\nbw.fit(model=models.model_hierarchical, do_make_change='subtract',\n              sampler=numpyro.infer.BarkerMH,\n#               progress_bar=True,\n              use_gpu=False, num_chains=1, n_draws=5000, num_warmup=3000,\n              dist_y='student',\n              add_group_slope=True, add_group_intercept=True,\n              fold_change_index_cols=('stim', 'mouse', 'neuron'),\n              fit_method=fitting.fit_numpyro,\n             )\n\nbw.plot(x='neuron', color='mouse', independent_axes=True, finalize=True)\n\n\nbw.facet(column='mouse',width=200,height=200).display()")


# ## Fit using SVI

# In[ ]:


get_ipython().run_cell_magic('time', '', "from bayes_window import fitting\nbw = BayesRegression(df=df_monster, y='isi', treatment='stim', condition=['neuron', 'mouse'], group='mouse')\nbw.fit(model=models.model_hierarchical, do_make_change='subtract',\n              n_draws=5000,\n              dist_y='gamma',\n              add_group_slope=True, add_group_intercept=False,\n              fold_change_index_cols=('stim', 'mouse', 'neuron'),\n              fit_method=fitting.fit_svi,\n             )\n\nbw.plot(x='neuron', color='mouse', independent_axes=True, finalize=True)\n\n\nbw.facet(column='mouse',width=200,height=200).display()")


# In[ ]:


get_ipython().run_cell_magic('time', '', "from bayes_window import fitting\nimport numpyro\nfrom importlib import reload\nreload(fitting)\nbw = BayesRegression(df=df_monster, y='isi', treatment='stim', condition=['neuron', 'mouse'], group='mouse')\nbw.fit(model=models.model_hierarchical, do_make_change='subtract',\n\n            autoguide=numpyro.infer.autoguide.AutoLaplaceApproximation,\n            optim=numpyro.optim.Adam(step_size=0.0005),\n            loss=numpyro.infer.Trace_ELBO(),\n              dist_y='lognormal',\n              add_group_slope=True, add_group_intercept=True,\n              fold_change_index_cols=('stim', 'mouse', 'neuron'),\n              fit_method=fitting.fit_svi,\n\n              n_draws=int(1e5),\n              num_warmup=int(1e5),\n             )\n\nbw.plot(x='neuron', color='mouse', independent_axes=True, finalize=True)\n\n\nbw.facet(column='mouse',width=200,height=200).display()")


# In[ ]:


get_ipython().run_cell_magic('time', '', "from bayes_window import fitting\nimport numpyro\n#numpyro.enable_validation(False)\nbw = BayesRegression(df=df_monster, y='isi', treatment='stim', condition=['neuron', 'mouse'], group='mouse')\nbw.fit(model=models.model_hierarchical, do_make_change='subtract',\n              #use_gpu=True,\n            autoguide=numpyro.infer.autoguide.AutoLaplaceApproximation,\n            optim=numpyro.optim.Adam(1),\n            loss=numpyro.infer.Trace_ELBO(),\n              dist_y='lognormal',\n              add_group_slope=True, add_group_intercept=False,\n              fold_change_index_cols=('stim', 'mouse', 'neuron'),\n              fit_method=fitting.fit_svi,\n              #progress_bar=False,\n              n_draws=int(1e5),\n              num_warmup=int(1e5),\n             )\n\nbw.plot(x='neuron', color='mouse', independent_axes=True, finalize=True)\n\n\nbw.facet(column='mouse',width=200,height=200).display()")


# Pretty model

# In[ ]:


reload(models)
bw = BayesRegression(df=df, y='isi', treatment='stim', condition=['neuron', 'mouse'], group='mouse')
bw.fit(model=models.reparam_model(models.model_hierarchical_for_render), do_make_change='subtract',
              progress_bar=True,
              use_gpu=False, num_chains=1, n_draws=1500, num_warmup=1500,
              dist_y='normal',
              add_group_slope=True, add_group_intercept=True,
              fold_change_index_cols=('stim', 'mouse', 'neuron'))

bw.plot(x='neuron', color='mouse', independent_axes=True, finalize=True)


bw.facet(column='mouse',width=200,height=200).display()


# In[ ]:


#!pip install git+https://github.com/pyro-ppl/numpyro.git
from numpyro.contrib.render import render_model
reload(models)
render_model(models.model_hierarchical_for_render, model_args=(1, 1, 1, 1,
                                                    'gamma', True,
                                                    True),
             render_distributions=True)


# In[ ]:


#!pip install git+https://github.com/pyro-ppl/numpyro.git
from numpyro.contrib.render import render_model
reload(models)
render_model(models.model_hierarchical_for_render, model_args=(1, 1, 1, 1,
                                                    'gamma', True,
                                                    False),
             render_distributions=True)


# In[ ]:


#!pip install git+https://github.com/pyro-ppl/numpyro.git
from numpyro.contrib.render import render_model
reload(models)
render_model(models.model_hier_stim_one_codition, model_args=(1, 1, 1,
                                                    'gamma',
                                                    ),
             render_distributions=True)


# ## Two-step

# ## Packaged version 1
# Separate levels

# In[3]:


# bw = BayesRegression(df=df_monster, y='isi',treatment='stim', condition=['neuron_x_mouse'],
#                           group='mouse', detail='i_trial')
# bw=bw.fit_twostep_by_group(dist_y_step_one='gamma', dist_y='student')

# bw.chart


# ## Packaged version 2
# No grouping in first step

# In[3]:


# bw = BayesRegression(df=df_monster, y='isi',treatment='stim', condition=['neuron_x_mouse'], group='mouse', detail='i_trial')
# bw=bw.fit_twostep(dist_y_step_one='gamma', dist_y='student')


# In[ ]:


# bw.chart

