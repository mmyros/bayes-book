#!/usr/bin/env python
# coding: utf-8

# # Neurons example with stim types
# ## Generate some data

# In[1]:


from bayes_window import models, fake_spikes_explore, BayesWindow, BayesRegression
from bayes_window.generative_models import generate_fake_spikes,generate_spikes_stim_types
from importlib import reload
import numpy as np
import altair as alt


# In[2]:


df = generate_spikes_stim_types(mouse_response_slope=3,
                                n_trials=2,
                                n_neurons=3,
                                n_mice=4,
                                dur=2, )


# In[3]:



window = BayesRegression(df=df, y='isi', treatment='stim', condition=['stim_strength','neuron_x_mouse'],
                              group='mouse')
window.fit(model=models.model_hierarchical)


# In[4]:


window.plot(x='stim_strength',column='mouse',independent_axes=False,row='neuron_x_mouse')
# window.chart_posterior_hdi


# In[5]:


window.chart

