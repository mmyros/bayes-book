#!/usr/bin/env python
# coding: utf-8

# # Neurons example, no slopes
# 

# In[1]:


from importlib import reload

from bayes_window import visualization
from bayes_window import workflow, models, BayesWindow, BayesConditions, BayesRegression
from bayes_window.generative_models import generate_fake_spikes


# ## Make fake data

# In[2]:


df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=5,
                                                                n_neurons=3,
                                                                n_mice=4,
                                                                dur=7,
                                                               mouse_response_slope=16)


# ## Quick workflow

# In[3]:


bw=workflow.BayesWindow(df,y='isi',treatment='stim', condition='neuron', group='mouse',)
bw.plot(x='stim').facet('neuron')


# TODO this default is fishy

# In[4]:


bw.plot()


# In[5]:


bw.plot(x='stim', detail='neuron', color='mouse:N')


# Is this bc chart_p and chart_d traded places?

# In[6]:


bw=BayesConditions(df=df,y='isi',treatment='stim', condition='neuron', group='mouse')

bw.fit(model=models.model_single, );


# TODO data is fine, posterior is not
# 
# TODO Need saner facet call

# In[7]:


bw.plot(x='stim:O',color='neuron:N',independent_axes=False,add_data=True)#.display()
BayesWindow.facet(bw, column='mouse')


# In[8]:


df.neuron=df.neuron.astype(int)
bw=BayesConditions(df=df,y='isi',treatment='stim', condition='neuron', group='mouse')

bw.fit(model=models.model_single, )

bw.plot(x='stim:O',independent_axes=False,add_data=True);


# In[9]:


#builtin facet
bw.chart.properties(height=60).facet(column='neuron', row='mouse').display()


# In[10]:


# smart facet TODO

BayesWindow.facet(bw,column='neuron', row='mouse',height=60).display()


# In[11]:


#Full: add_data=True, independent_axes=True
bw = BayesConditions(df=df, y='isi', treatment='stim', condition='neuron', group='mouse')
bw.fit(model=models.model_single)
bw.plot(x='stim:O', independent_axes=True, add_data=True).display()
BayesWindow.facet(bw, column='neuron', row='mouse', width=90,height=120).display()


# ## Detailed steps
# ### 1. Estimate and make posterior plot

# ### 2. Make data slopeplot

# In[12]:


import altair as alt
fig_trials=visualization.plot_data_slope_trials(
                                                x='stim:O', y='log_firing_rate',
                                                color='neuron:N',detail='i_trial',
                                                base_chart=alt.Chart(df))


#chart = visualization.AltairHack(fig_trials)
#import types
#fig_trials.facetz = types.MethodType(facet, fig_trials )
#fig_trials.facetz(row='mouse')

#facet=visualization.facet
facet=fig_trials.facet
facet(column='mouse_code')


# In[13]:


visualization.facet(fig_trials,column='mouse_code')


# TODO neuron dont work here anymore, only neuron_code

# In[14]:


fig_trials.properties(width=50,height=50).facet(row='mouse',column='neuron')


# In[15]:


# Resolve scale doesnt work with facets yet:
#https://github.com/vega/vega-lite/issues/4373#issuecomment-447726094
# alt.layer(cposter, fig_trials, data=df_both).resolve_scale(y='independent').facet(row='mouse', column='neuron')

