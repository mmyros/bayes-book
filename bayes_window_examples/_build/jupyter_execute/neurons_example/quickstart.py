#!/usr/bin/env python
# coding: utf-8

# # Neurons example, pt. 1
# ## Generate some data

# In[1]:


import altair as alt
import numpy as np
from bayes_window import models, fake_spikes_explore, BayesWindow, BayesRegression, LMERegression
from bayes_window.generative_models import generate_fake_spikes

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


# ## Exploratory plot without any fitting

# Three mice, five neurons each. Mouse #0/neuron #4 has the least effect, mouse #2/neuron #0 the most

# In[3]:



charts = fake_spikes_explore(df=df, df_monster=df_monster, index_cols=index_cols)
[chart.display() for chart in charts];
#fig_mice, fig_select, fig_neurons, fig_trials, fig_isi + fig_overlay, bar, box, fig_raster, bar_combined


# ## Estimate with neuron as condition

# ### ISI

# In[4]:


df['log_isi'] = np.log10(df['isi'])


# In[5]:


bw = BayesWindow(df_monster, y='isi', treatment='stim', condition=['neuron', 'mouse'], group='mouse')
bw.plot(x='neuron', color='stim', detail='i_trial', add_box=False).facet(column='mouse', )


# In[6]:


bw = BayesWindow(df=df, y='isi', treatment='stim', condition=['neuron', 'mouse'], group='mouse')
bw.plot(x='neuron', add_box=True).facet(row='mouse', column='stim')


# ## Vanilla regression

# In[7]:


bw = BayesRegression(df=df, y='isi', treatment='stim', condition=['neuron_x_mouse'], group='mouse', detail='i_trial')
bw.fit(model=(models.model_hierarchical),
       do_make_change='divide',
       dist_y='normal',
       )

bw.chart


# ## GLM
# ($y\sim Gamma(\theta)$)

# In[8]:


bw = BayesRegression(df=df, y='isi', treatment='stim', condition=['neuron', 'mouse'], group='mouse', detail='i_trial')
bw.fit(model=models.model_hierarchical,
       do_make_change='subtract',
       dist_y='gamma',
       add_group_intercept=True,
       add_group_slope=True,
       fold_change_index_cols=('stim', 'mouse', 'neuron', 'neuron_x_mouse', 'i_trial'))

bw.plot(x='neuron', color='mouse', independent_axes=True, finalize=True)

bw.facet(column='mouse', width=200, height=200).display()


# In[9]:


import altair as alt

slopes = bw.trace.posterior['slope_per_group'].mean(['chain', 'draw']).to_dataframe().reset_index()
chart_slopes = alt.Chart(slopes).mark_bar().encode(
    x=alt.X('mouse_:O', title='Mouse'),
    y=alt.Y('slope_per_group', title='Slope')
)
chart_slopes


# In[10]:


bw = LMERegression(df=df, y='firing_rate', treatment='stim', condition='neuron_x_mouse', group='mouse', )
#bw.fit_anova()
bw.fit()


# In[11]:


bw.plot(x='neuron_x_mouse:O')


# ### Firing rate

# In[12]:


bw = BayesRegression(df=df, y='firing_rate', treatment='stim', condition='neuron_x_mouse', group='mouse')
bw.fit(model=models.model_hierarchical, do_make_change='subtract',
       progress_bar=False,
       dist_y='student',
       add_group_slope=True, add_group_intercept=False,
       fold_change_index_cols=('stim', 'mouse', 'neuron', 'neuron_x_mouse'))

bw.plot(x='neuron', color='mouse', independent_axes=True, finalize=True)
bw.facet(column='mouse', width=200, height=200).display()


# ANOVA may not be appropriate here: It considers every neuron. If we look hard enough, surely we'll find a responsive neuron or two out of hundreds?

# In[13]:


bw = LMERegression(df=df, y='firing_rate', treatment='stim', condition='neuron_x_mouse', group='mouse')

bw.fit(formula='firing_rate ~ stim+ mouse + stim*mouse + neuron_x_mouse + stim * neuron_x_mouse');


# ## Model quality

# In[14]:


# Vanilla robust no interept or slope
bw = BayesRegression(df=df, y='isi', treatment='stim', condition=['neuron', 'mouse'], group='mouse')
bw.fit(model=(models.model_hierarchical),
       do_make_change='subtract',
       dist_y='student',
       robust_slopes=True,
       add_group_intercept=False,
       add_group_slope=False,
       fold_change_index_cols=('stim', 'mouse', 'neuron', 'neuron_x_mouse'))

bw.plot_model_quality()


# In[15]:


# Vanilla robust, intercept only
bw = BayesRegression(df=df, y='isi', treatment='stim', condition=['neuron', 'mouse'], group='mouse')
bw.fit(model=(models.model_hierarchical),
       do_make_change='subtract',
       dist_y='student',
       robust_slopes=True,
       add_group_intercept=True,
       add_group_slope=False,
       fold_change_index_cols=('stim', 'mouse', 'neuron', 'neuron_x_mouse'))

bw.plot_model_quality()


# In[16]:


# Vanilla robust, slopes only
bw = BayesRegression(df=df, y='isi', treatment='stim', condition=['neuron', 'mouse'], group='mouse')
bw.fit(model=(models.model_hierarchical),
       do_make_change='subtract',
       dist_y='student',
       robust_slopes=True,
       add_group_intercept=False,
       add_group_slope=True,
       fold_change_index_cols=('stim', 'mouse', 'neuron', 'neuron_x_mouse'))

bw.plot_model_quality()


# In[17]:


# Vanilla robust intercept and group
bw = BayesRegression(df=df, y='isi', treatment='stim', condition=['neuron', 'mouse'], group='mouse')
bw.fit(model=(models.model_hierarchical),
       do_make_change='subtract',
       dist_y='student',
       robust_slopes=True,
       add_group_intercept=True,
       add_group_slope=True,
       fold_change_index_cols=('stim', 'mouse', 'neuron', 'neuron_x_mouse'))

bw.plot_model_quality()


# In[18]:


# Gamma GLM intercept only
bw = BayesRegression(df=df, y='isi', treatment='stim', condition=['neuron', 'mouse'], group='mouse')
bw.fit(model=(models.model_hierarchical),
       do_make_change='subtract',
       dist_y='gamma',
       robust_slopes=False,
       add_group_intercept=True,
       add_group_slope=False,
       fold_change_index_cols=('stim', 'mouse', 'neuron', 'neuron_x_mouse'))

bw.plot_model_quality()


# group slopes+ group intercepts=>divergences

# ## LME fails

# In[19]:


bw = LMERegression(df=df, y='isi', treatment='stim', condition=['neuron_x_mouse'], group='mouse', )
bw.fit(add_data=False, add_group_intercept=True, add_group_slope=False)


# In[20]:


bw.chart.display()
#bw.facet(column='mouse').display()
"Proper faceting will work when data addition is implemented in fit_lme()"


# In[21]:


bw = LMERegression(df=df, y='isi', treatment='stim', condition=['neuron_x_mouse'], group='mouse', )
bw.fit(add_data=False, add_group_intercept=True, add_group_slope=True)


# In[22]:


bw.chart


# Need nested design, but get singular matrix:

# In[23]:


bw = LMERegression(df=df, y='isi', treatment='stim', condition=['neuron_x_mouse'], group='mouse', )
try:
    bw.fit(add_data=False, add_group_intercept=True, add_group_slope=True, add_nested_group=True)
except Exception as e:
    print(e)

