#!/usr/bin/env python
# coding: utf-8

# # LFP example

# In[1]:


import altair as alt
from bayes_window import BayesWindow, models, BayesRegression, LMERegression
from bayes_window.generative_models import generate_fake_lfp

try:
    alt.renderers.enable('altair_saver', fmts=['png'])
except Exception:
    pass


# ## Make and visualize model oscillation power
# 40 trials of "theta power" is generated for every animal. It is drawn randomly as a poisson process.
# 
# This is repeated for "stimulation" trials, but poisson rate is higher.

# In[2]:


# Draw some fake data:
df, df_monster, index_cols, _ = generate_fake_lfp(mouse_response_slope=15, n_trials=30)


# Mice vary in their baseline power.
# 
# Higher-baseline mice tend to have smaller stim response:

# In[3]:


BayesWindow(df=df, y='Log power', treatment='stim', group='mouse').plot(x='mouse').facet(column='stim')


# In[4]:


BayesWindow(df=df, y='Log power', treatment='stim', group='mouse', detail='i_trial').data_box_detail().facet(
    column='mouse')


# ## Fit a Bayesian hierarchical model and plot slopes
# In a hierarchical model, parameters are viewed as a sample from a population distribution of parameters. Thus, we view them as being neither entirely different or exactly the same. This is ***partial pooling***:
# 
# ![hierarchical](../motivation/parpooled.png)
# This model allows intercepts to vary across mouse, according to a random effect. We just add a fixed slope for the predictor (i.e all mice will have the same slope):
# 
# $$y_i = \alpha_{j[i]} + \beta x_{i} + \epsilon_i$$
# 
# where:
# - $j$ is mouse index
# - $i$ is observation index
# - $y_i$ is observed power
# - $x_i$ is 0 (no stimulation) or 1 (stimulation)
# - $\epsilon_i \sim N(0, \sigma_y^2)$, error
# - $\alpha_{j[i]} \sim N(\mu_{\alpha}, \sigma_{\alpha}^2)$, Random intercept
# 
# We set a separate intercept for each mouse, but rather than fitting separate regression models for each mouse, multilevel modeling **shares strength** among mice, allowing for more reasonable inference in mice with little data.

# The wrappers in this library allow us to fit and plot this inference in just three lines of code. Under the hood, it uses the following Numpyro code:
# ```python
# # Given: y, treatment, group, n_subjects
# # Sample intercepts
# a = sample('a', Normal(0, 1))
# a_subject = sample('a_subject', Normal(jnp.tile(0, n_subjects), 1))
# 
# # Sample variances
# sigma_a_subject = sample('sigma_a', HalfNormal(1))
# sigma_obs = sample('sigma_obs', HalfNormal(1))
# 
# # Sample slope - this is what we are interested in!
# b = sample('b_stim', Normal(0, 1))
# 
# # Regression equation
# theta = a + a_subject[group] * sigma_a_subject + b * treatment
# 
# # Sample power
# sample('y', Normal(theta, sigma_obs), obs=y)
# ```
# 
# 
# Above is the contents of `model_hier_stim_one_codition.py`, the function passed as argument in line 4 below.

# In[5]:


# Initialize:
window = BayesRegression(df=df, y='Power', treatment='stim', group='mouse')
# Fit:
window.fit(model=models.model_hierarchical, add_group_intercept=True,
           add_group_slope=False, robust_slopes=False,
           do_make_change='subtract', dist_y='gamma')

chart_power_difference = (window.chart + window.chart_posterior_kde).properties(title='Posterior')


# In[6]:


chart_power_difference


# In this chart:
# 
# - The black line is the 94% posterior highest density interval
# 
# - Shading is posterior density
# 
# - Barplot comes directly from the data

# In[7]:


# TODO diff_y is missing from data_and posterior
# chart_power_difference_box
window.data_and_posterior.rename({'Power': 'Power diff'}, axis=1, inplace=True)
# window.plot(x=':O',independent_axes=True).properties(title='Posterior')
window.chart


# In this chart:
# 
# - The blue dot is the mean of posterior
# 
# - The black line is the 94% highest density interval
# 
# - The boxplot is made from difference between groups in the data (no fitting)
# 
# - Left Y scale is for posterior, right for data

# ## Compare to non-bayesian approaches
# ### Off-the-shelf OLS ANOVA

# ANOVA does not pick up the effect of stim as significant:

# In[8]:


window = LMERegression(df=df, y='Log power', treatment='stim', group='mouse')
window.fit_anova();


# In[9]:


window = LMERegression(df=df, y='Log power', treatment='stim')
window.fit_anova();


# In[10]:


window = LMERegression(df=df, y='Power', treatment='stim', group='mouse')
window.fit_anova();


# In[11]:


window = LMERegression(df=df, y='Power', treatment='stim')
window.fit_anova();


# Including mouse as predictor helps, and we get no interaction:

# In[12]:


window.fit_anova(formula='Log_power ~ stim + mouse + mouse*stim');


# #### OLS ANOVA with heteroscedasticity correction

# In[13]:


window.fit_anova(formula='Power ~ stim + mouse ', robust="hc3");


# In[14]:


window.fit_anova(formula='Log_power ~ stim +mouse', robust="hc3");


# A linear mixed-effect model shows the effect of stim (slope) as significant. It includes intercepts of mouse, which also vary significantly:

# In[15]:


# Initialize:
window = LMERegression(df=df, y='Log power', treatment='stim', group='mouse')
window.fit(add_data=False);


# In[16]:


chart_power_difference_lme = window.plot().properties(title='LME')
chart_power_difference_lme


# ## Compare LME and Bayesian slopes side by side

# In[17]:


chart_power_difference | chart_power_difference_lme


# ## Inspect Bayesian result further
# Let's take a look at the intercepts and compare them to levels of power in the original data:

# In[18]:


# Initialize:
window = BayesRegression(df=df, y='Power', treatment='stim', group='mouse', detail='i_trial')
# Fit:
window.fit(model=models.model_hierarchical, add_group_intercept=True,
           add_group_slope=False, robust_slopes=False,
           do_make_change='subtract', dist_y='gamma');

chart_detail_and_intercepts = window.plot_intercepts(x='mouse')
window.chart_posterior_intercept


# In[19]:


chart_detail_and_intercepts


# Our plotting backend's flexibility allows us to easily concatenate multiple charts in the same figures with the | operator:

# In[20]:


window.chart_posterior_intercept | chart_power_difference | chart_power_difference_lme


# ## Check for false-positives with null model
# They sometimes appear with non-transformed data + "normal" model

# In[21]:


# Initialize:
df_null, df_monster_null, _, _ = generate_fake_lfp(mouse_response_slope=0, n_trials=30)
window = BayesRegression(df=df_null, y='Power', treatment='stim', group='mouse')
# Fit:
window.fit(model=models.model_hierarchical, add_group_intercept=True,
           add_group_slope=False, robust_slopes=False,
           do_make_change='subtract', dist_y='normal')

# Plot:
chart_power_difference = window.plot(independent_axes=False,
                                     ).properties(title='Posterior')

chart_power_difference


# This does not happen if we estimate group slopes.
# 
# GLM is more robust to no differences in the case of no effect:

# In[22]:


# Initialize:
window = BayesRegression(df=df_null, y='Power', treatment='stim', group='mouse')
# Fit:
window.fit(model=models.model_hierarchical, add_group_intercept=True,
           add_group_slope=False, robust_slopes=False,
           do_make_change='subtract', dist_y='gamma')
# Plot:
window.plot(independent_axes=False,
            ).properties(title='Posterior')


# ## Include all samples in each trial
# The mean of every one of the 30 trials we drew for each mouse is a manifestation of the same underlying process that generates power for each mouse. Let's try to include all samples that come in each trial

# In[23]:


# NBVAL_SKIP
# Initialize:
window = BayesRegression(df=df_monster, y='Power', treatment='stim', group='mouse')
# Fit:
window.fit(model=models.model_hierarchical, add_group_intercept=True,
           num_warmup=500, n_draws=160, progress_bar=True,
           add_group_slope=False, robust_slopes=False,
           do_make_change='subtract', dist_y='gamma');


# In[24]:


# NBVAL_SKIP
alt.data_transformers.disable_max_rows()
chart_power_difference_monster = window.plot(independent_axes=False).properties(title='Posterior')
chart_power_difference_monster


# Much tighter credible intervals here!
# 
# Same with linear mixed model:

# In[25]:


# NBVAL_SKIP
window = LMERegression(df=df_monster,
                         y='Log power', treatment='stim', group='mouse')
window.fit()

chart_power_difference_monster_lme = window.plot().properties(title='LME')
chart_power_difference_monster_lme


# In[26]:


# NBVAL_SKIP
(chart_power_difference_monster | chart_power_difference_monster_lme)

