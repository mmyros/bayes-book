# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.8.2
#   kernelspec:
#     display_name: PyCharm (jup)
#     language: python
#     name: pycharm-d5912792
# ---

# # Neurons example, no slopes
#

# +
from importlib import reload

from bayes_window import visualization
from bayes_window import workflow, models
from bayes_window.generative_models import generate_fake_spikes
# -

# ## Make fake data

df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=5,
                                                                n_neurons=3,
                                                                n_mice=4,
                                                                dur=7,
                                                               mouse_response_slope=16)


# ## Quick workflow

bw=workflow.BayesWindow(df,y='isi',treatment='stim', condition='neuron', group='mouse',)
bw.plot(x='stim').facet('neuron')

# TODO this default is fishy

bw.plot_posteriors_no_slope()

# This is stil fishy: why vertical lines? all sources of variance should be accounted for

bw.plot_posteriors_no_slope(x='stim', detail='neuron', color='mouse:N')

# It's because of trial, look:

bw.plot_posteriors_no_slope(x='stim', color='neuron:N')
bw.facet(column='mouse')

bw.detail='i_trial'
bw.plot_posteriors_no_slope(x='stim', color='neuron:N')

#  And this default is thus fishy,too

bw.facet(column='mouse',height=80,width=80)

# TODO posterior facets dont work

# Is this bc chart_p and chart_d traded places?

# +
reload(workflow)
reload(visualization)
bw=workflow.BayesWindow(df,y='isi',treatment='stim', condition='neuron', group='mouse')

bw.fit_conditions(model=models.model_single, )
# -


# TODO data is fine, posterior is not

bw.plot_posteriors_no_slope(x='stim:O',color='neuron:N',independent_axes=False,add_data=True)#.display()
bw.facet(column='mouse')


# +
df.neuron=df.neuron.astype(int)
bw=workflow.BayesWindow(df,y='isi',treatment='stim', condition='neuron', group='mouse')

bw.fit_conditions(model=models.model_single, )

bw.plot_posteriors_no_slope(x='stim:O',independent_axes=False,add_data=True);
# -

#builtin facet
bw.chart.properties(height=60).facet(column='neuron', row='mouse').display()

# smart facet
bw.facet(column='neuron', row='mouse',height=60).display()

#Full: add_data=True, independent_axes=True
reload(workflow)
reload(visualization)
bw = workflow.BayesWindow(df, y='isi', treatment='stim', condition='neuron', group='mouse')
bw.fit_conditions(model=models.model_single)
bw.plot(x='stim:O', independent_axes=True, add_data=True).display()
bw.facet(column='neuron', row='mouse', width=90,height=120).display()

# ## Detailed steps
# ### 1. Estimate and make posterior plot

# ### 2. Make data slopeplot

# +
reload(visualization)
reload(workflow)
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
# -

visualization.facet(fig_trials,column='mouse_code')

# TODO neuron dont work here anymore, only neuron_code

reload(visualization)
reload(workflow)
fig_trials.properties(width=50,height=50).facet(row='mouse',column='neuron')


# +
# Resolve scale doesnt work with facets yet:
#https://github.com/vega/vega-lite/issues/4373#issuecomment-447726094
#alt.layer(cposter, fig_trials, data=df_both).resolve_scale(y='independent').facet(row='mouse', column='neuron')
