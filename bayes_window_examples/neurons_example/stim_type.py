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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] slideshow={"slide_type": "slide"} hideCode=false hidePrompt=false
# # Neurons example with stim types
# ## Generate some data

# + slideshow={"slide_type": "skip"} hideCode=false hidePrompt=false
from bayes_window import models, fake_spikes_explore, BayesWindow
from bayes_window.generative_models import generate_fake_spikes,generate_spikes_stim_types
from importlib import reload
import numpy as np
import altair as alt
alt.data_transformers.disable_max_rows()
try:
    alt.renderers.enable('altair_saver', fmts=['png'])
except Exception:
    pass

# + slideshow={"slide_type": "skip"} hideCode=false hidePrompt=false
df = generate_spikes_stim_types(mouse_response_slope=3,
                                n_trials=2,
                                n_neurons=3,
                                n_mice=4,
                                dur=2, )


# + slideshow={"slide_type": "skip"} hideCode=false hidePrompt=false
from bayes_window import workflow
reload(workflow)
window = workflow.BayesWindow(df, y='isi', treatment='stim', condition=['stim_strength','neuron_x_mouse'], 
                              group='mouse')
window.fit_slopes(model=models.model_hierarchical)
    
# -


window.regression_charts(x='stim_strength',column='mouse',independent_axes=False,row='neuron_x_mouse')
# window.chart_posterior_hdi

window.chart
