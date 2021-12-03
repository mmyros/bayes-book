# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.3
#   kernelspec:
#     display_name: PyCharm (jup)
#     language: python
#     name: pycharm-d5912792
# ---

# + [markdown] slideshow={"slide_type": "slide"}
# # Compare to ANOVA and LMM: ROC curve

# +

import numpy as np

from bayes_window import model_comparison, BayesWindow
from bayes_window.generative_models import generate_fake_lfp

# -

# ## Data plot
# - Intercept varies across mice, but not slope
#     - (Same response, different baseline)
# - We vary the number of trials and then test raw or log-transformed data

# +
df, df_monster, index_cols, firing_rates = generate_fake_lfp(n_trials=70, mouse_response_slope=10)

BayesWindow(df, y='Power', treatment='stim', group='mouse', detail='i_trial').chart_data_box_detail.display()
BayesWindow(df, y='Log power', treatment='stim', group='mouse', detail='i_trial').chart_data_box_detail.display()


# + [markdown] slideshow={"slide_type": "slide"}
# ## ROC and CM, original scale

# + slideshow={"slide_type": "slide"}
# NBVAL_SKIP
# Note: Only works with single ys and single true_slopes
res = model_comparison.run_conditions(true_slopes=np.hstack([np.zeros(15),
                                                             np.tile(10, 15)]),
                                      #                                                              np.tile(np.linspace(20, 40, 3), 15)]),
                                      n_trials=np.linspace(10, 70, 5).astype(int),
                                      #                                       trial_baseline_randomness=np.linspace(.2, 11, 3),
                                      ys=('Power',),
                                      parallel=True)
# -

# ### Confusion matrix

# NBVAL_SKIP
model_comparison.plot_confusion(
    model_comparison.make_confusion_matrix(res[res['y'] == 'Power'], ('method', 'y', 'randomness', 'n_trials')
                                           )).properties(width=140).facet(row='method', column='n_trials')

# + [markdown] slideshow={"slide_type": "slide"}
# ### ROC curve

# +
df = model_comparison.make_roc_auc(res, binary=False, groups=('method', 'y', 'n_trials'))

bars, roc = model_comparison.plot_roc(df)
bars.facet(column='n_trials', row='y').properties().display()
roc.facet(column='n_trials', row='y').properties()
# NBVAL_SKIP
# -

# ## Log-transformed

reslog = model_comparison.run_conditions(true_slopes=np.hstack([np.zeros(15),
                                                                np.tile(10, 15)]),
                                         #                                                              np.tile(np.linspace(20, 40, 3), 15)]),
                                         n_trials=np.linspace(10, 70, 5).astype(int),
                                         #                                       trial_baseline_randomness=np.linspace(.2, 11, 3),
                                         ys=('Log power',),
                                         parallel=True)
# NBVAL_SKIP

# ### Confusion matrix

model_comparison.plot_confusion(
    model_comparison.make_confusion_matrix(reslog, ('method', 'y', 'randomness', 'n_trials')
                                           )).properties(width=140).facet(row='method', column='n_trials')
# NBVAL_SKIP

# + [markdown] slideshow={"slide_type": "slide"}
# ### ROC curve

# +
df = model_comparison.make_roc_auc(reslog, binary=False, groups=('method', 'y', 'n_trials'))

bars, roc = model_comparison.plot_roc(df)
bars.facet(column='n_trials', row='y').properties().display()
roc.facet(column='n_trials', row='y').properties()
# NBVAL_SKIP

