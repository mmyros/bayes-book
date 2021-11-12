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



# + [markdown] hideCode=false hidePrompt=false
# # Neurons example via low-level, flexible interface
# ## Prepare

# + hideCode=false hidePrompt=false
from bayes_window import models
from bayes_window.fitting import fit_numpyro
from bayes_window.generative_models import generate_fake_spikes
import numpy as np
from sklearn.preprocessing import LabelEncoder

trans = LabelEncoder().fit_transform


# + [markdown] hideCode=false hidePrompt=false
# ## Make some data
#

# + hideCode=false hidePrompt=false
df, df_monster, index_cols, firing_rates = generate_fake_spikes(n_trials=2,
                                                                n_neurons=8,
                                                                n_mice=4,
                                                                dur=7, )


# df['log_isi'] = np.log10(df['isi'])

from bayes_window import visualization, utils
from importlib import reload

reload(visualization)
reload(utils)
y = 'isi'
ddf, dy = utils.make_fold_change(df,
                                 y=y,
                                 index_cols=('stim', 'mouse', 'neuron'),
                                 treatment_name='stim',
                                 do_take_mean=True)

visualization.plot_data(x='neuron', y=dy, color='mouse',  df=ddf)[0]
# -

# TODO leave axis labels here somehow

# + [markdown] hideCode=false hidePrompt=false
#
# ## Estimate model

# + hideCode=false hidePrompt=false
# y = list(set(df.columns) - set(index_cols))[0]
trace = fit_numpyro(y=df[y].values,
                    treatment=(df['stim']).astype(int).values,
                    condition=trans(df['neuron']),
                    group=trans(df['mouse']),
                    progress_bar=True,
                    model=models.model_hierarchical,
                    n_draws=100, num_chains=1, )

# + [markdown] hideCode=false hidePrompt=false
# ## Add data back 

# + hideCode=false hidePrompt=false
# reload(utils)
# df_both, trace = utils.add_data_to_posterior(df, posterior=trace.posterior, y=y,
#                                              fold_change_index_cols=['neuron', 'stim', 'mouse_code', ],
#                                              treatment_name='stim', b_name='slope_per_condition',
#                                              posterior_index_name='neuron', group_name='mouse')
#
# # + [markdown] hideCode=false hidePrompt=false
# # ## Plot data and posterior
#
# # + hideCode=false hidePrompt=false
# # BayesWindow.regression_charts(df_both, y=f'{y} diff', x='neuron',color='mouse_code',title=y,hold_for_facet=False,add_box=False)
# reload(visualization)
#
# chart_d, _ = visualization.plot_data(df=df_both, x='neuron', y=f'{y} diff', color='mouse_code', highlight=False)
# chart_d
#
# # + hideCode=false hidePrompt=false
# chart_p = visualization.plot_posterior(df=df_both, title=f'd_{y}', x='neuron', )
# chart_p
#
# # + hideCode=false hidePrompt=false
# (chart_d + chart_p).resolve_scale(y='independent')
#
# # + hideCode=false hidePrompt=false
# (chart_d + chart_p).facet(column='neuron')
# -

# ## Appendix: Elements of interactivity (WIP)

# +
import altair as alt
y='isi'
color='mouse'
x='neuron'

base=alt.Chart(df).encode(
            x=x,
            color=f'{color}',
            y=alt.Y(f'mean({y})',
                    scale=alt.Scale(zero=False,
                                    domain=list(np.quantile(df[y], [.05, .95])))),
)
highlight = alt.selection(type='single', on='mouseover',
                          fields=[color], nearest=True)


lines=base.mark_line(clip=True, fill=None, opacity=.6, ).encode(
            size=alt.condition(~highlight, alt.value(1), alt.value(3))
        )
points = base.mark_circle().encode(
    opacity=alt.value(0),
    #axis=alt.Axis(labels=False, tickCount=0, title='')
).add_selection(
    highlight
)

lines+points

# +
import altair as alt
y='isi'
color='mouse'
x='neuron'

base=alt.Chart(df).encode(
            x=x,
            color=f'{color}',
            y=alt.Y(f'mean({y})',
                    scale=alt.Scale(zero=False,
                                    domain=list(np.quantile(df[y], [.05, .95])))),
)

# Create a selection that chooses the nearest point & selects based on x-value
nearest = alt.selection(type='single', nearest=True, on='mouseover',
                        fields=[x], empty='none')


# Transparent selectors across the chart. This is what tells us
# the x-value of the cursor
selectors = alt.Chart(df).mark_point().encode(
    x=x,
    opacity=alt.value(0),
).add_selection(
    nearest
)

highlight = alt.selection(type='single', on='mouseover',
                          fields=[color], nearest=True)

lines=base.mark_line(clip=True, fill=None, opacity=.6, ).encode(
            #tooltip=color,
            size=alt.condition(~highlight, alt.value(1), alt.value(3))
        )

# Draw text labels near the points, and highlight based on selection
text = lines.mark_text(align='left', dx=5, dy=-5).encode(
    text=alt.condition(nearest, y, alt.value(' '))
)

points = base.mark_circle().encode(
    opacity=alt.value(0)
).add_selection(
    highlight
)

alt.layer(
    lines, selectors, points, text
)