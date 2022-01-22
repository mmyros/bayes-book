# Intro

Regression examples illustrating Bayes window, a suite of classes for onle-liner estimation of posteriors and linear mixed models.

## Why bayes-window
- Compact envocation
  - Bayesian models up to hiearchical regression with just a single call
  - Faceted visualization included
- API free of Bayesian jargon
- Robust visualization with or without fitting a model 
- Baked-in overlay of model output onto exploratory plots (eg boxplot)
  - Easy to see when model fails to capture data
- Composable graphs (eg lme_ci_graph | bayes_hdi_with_boxplot)

## Why estimation statistics at all
- Visual representation of confidence interval
  - Allows to eyeball effect size
  - Easy to explain to non-scientists
- Hypothesis testing-free (Except LME)
- Non-standard distributions
  - Count data (eg action potentials of neurons)
  - Lognormal effects (common in many fields)

## Why not arviz?
- No faceting 
- No integration with model's intention
For a meaningful pub-ready presentation of posteriors over even one type of condition, arviz output is only usable after at least a page of code 
