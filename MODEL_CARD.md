# Model card: customer lapse model

**Model:** scikit-learn GradientBoostingClassifier (default settings, random_state 42), chosen over logistic regression
and a depth-4 decision tree on ROC AUC.

**Intended use:** rank existing customers by the chance they will make no purchase in the next 90 days, so a CRM team
can prioritise win-back contact. A person reviews the list; the model should not trigger automatic decisions such as
withdrawing offers or changing prices.

**Not for:** credit, pricing or any decision about an individual's eligibility for something.

**Training data:** 5,214 customers with purchases before 1 Sep 2011 (UCI Online Retail II, CC BY 4.0). Label = no purchase
from 1 Sep to 29 Nov 2011. Base rate 57%. Features: recency, frequency, spend, tenure, average basket, distinct products,
UK flag. No personal or protected characteristics are available or used.

**Evaluation (25% stratified hold-out, 1,304 customers):** ROC AUC 0.82, precision 0.75, recall 0.83, F1 0.79 at threshold 0.5.

**Fairness and performance by group** (`outputs/fairness_lapse_model.csv`):
| Group | Lapse rate | Recall | False-positive rate | ROC AUC |
|---|---|---|---|---|
| UK | 58% | 0.83 | 0.37 | 0.81 |
| International | 51% | 0.84 | 0.29 | 0.83 |
| Lowest 25% spend | 80% | 0.97 | 0.86 | 0.71 |
| Top 25% spend | 26% | **0.38** | 0.08 | 0.80 |

**Known issue:** at one threshold, the model under-detects lapse among top spenders (their base rate is much lower) and
over-flags low spenders. **Mitigation:** use a threshold per spend band (0.3 for the top band gives recall 0.69 and a
false-positive rate of 0.26) and review the list by hand.

**Limitations:** two years of data; wholesale ordering cycles blur what "lapse" means; one random split (metrics move by
a few points with other seeds); no demographic data, so fairness can only be checked by market and spend.

**Monitoring if deployed:** monthly AUC and recall by spend band, lapse base rate drift, and feature drift for recency and frequency.
