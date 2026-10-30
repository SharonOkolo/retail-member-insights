# Retail member insights

Customer segmentation, lapse prediction and an 8-week demand forecast on 1.07 million transactions from a UK online
retailer, with a fairness check on the model. Read **[Project_Report.docx](Project_Report.docx)** for the full
write-up: the problem, process, results, charts, recommendations, model card and data dictionary.

| File | What it is |
|---|---|
| `Project_Report.docx` | The project explained, with results and charts |
| `analysis.sql` | Layered SQL: raw → clean → analysis-ready tables, with every excluded row counted |
| `analysis.py` | Runs the SQL, then k-means segments, lapse model, demand forecast, fairness check and charts |
| `databricks_notebook.py` | The same pipeline as Spark SQL Delta tables on Databricks, with MLflow tracking |

**Headline results**
- 20% of customers (the "Champions") bring in 73% of £16.3m revenue.
- A gradient-boosting model predicts which customers will stop buying (ROC AUC 0.82) and flags 408 high-value customers for win-back.
- The demand forecast cuts error by 18% against "same as last week", and the fairness check found the model missed
  most lapsing top spenders at one threshold: a per-band threshold raised that from 38% to 69%.

**Run it** (Python 3 with duckdb, pandas, scikit-learn, matplotlib, openpyxl)
```bash
python3 analysis.py
```
