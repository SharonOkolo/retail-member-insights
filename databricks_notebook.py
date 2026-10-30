# Databricks notebook source
# MAGIC %md
# MAGIC # Retail member insights: Databricks pipeline
# MAGIC Bronze → silver → gold Delta tables in Spark SQL, then segmentation, lapse prediction and forecasting tracked in MLflow.
# MAGIC
# MAGIC **Setup (Databricks Free Edition):** add this repo as a Git folder, create a volume
# MAGIC `workspace.default.retail`, and upload `data/online_retail_ii.csv` (created by `python3 analysis.py`) to it.

# COMMAND ----------

# MAGIC %pip install duckdb==2.0.0
# MAGIC %restart_python

# COMMAND ----------

SOURCE = "/Volumes/workspace/default/retail/online_retail_ii.csv"
spark.sql("USE workspace.default")
(spark.read.csv(SOURCE, header=True, inferSchema=False)
      .selectExpr("CAST(Invoice AS STRING) invoice", "CAST(StockCode AS STRING) stock_code", "Description description",
                  "CAST(Quantity AS INT) quantity", "CAST(InvoiceDate AS TIMESTAMP) invoice_ts", "CAST(Price AS DOUBLE) price",
                  "CAST(CAST(`Customer ID` AS DOUBLE) AS BIGINT) customer_id", "Country country")
      .write.mode("overwrite").saveAsTable("bronze_transactions"))

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Silver: same rules as the silver block in analysis.sql (Spark dialect)
# MAGIC CREATE OR REPLACE TABLE silver_flagged AS
# MAGIC SELECT *, CASE
# MAGIC     WHEN invoice LIKE 'C%' THEN 'cancellation'
# MAGIC     WHEN customer_id IS NULL THEN 'no customer id'
# MAGIC     WHEN EXISTS (SELECT 1 FROM bronze_transactions c WHERE c.invoice LIKE 'C%' AND c.customer_id = t.customer_id
# MAGIC                  AND c.stock_code = t.stock_code AND c.quantity = -t.quantity AND c.invoice_ts >= t.invoice_ts)
# MAGIC       THEN 'order line cancelled later'
# MAGIC     WHEN quantity <= 0 THEN 'non-positive quantity'
# MAGIC     WHEN price <= 0 THEN 'non-positive price'
# MAGIC     WHEN NOT stock_code RLIKE '^[0-9]{5}' THEN 'non-product code (postage, fees, adjustments)'
# MAGIC   END AS exclude_reason
# MAGIC FROM bronze_transactions t;
# MAGIC
# MAGIC CREATE OR REPLACE TABLE silver_sales AS
# MAGIC SELECT DISTINCT invoice, stock_code, description, quantity, invoice_ts, price, customer_id, country, quantity * price AS revenue
# MAGIC FROM silver_flagged WHERE exclude_reason IS NULL;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT coalesce(exclude_reason, 'kept') AS reason, count(*) AS row_count FROM silver_flagged GROUP BY 1 ORDER BY 2 DESC

# COMMAND ----------

def customer_features(snapshot):
    return spark.sql(f"""
      WITH hist AS (SELECT * FROM silver_sales WHERE invoice_ts < TIMESTAMP '{snapshot}'),
      inv AS (SELECT customer_id, invoice, min(invoice_ts) ts, sum(revenue) basket FROM hist GROUP BY 1, 2),
      extra AS (SELECT customer_id, count(DISTINCT stock_code) distinct_products, mode(country) country FROM hist GROUP BY 1)
      SELECT i.customer_id, datediff(TIMESTAMP '{snapshot}', max(ts)) recency_days, count(*) frequency,
             sum(basket) monetary, datediff(TIMESTAMP '{snapshot}', min(ts)) tenure_days, avg(basket) avg_basket,
             first(e.distinct_products) distinct_products, first(e.country) country
      FROM inv i JOIN extra e USING (customer_id) GROUP BY i.customer_id""")

customer_features("2011-12-10").write.mode("overwrite").saveAsTable("gold_customer_features")
customer_features("2011-09-01").write.mode("overwrite").saveAsTable("gold_lapse_features")
spark.sql("""
  CREATE OR REPLACE TABLE gold_lapse_labels AS
  SELECT f.customer_id, CASE WHEN s.customer_id IS NULL THEN 1 ELSE 0 END lapsed
  FROM gold_lapse_features f
  LEFT JOIN (SELECT DISTINCT customer_id FROM silver_sales
             WHERE invoice_ts >= TIMESTAMP '2011-09-01' AND invoice_ts < TIMESTAMP '2011-09-01' + INTERVAL 90 DAYS) s
  USING (customer_id)""")
spark.sql("""
  CREATE OR REPLACE TABLE gold_weekly_product_sales AS
  SELECT stock_code, CAST(date_trunc('WEEK', invoice_ts) AS DATE) week_start, sum(quantity) units, sum(revenue) revenue
  FROM silver_sales GROUP BY 1, 2""")

# COMMAND ----------

import os, sys
import mlflow, mlflow.sklearn
import numpy as np
sys.path.append(os.path.abspath("."))
from analysis import FEATURES, forecast, lapse_models, segment, weekly_matrix

cust = spark.table("gold_customer_features").toPandas()
train = spark.table("gold_lapse_features").join(spark.table("gold_lapse_labels"), "customer_id").toPandas()
for df in (cust, train):
    df["is_uk"] = (df.country == "United Kingdom").astype(int)

with mlflow.start_run(run_name="segmentation"):
    cust, prof, sil, k = segment(cust)
    mlflow.log_param("k", k)
    mlflow.log_metrics({f"silhouette_k{kk}": v for kk, v in sil.items()})

with mlflow.start_run(run_name="lapse_model"):
    model, best, results, test = lapse_models(train)
    mlflow.log_param("best_model", best)
    for name, m in results.items():
        mlflow.log_metrics({f"{name.replace(' ', '_')}_{k}": v for k, v in m.items()})
    mlflow.sklearn.log_model(model, "lapse_model", input_example=train[FEATURES].head(5))

with mlflow.start_run(run_name="demand_forecast"):
    wide = weekly_matrix(spark.table("gold_weekly_product_sales").toPandas())
    fc, actual, wape, abs_err, origin = forecast(wide)
    mlflow.log_metrics({f"wape_{n.split(' ')[0]}_{i}": v for i, (n, v) in enumerate(wape.items())})

display(prof)
