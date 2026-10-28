"""Run the layered SQL (bronze -> silver -> gold) in DuckDB. Databricks runs the same layers in notebooks/."""
import os
import duckdb

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SQL = os.path.join(ROOT, "sql")
LAPSE_SNAPSHOT = "2011-09-01"   # features before this date, label = no purchase in the following 90 days
FULL_SNAPSHOT = "2011-12-10"    # day after the last transaction: used for segmentation
HORIZON = 90


def run_sql(con, name, **params):
    sql = open(os.path.join(SQL, name)).read()
    for k, v in params.items():
        sql = sql.replace("{" + k + "}", str(v))
    con.execute(sql)


def build(source, db=":memory:"):
    con = duckdb.connect(db)
    run_sql(con, "01_bronze.sql", source=source)
    run_sql(con, "02_silver.sql")
    run_sql(con, "03_gold_customers.sql", snapshot=LAPSE_SNAPSHOT)
    run_sql(con, "04_gold_lapse_labels.sql", snapshot=LAPSE_SNAPSHOT, horizon=HORIZON)
    con.execute("CREATE OR REPLACE TABLE gold_lapse_features AS SELECT * FROM gold_customer_features")
    run_sql(con, "03_gold_customers.sql", snapshot=FULL_SNAPSHOT)
    run_sql(con, "05_gold_weekly_sales.sql")
    return con


def quality_checks(con):
    """Fail loudly if the clean layer breaks its contract."""
    q = lambda s: con.execute(s).fetchone()[0]
    assert q("SELECT count(*) FROM silver_sales") > 0, "silver is empty"
    assert q("SELECT count(*) FROM silver_sales WHERE customer_id IS NULL OR quantity <= 0 OR price <= 0") == 0
    assert q("SELECT count(*) FROM silver_sales WHERE invoice LIKE 'C%'") == 0
    assert q("SELECT count(*) - count(DISTINCT customer_id) FROM gold_customer_features") == 0, "duplicate customers"
    assert q("SELECT min(recency_days) FROM gold_customer_features") >= 0
    kept = q("SELECT row_count FROM silver_dq_log WHERE reason = 'kept'")
    dupes = q("SELECT row_count FROM silver_dq_log WHERE reason = 'exact duplicates removed'")
    assert kept - dupes == q("SELECT count(*) FROM silver_sales"), "row counts don't reconcile"
    return True


if __name__ == "__main__":
    con = build(os.path.join(ROOT, "data", "raw", "online_retail_ii.csv"), os.path.join(ROOT, "data", "retail.duckdb"))
    quality_checks(con)
    print(con.execute("SELECT * FROM silver_dq_log ORDER BY row_count DESC").df())
    for t in ["bronze_transactions", "silver_sales", "gold_customer_features", "gold_lapse_features", "gold_weekly_product_sales"]:
        print(t, con.execute(f"SELECT count(*) FROM {t}").fetchone()[0])
    print(con.execute("SELECT avg(lapsed), count(*) FROM gold_lapse_labels").fetchone())
