-- Layered SQL (DuckDB): bronze = raw, silver = clean, gold = analysis-ready.
-- analysis.py runs each `-- @block` in turn; {placeholders} are filled in by Python.

-- @block bronze
-- Bronze: raw transactions exactly as supplied, columns renamed only.
CREATE OR REPLACE TABLE bronze_transactions AS
SELECT CAST(Invoice AS VARCHAR)      AS invoice,
       CAST(StockCode AS VARCHAR)    AS stock_code,
       Description                   AS description,
       CAST(Quantity AS INTEGER)     AS quantity,
       CAST(InvoiceDate AS TIMESTAMP) AS invoice_ts,
       CAST(Price AS DOUBLE)         AS price,
       CAST("Customer ID" AS BIGINT) AS customer_id,
       Country                       AS country
FROM read_csv_auto('{source}', header = true, all_varchar = true);

-- @block silver
-- Silver: one clean sales line per row. Every exclusion is counted in silver_dq_log.
CREATE OR REPLACE TABLE silver_flagged AS
SELECT *,
       CASE
         WHEN invoice LIKE 'C%'                              THEN 'cancellation'
         WHEN customer_id IS NULL                            THEN 'no customer id'
         WHEN EXISTS (SELECT 1 FROM bronze_transactions c
                      WHERE c.invoice LIKE 'C%' AND c.customer_id = t.customer_id AND c.stock_code = t.stock_code
                        AND c.quantity = -t.quantity AND c.invoice_ts >= t.invoice_ts)
                                                             THEN 'order line cancelled later'
         WHEN quantity <= 0                                  THEN 'non-positive quantity'
         WHEN price <= 0                                     THEN 'non-positive price'
         WHEN NOT regexp_matches(stock_code, '^[0-9]{5}')    THEN 'non-product code (postage, fees, adjustments)'
         ELSE NULL
       END AS exclude_reason
FROM bronze_transactions t;

CREATE OR REPLACE TABLE silver_dq_log AS
SELECT coalesce(exclude_reason, 'kept') AS reason, count(*) AS row_count
FROM silver_flagged GROUP BY 1;

CREATE OR REPLACE TABLE silver_sales AS
SELECT DISTINCT invoice, stock_code, description, quantity, invoice_ts, price, customer_id, country,
       quantity * price AS revenue
FROM silver_flagged
WHERE exclude_reason IS NULL;

INSERT INTO silver_dq_log
SELECT 'exact duplicates removed', (SELECT count(*) FROM silver_flagged WHERE exclude_reason IS NULL) - (SELECT count(*) FROM silver_sales);

-- @block gold_customers
-- Gold: customer features at a snapshot date. {snapshot} = first day NOT included in the features.
CREATE OR REPLACE TABLE gold_customer_features AS
WITH hist AS (SELECT * FROM silver_sales WHERE invoice_ts < TIMESTAMP '{snapshot}'),
inv AS (
  SELECT customer_id, invoice, min(invoice_ts) AS ts, sum(revenue) AS basket
  FROM hist GROUP BY customer_id, invoice
)
SELECT i.customer_id,
       date_diff('day', max(i.ts), TIMESTAMP '{snapshot}')  AS recency_days,
       count(*)                                             AS frequency,
       sum(i.basket)                                        AS monetary,
       date_diff('day', min(i.ts), TIMESTAMP '{snapshot}')  AS tenure_days,
       avg(i.basket)                                        AS avg_basket,
       (SELECT count(DISTINCT stock_code) FROM hist h WHERE h.customer_id = i.customer_id) AS distinct_products,
       (SELECT mode(country) FROM hist h WHERE h.customer_id = i.customer_id)             AS country
FROM inv i GROUP BY i.customer_id;

-- @block gold_lapse_labels
-- Gold: lapse label. A customer active before {snapshot} has lapsed if they buy nothing in the next {horizon} days.
CREATE OR REPLACE TABLE gold_lapse_labels AS
SELECT f.customer_id,
       CASE WHEN EXISTS (
         SELECT 1 FROM silver_sales s
         WHERE s.customer_id = f.customer_id
           AND s.invoice_ts >= TIMESTAMP '{snapshot}'
           AND s.invoice_ts <  TIMESTAMP '{snapshot}' + INTERVAL {horizon} DAY
       ) THEN 0 ELSE 1 END AS lapsed
FROM gold_customer_features f;

-- @block gold_weekly_sales
-- Gold: units sold per product per week (weeks start Monday), for the forecasting mart.
CREATE OR REPLACE TABLE gold_weekly_product_sales AS
SELECT stock_code,
       CAST(date_trunc('week', invoice_ts) AS DATE) AS week_start,
       sum(quantity) AS units,
       sum(revenue)  AS revenue
FROM silver_sales
GROUP BY 1, 2;
