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
