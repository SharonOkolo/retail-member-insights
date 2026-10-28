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
