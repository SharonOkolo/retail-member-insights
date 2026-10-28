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
