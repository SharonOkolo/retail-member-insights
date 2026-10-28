-- Gold: units sold per product per week (weeks start Monday), for the forecasting mart.
CREATE OR REPLACE TABLE gold_weekly_product_sales AS
SELECT stock_code,
       CAST(date_trunc('week', invoice_ts) AS DATE) AS week_start,
       sum(quantity) AS units,
       sum(revenue)  AS revenue
FROM silver_sales
GROUP BY 1, 2;
