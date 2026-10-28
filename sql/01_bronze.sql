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
