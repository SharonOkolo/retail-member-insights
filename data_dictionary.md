# Data dictionary

## silver_sales (one clean sales line)
| Column | Type | Meaning |
|---|---|---|
| invoice | text | Invoice number (6 digits; cancellations start with C and are excluded) |
| stock_code | text | Product code (5 digits, optionally a letter suffix) |
| description | text | Product name |
| quantity | int | Units on the line (> 0) |
| invoice_ts | timestamp | Invoice date and time |
| price | float | Unit price, £ (> 0) |
| customer_id | int | Anonymous customer number (never null in silver) |
| country | text | Customer's country |
| revenue | float | quantity × price |

**Exclusion rules** (counted in `silver_dq_log` / `outputs/data_quality_log.csv`): cancellation invoices; no customer ID;
order lines matched to a later cancellation (same customer, product and opposite quantity); quantity ≤ 0; price ≤ 0;
non-product codes (postage, bank charges, manual adjustments); exact duplicate rows.

## gold_customer_features (one row per customer at a snapshot date)
| Column | Meaning |
|---|---|
| recency_days | Days from the last order to the snapshot |
| frequency | Number of distinct invoices |
| monetary | Total spend, £ |
| tenure_days | Days from the first order to the snapshot |
| avg_basket | Average invoice value, £ |
| distinct_products | Number of different products bought |
| country | Most common country on the customer's orders |

Snapshots: **2011-09-01** for training the lapse model (`gold_lapse_features`) and **2011-12-10** for segmentation and
current scoring (`gold_customer_features`).

## gold_lapse_labels
| Column | Meaning |
|---|---|
| lapsed | 1 if the customer made no purchase in the 90 days from 2011-09-01, else 0 |

## gold_weekly_product_sales
| Column | Meaning |
|---|---|
| stock_code, week_start | Product and week (weeks start on Monday) |
| units, revenue | Units sold and revenue that week |

## outputs/customer_scores.csv
Customer features plus `cluster`, `segment` and `lapse_prob` (gradient boosting, scored at 2011-12-10).
