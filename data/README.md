# Data

**Source:** Online Retail II, UCI Machine Learning Repository (Chen, D. 2012). https://archive.ics.uci.edu/dataset/502/online+retail+ii
**Licence:** Creative Commons Attribution 4.0 (CC BY 4.0).
**Content:** all transactions of a UK-based, non-store online retailer of gift-ware, 01/12/2009 to 09/12/2011.
1,067,371 rows, 8 columns. Prices are in pounds sterling. Customer IDs are anonymous numbers.

**Download:** `python src/ingest.py` (downloads the 45 MB zip and writes `data/raw/online_retail_ii.csv`).
Raw files are not committed (`.gitignore`); the aggregated outputs in `outputs/` are.
