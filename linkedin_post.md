# LinkedIn post draft

Two customers looked like this retailer's biggest spenders. Both had cancelled almost everything they ordered.

I built an end-to-end retail analytics project on 1.07 million real transactions from a UK online retailer, to answer
three questions a membership or CRM team asks every week: who are our customers, who is about to stop buying, and how
much stock do we need?

🧹 The problem: 23% of rows had no customer ID, and about 30,000 were cancellations or orders later reversed. One
customer showed £168k of "spend" that had been fully cancelled.

⚙️ The process: layered SQL (bronze → silver → gold) that runs on Databricks and locally, a data-quality log that
fails the run if row counts don't reconcile, k-means segmentation, a lapse model and an 8-week demand forecast,
all rebuilt from raw data in about 6 seconds, with tests in CI.

📊 The results:
• 20% of customers (the "Champions") bring in 73% of revenue
• A gradient-boosting lapse model scores every customer before they leave (ROC AUC 0.82)
• A global forecasting model cut forecast error by 18% against "same as last week": about 15,800 fewer units planned wrongly in 8 weeks
• 408 high-value customers (£1.8m of spend) flagged for win-back

⚖️ The part I'm proudest of: a fairness check showed that at one threshold the model caught only 38% of lapsing
top spenders, the customers worth most. Using a threshold per spend band raised that to 69%. One overall accuracy
number would have hidden it.

🔗 Code: https://github.com/SharonOkolo/retail-member-insights
🔗 App: [Streamlit link]

#DataScience #Databricks #MachineLearning #ResponsibleAI #RetailAnalytics #Python #SQL

---
**Attach (in this order):** `screenshots/05_app_winback.png`, `screenshots/01_segments.png`,
`screenshots/03_fairness_recall.png`, `screenshots/02_forecast_error.png`.
