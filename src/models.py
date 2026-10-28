"""Segmentation (k-means), lapse prediction (decision tree vs gradient boosting) and 8-week demand forecasting."""
import json, os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from pipeline import ROOT, build, quality_checks

OUT = os.path.join(ROOT, "outputs")
SEED = 42
FEATURES = ["recency_days", "frequency", "monetary", "tenure_days", "avg_basket", "distinct_products", "is_uk"]


# ---------- segmentation ----------
def rfm_matrix(df):
    return StandardScaler().fit_transform(np.log1p(df[["recency_days", "frequency", "monetary"]]))


def segment(customers, k=None):
    X = rfm_matrix(customers)
    scores = {}
    for kk in range(3, 8):
        labels = KMeans(kk, n_init=10, random_state=SEED).fit_predict(X)
        scores[kk] = float(silhouette_score(X, labels, sample_size=min(3000, len(X)), random_state=SEED))
    k = k or max(scores, key=scores.get)
    km = KMeans(k, n_init=10, random_state=SEED).fit(X)
    customers = customers.assign(cluster=km.labels_)
    prof = customers.groupby("cluster").agg(customers=("customer_id", "size"), recency_days=("recency_days", "median"),
                                            frequency=("frequency", "median"), monetary=("monetary", "median"),
                                            revenue=("monetary", "sum"))
    # Name clusters by rule so names stay right if k or the data change.
    prof["value_score"] = prof.frequency * prof.monetary / (prof.recency_days + 1)
    top = prof.value_score.idxmax()
    med_spend = customers.monetary.median()
    def name(c, r):
        if c == top:
            return "Champions"
        if r.recency_days < 90:
            return "Recent occasional"
        return "Slipping high-value" if r.monetary >= med_spend else "Lapsed one-offs"
    prof["segment"] = [name(c, r) for c, r in prof.iterrows()]
    prof["revenue_share"] = prof.revenue / prof.revenue.sum()
    customers["segment"] = customers.cluster.map(prof.segment)
    return customers, prof.sort_values("value_score", ascending=False), scores, k


# ---------- lapse model ----------
def lapse_models(train_df):
    X, y = train_df[FEATURES], train_df.lapsed
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    models = {
        "logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "decision tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, random_state=SEED),
        "gradient boosting": GradientBoostingClassifier(random_state=SEED),
    }
    results = {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        yhat = (p >= 0.5).astype(int)
        results[name] = {"roc_auc": roc_auc_score(yte, p), "precision": precision_score(yte, yhat),
                         "recall": recall_score(yte, yhat), "f1": f1_score(yte, yhat)}
    best = max(results, key=lambda n: results[n]["roc_auc"])
    test = Xte.assign(lapsed=yte, prob=models[best].predict_proba(Xte)[:, 1])
    return models[best], best, results, test


def fairness(test, groups):
    """Error rates by group for the chosen lapse model (threshold 0.5)."""
    rows = []
    test = test.assign(pred=(test.prob >= 0.5).astype(int))
    for gname, gcol in groups.items():
        for val, g in test.groupby(gcol, observed=True):
            pos, neg = g[g.lapsed == 1], g[g.lapsed == 0]
            rows.append({"dimension": gname, "group": str(val), "customers": len(g), "lapse_rate": g.lapsed.mean(),
                         "recall": pos.pred.mean() if len(pos) else np.nan,
                         "false_positive_rate": neg.pred.mean() if len(neg) else np.nan,
                         "roc_auc": roc_auc_score(g.lapsed, g.prob) if g.lapsed.nunique() == 2 else np.nan})
    return pd.DataFrame(rows)


# ---------- forecasting ----------
def weekly_matrix(weekly, top_n=20):
    weekly = weekly.assign(week_start=pd.to_datetime(weekly.week_start))
    last = weekly.week_start.max()
    weekly = weekly[weekly.week_start < last]                     # final week is partial (data ends on a Friday)
    top = weekly.groupby("stock_code").units.sum().nlargest(top_n).index
    wide = weekly[weekly.stock_code.isin(top)].pivot_table(index="week_start", columns="stock_code", values="units", aggfunc="sum")
    full = pd.date_range(wide.index.min(), wide.index.max(), freq="W-MON")
    return wide.reindex(full).fillna(0)


def origin_features(wide, origin, h):
    """Features known at forecast origin (the last observed week) for target week origin+h."""
    i = wide.index.get_loc(origin)
    t = i + h
    recent = wide.iloc[i - 3:i + 1].mean()
    ly_recent = wide.iloc[i - 55:i - 51].mean()
    return pd.DataFrame({"stock_code": wide.columns, "h": h, "lag52": wide.iloc[t - 52].values,
                         "recent4": recent.values, "ly_recent4": ly_recent.values,
                         "week_of_year": wide.index[t].isocalendar().week})


def forecast(wide, horizon=8):
    origin = wide.index[-horizon - 1]
    test_weeks = wide.index[-horizon:]
    actual = wide.loc[test_weeks]
    oi = wide.index.get_loc(origin)
    fc = {"naive (last week)": pd.DataFrame([wide.loc[origin]] * horizon, index=test_weeks),
          "seasonal naive (same week last year)": wide.iloc[oi + 1 - 52:oi + 1 - 52 + horizon].set_axis(test_weeks)}
    growth = (wide.iloc[oi - 7:oi + 1].sum() + 1) / (wide.iloc[oi - 59:oi - 51].sum() + 1)
    fc["seasonal naive x recent growth"] = fc["seasonal naive (same week last year)"] * growth.clip(0.25, 4)

    # Global gradient-boosting model trained on past origins (all targets fall before the test window).
    rows = []
    for o in wide.index[55:oi - horizon + 1]:
        for h in range(1, horizon + 1):
            f = origin_features(wide, o, h)
            f["y"] = wide.iloc[wide.index.get_loc(o) + h].values
            rows.append(f)
    train = pd.concat(rows)
    cols = ["h", "lag52", "recent4", "ly_recent4", "week_of_year"]
    gbr = GradientBoostingRegressor(random_state=SEED, n_estimators=300, max_depth=3, learning_rate=0.05)
    gbr.fit(np.log1p(train[cols].astype(float)), np.log1p(train.y))
    pred = []
    for h in range(1, horizon + 1):
        f = origin_features(wide, origin, h)
        pred.append(np.expm1(gbr.predict(np.log1p(f[cols].astype(float)))).clip(0))
    fc["gradient boosting (global model)"] = pd.DataFrame(pred, index=test_weeks, columns=wide.columns)

    wape = {n: float((f - actual).abs().values.sum() / actual.values.sum()) for n, f in fc.items()}
    abs_err = {n: float((f - actual).abs().values.sum()) for n, f in fc.items()}
    return fc, actual, wape, abs_err, origin


def main():
    con = build(os.path.join(ROOT, "data", "raw", "online_retail_ii.csv"))
    quality_checks(con)
    os.makedirs(OUT, exist_ok=True)
    dq = con.execute("SELECT * FROM silver_dq_log ORDER BY row_count DESC").df()
    dq.to_csv(os.path.join(OUT, "data_quality_log.csv"), index=False)

    cust = con.execute("SELECT * FROM gold_customer_features ORDER BY customer_id").df()  # fixed order = reproducible splits
    cust["is_uk"] = (cust.country == "United Kingdom").astype(int)
    cust, prof, sil, k = segment(cust)
    prof.to_csv(os.path.join(OUT, "segments_summary.csv"))

    train = con.execute("SELECT f.*, l.lapsed FROM gold_lapse_features f JOIN gold_lapse_labels l USING (customer_id) ORDER BY customer_id").df()
    train["is_uk"] = (train.country == "United Kingdom").astype(int)
    model, best, lapse_res, test = lapse_models(train)
    test["market"] = np.where(test.is_uk == 1, "UK", "International")
    test["spend_band"] = pd.qcut(test.monetary, 4, labels=["lowest 25%", "25-50%", "50-75%", "top 25%"])
    fair = fairness(test, {"market": "market", "spend band": "spend_band"})
    fair.to_csv(os.path.join(OUT, "fairness_lapse_model.csv"), index=False)
    top_band = test[(test.spend_band == "top 25%")]
    top_recall_03 = float((top_band[top_band.lapsed == 1].prob >= 0.3).mean())
    top_fpr_03 = float((top_band[top_band.lapsed == 0].prob >= 0.3).mean())

    # Score every current customer: who to contact first. Exclude customers already gone 180+ days.
    cust["lapse_prob"] = model.predict_proba(cust[FEATURES])[:, 1]
    cust.to_csv(os.path.join(OUT, "customer_scores.csv"), index=False)
    # Win-back list: top-quartile spenders the model flags at the lower 0.3 threshold recommended for that band.
    at_risk_value = cust[(cust.monetary >= cust.monetary.quantile(0.75)) & (cust.lapse_prob >= 0.3)]

    weekly = con.execute("SELECT * FROM gold_weekly_product_sales ORDER BY stock_code, week_start").df()
    desc = con.execute("SELECT stock_code, mode(description) AS description, median(price) AS price FROM silver_sales GROUP BY 1").df().set_index("stock_code")
    wide = weekly_matrix(weekly)
    fc, actual, wape, abs_err, origin = forecast(wide)
    long = actual.stack().rename("actual").to_frame()
    for n, f in fc.items():
        long[n] = f.stack()
    long = long.reset_index().rename(columns={"level_0": "week_start"})
    long["description"] = long.stock_code.map(desc.description)
    long.to_csv(os.path.join(OUT, "forecast_holdout.csv"), index=False)
    hist = wide.stack().rename("units").reset_index().rename(columns={"level_0": "week_start"})
    hist["description"] = hist.stock_code.map(desc.description)
    hist.to_csv(os.path.join(OUT, "weekly_history_top20.csv"), index=False)

    best_fc = min(wape, key=wape.get)
    units_saved = abs_err["naive (last week)"] - abs_err[best_fc]
    avg_price = float(desc.loc[wide.columns, "price"].mean())
    metrics = {
        "rows_bronze": int(con.execute("SELECT count(*) FROM bronze_transactions").fetchone()[0]),
        "rows_silver": int(con.execute("SELECT count(*) FROM silver_sales").fetchone()[0]),
        "customers": int(len(cust)), "silhouette_by_k": sil, "k": int(k),
        "segments": prof[["segment", "customers", "recency_days", "frequency", "monetary", "revenue_share"]].to_dict("records"),
        "lapse_train_customers": int(len(train)), "lapse_base_rate": float(train.lapsed.mean()),
        "lapse_models": lapse_res, "lapse_best": best,
        "top_band_recall_at_0_3": top_recall_03, "top_band_fpr_at_0_3": top_fpr_03,
        "high_value_at_risk_customers": int(len(at_risk_value)),
        "high_value_at_risk_revenue": float(at_risk_value.monetary.sum()),
        "forecast_origin": str(origin.date()), "forecast_wape": wape, "forecast_best": best_fc,
        "forecast_units_abs_error": abs_err, "units_error_avoided_vs_naive": units_saved,
        "avg_unit_price_top20": avg_price,
    }
    json.dump(metrics, open(os.path.join(OUT, "metrics.json"), "w"), indent=2, default=float)
    return metrics


if __name__ == "__main__":
    m = main()
    print(json.dumps(m, indent=2, default=float))
