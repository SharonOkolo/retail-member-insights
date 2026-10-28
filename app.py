"""Streamlit app: member segments, win-back list, demand forecast and responsible-AI checks."""
import json, os
import pandas as pd
import plotly.express as px
import streamlit as st

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
st.set_page_config(page_title="Retail member insights", layout="wide")


@st.cache_data
def load():
    r = lambda f: pd.read_csv(os.path.join(OUT, f))
    return (json.load(open(os.path.join(OUT, "metrics.json"))), r("segments_summary.csv"), r("customer_scores.csv"),
            r("forecast_holdout.csv"), r("weekly_history_top20.csv"), r("fairness_lapse_model.csv"), r("data_quality_log.csv"))


m, seg, cust, fc, hist, fair, dq = load()
st.title("Retail member insights")
st.caption("UK online retailer, Dec 2009 – Dec 2011 (UCI Online Retail II, CC BY 4.0). "
           "Pipeline: bronze → silver → gold SQL layers, built for Databricks and runnable locally on DuckDB.")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Transactions cleaned", f"{m['rows_silver']:,}", help=f"From {m['rows_bronze']:,} raw rows: see the Data quality tab")
k2.metric("Customers segmented", f"{m['customers']:,}")
k3.metric("Lapse model ROC AUC", f"{m['lapse_models'][m['lapse_best']]['roc_auc']:.2f}")
cut = 1 - m["forecast_wape"][m["forecast_best"]] / m["forecast_wape"]["naive (last week)"]
k4.metric("Forecast error vs naive", f"-{cut:.0%}")

t1, t2, t3, t4, t5 = st.tabs(["Segments", "Win-back list", "Demand forecast", "Responsible AI", "Data quality"])

with t1:
    long = seg.melt(id_vars="segment", value_vars=["customers", "revenue"], var_name="measure")
    long["share"] = long.groupby("measure").value.transform(lambda v: v / v.sum())
    st.plotly_chart(px.bar(long, y="segment", x="share", color="measure", barmode="group", orientation="h",
                           labels={"share": "Share", "segment": ""}).update_xaxes(tickformat=".0%"), use_container_width=True)
    st.dataframe(seg[["segment", "customers", "recency_days", "frequency", "monetary", "revenue_share"]]
                 .rename(columns={"recency_days": "median days since last order", "frequency": "median orders",
                                  "monetary": "median spend (£)", "revenue_share": "share of revenue"}),
                 hide_index=True, use_container_width=True)
    st.markdown("k-means on log-scaled recency, frequency and spend; k chosen by silhouette score "
                f"(best k = {m['k']}). Segment names come from rules, so they stay right if the data changes.")

with t2:
    st.markdown("Customers ranked by **lapse probability × spend**: the value at stake if they stop buying.")
    c1, c2 = st.columns(2)
    segs = c1.multiselect("Segments", seg.segment.tolist(), default=["Champions", "Slipping high-value"])
    thr = c2.slider("Flag customers with lapse probability of at least", 0.1, 0.9, 0.3, 0.05)
    view = cust[cust.segment.isin(segs) & (cust.lapse_prob >= thr)].assign(value_at_stake=lambda d: d.lapse_prob * d.monetary)
    view = view.sort_values("value_at_stake", ascending=False)
    st.metric("Customers flagged", f"{len(view):,}")
    st.caption(f"£{view.monetary.sum():,.0f} spent by these customers over two years")
    st.dataframe(view[["customer_id", "segment", "country", "recency_days", "frequency", "monetary", "lapse_prob", "value_at_stake"]]
                 .head(200).round(2), hide_index=True, use_container_width=True)
    st.download_button("Download list (CSV)", view.to_csv(index=False), "win_back_list.csv")

with t3:
    names = hist.drop_duplicates("stock_code").set_index("stock_code").description.str.strip()
    code = st.selectbox("Product (top 20 by units)", names.index, format_func=lambda c: f"{c} · {names[c]}")
    h = hist[hist.stock_code.astype(str) == str(code)]
    f = fc[fc.stock_code.astype(str) == str(code)].melt(id_vars=["week_start"], value_vars=list(m["forecast_wape"]),
                                                        var_name="series", value_name="units")
    plot = pd.concat([h.assign(series="actual")[["week_start", "series", "units"]], f])
    st.plotly_chart(px.line(plot, x="week_start", y="units", color="series", labels={"week_start": ""}), use_container_width=True)
    w = pd.Series(m["forecast_wape"]).rename("WAPE (all 20 products)").to_frame()
    st.dataframe(w.style.format("{:.0%}"), use_container_width=True)
    st.markdown(f"Hold-out: the 8 weeks after {m['forecast_origin']}. The global gradient-boosting model avoided "
                f"**{m['units_error_avoided_vs_naive']:,.0f} units** of forecast error compared with repeating last week's sales.")

with t4:
    st.markdown("Does the lapse model work equally well for every group? Error rates on the test set at a 0.5 threshold:")
    st.dataframe(fair.round(2), hide_index=True, use_container_width=True)
    st.markdown(f"**Finding:** the model catches most lapsing low spenders but only "
                f"{fair[fair.group == 'top 25%'].recall.iloc[0]:.0%} of lapsing **top-25% spenders**, the customers worth most. "
                f"Lowering their threshold to 0.3 raises that to {m['top_band_recall_at_0_3']:.0%}, with "
                f"{m['top_band_fpr_at_0_3']:.0%} of loyal top spenders flagged by mistake. See the model card in the repo.")

with t5:
    st.dataframe(dq, hide_index=True, use_container_width=True)
    st.markdown("Every excluded row is counted by reason, and the pipeline's checks fail the run if the counts don't reconcile.")
