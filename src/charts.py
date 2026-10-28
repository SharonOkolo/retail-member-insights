"""Save the README / LinkedIn charts from the model outputs."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pipeline import ROOT

OUT, SHOTS = os.path.join(ROOT, "outputs"), os.path.join(ROOT, "screenshots")
INK, MUTED, ACCENT, GREY = "#1f2933", "#616e7c", "#2563eb", "#cbd2d9"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GREY, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})


def segments():
    s = pd.read_csv(os.path.join(OUT, "segments_summary.csv"))
    s["cust_share"] = s.customers / s.customers.sum()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    y = range(len(s))
    ax.barh([i + 0.2 for i in y], s.cust_share * 100, 0.4, color=GREY, label="Share of customers")
    ax.barh([i - 0.2 for i in y], s.revenue_share * 100, 0.4, color=ACCENT, label="Share of revenue")
    ax.set_yticks(list(y), s.segment); ax.invert_yaxis(); ax.set_xlabel("%")
    for i, r in s.iterrows():
        ax.text(r.revenue_share * 100 + 1, i - 0.2, f"{r.revenue_share:.0%}", va="center", color=INK)
    c = s[s.segment == "Champions"].iloc[0]
    fig.suptitle(f"{c.cust_share:.0%} of customers (Champions) bring in {c.revenue_share:.0%} of revenue", x=0.01, ha="left", color=INK, fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(SHOTS, "01_segments.png"), dpi=160); plt.close(fig)


def forecast():
    m = json.load(open(os.path.join(OUT, "metrics.json")))
    w = m["forecast_wape"]
    names = sorted(w, key=w.get, reverse=True)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.barh(names, [w[n] * 100 for n in names], color=[ACCENT if n == m["forecast_best"] else GREY for n in names])
    for i, n in enumerate(names):
        ax.text(w[n] * 100 + 1, i, f"{w[n]:.0%}", va="center", color=INK)
    ax.set_xlabel("Forecast error (WAPE, %) on the last 8 weeks, top 20 products: lower is better")
    cut = 1 - w[m["forecast_best"]] / w["naive (last week)"]
    fig.suptitle(f"Gradient boosting: {cut:.0%} less forecast error than a naive forecast", x=0.01, ha="left", color=INK, fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(SHOTS, "02_forecast_error.png"), dpi=160); plt.close(fig)


def fairness():
    f = pd.read_csv(os.path.join(OUT, "fairness_lapse_model.csv"))
    f = f[f.dimension == "spend band"]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar(f.group, f.recall * 100, color=[ACCENT if g == "top 25%" else GREY for g in f.group])
    for i, r in enumerate(f.itertuples()):
        ax.text(i, r.recall * 100 + 2, f"{r.recall:.0%}", ha="center", color=INK)
    ax.set_ylabel("Lapsing customers caught (%)"); ax.set_xlabel("Customer spend band"); ax.set_ylim(0, 110)
    fig.suptitle("At one threshold, the model misses most lapsing top spenders", x=0.01, ha="left", color=INK, fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(SHOTS, "03_fairness_recall.png"), dpi=160); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(SHOTS, exist_ok=True)
    segments(); forecast(); fairness()
    print(sorted(os.listdir(SHOTS)))
