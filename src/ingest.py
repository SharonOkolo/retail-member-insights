"""Download UCI Online Retail II and convert both Excel sheets to one CSV (bronze input)."""
import os, subprocess, zipfile
import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def main():
    os.makedirs(RAW, exist_ok=True)
    out = os.path.join(RAW, "online_retail_ii.csv")
    if os.path.exists(out):
        return out
    z = os.path.join(RAW, "online_retail_ii.zip")
    if not os.path.exists(z):
        subprocess.run(["curl", "-sSL", "--retry", "3", "-o", z, URL], check=True)
    zipfile.ZipFile(z).extractall(RAW)
    sheets = pd.read_excel(os.path.join(RAW, "online_retail_II.xlsx"), sheet_name=None, dtype={"Invoice": str, "StockCode": str})
    pd.concat(sheets.values(), ignore_index=True).to_csv(out, index=False)
    return out


if __name__ == "__main__":
    print(main())
