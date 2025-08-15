#!/usr/bin/env python3

# usage
python data_prep.py --csv samples_mixed.csv --src output_maps_csv --out data_cache --limit 970

import os, json, argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

def load_hist_txt(prefix: str):
    """读取 prefix 对应的线性与 log 直方图，返回 (mu_linear, cnt_linear, mu_log, cnt_log)，若均缺失则返回 None"""
    mags_path = prefix + "ipm_mags_numpixels.txt"
    log_path  = prefix + "ipm_log_mags_numpixels.txt"

    mu_lin = cnt_lin = None
    mu_log = cnt_log = None

    if os.path.exists(mags_path):
        arr = np.genfromtxt(mags_path, usecols=(0, 1))
        if arr.size:
            arr = np.atleast_2d(arr)
            mu  = arr[:, 0].astype(np.float64) / 1000.0
            cnt = arr[:, 1].astype(np.int64)
            order = np.argsort(mu)
            mu_lin = mu[order]
            cnt_lin = cnt[order]

    if os.path.exists(log_path):
        arr = np.genfromtxt(log_path, usecols=(0, 1))
        if arr.size:
            arr = np.atleast_2d(arr)
            b   = arr[:, 0].astype(np.float64)  # 文本文件的 bin 起点 (log μ * 100)
            cnt = arr[:, 1].astype(np.int64)
            L   = b / 100.0                     # bin 起点 log μ
            dL  = 0.01
            logmu_mid = L + 0.5 * dL             # bin 中点 log μ
            order = np.argsort(logmu_mid)
            mu_log = logmu_mid[order]            # 这里保存的是 log μ
            cnt_log = cnt[order]


    if mu_lin is None and mu_log is None:
        return None
    return mu_lin, cnt_lin, mu_log, cnt_log

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="samples_mixed.csv", help="input CSV (kappa,gamma,s)")
    ap.add_argument("--src", default="output_maps", help="folder where raw txt hists are")
    ap.add_argument("--out", default="data_cache", help="output cache folder")
    ap.add_argument("--limit", type=int, default=None, help="only first N rows (optional)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    hist_out = os.path.join(args.out, "hists"); os.makedirs(hist_out, exist_ok=True)

    # 读 CSV → 增加 row_index/rid
    df = pd.read_csv(args.csv)
    if args.limit is not None:
        df = df.iloc[:args.limit]
    df = df.reset_index(drop=True)
    df["row_index"] = df.index + 1
    df["rid"] = df["row_index"].apply(lambda i: f"r{i:04d}")

    # 保存 parquet
    # 保存 samples：优先 parquet，缺引擎则降级 csv
    parquet_path = os.path.join(args.out, "samples.parquet")
    csv_fallback = os.path.join(args.out, "samples.csv")

    parquet_ok = True
    try:
        # 只测试是否有任一 parquet 引擎
        try:
            import pyarrow  # noqa: F401
        except Exception:
            import fastparquet  # noqa: F401
    except Exception:
        parquet_ok = False

    if parquet_ok:
        df.to_parquet(parquet_path, index=False)
        print(f"[OK] wrote {parquet_path}")
    else:
        df.to_csv(csv_fallback, index=False)
        print(f"[WARN] parquet engine not found; wrote CSV fallback: {csv_fallback}")


    # 遍历缓存直方图
    manifest = {}
    for _, row in tqdm(df.iterrows()):
        rid = row["rid"]
        prefix_txt = os.path.join(args.src, f"{rid}_")
        got = load_hist_txt(prefix_txt)
        if got is None:
            continue
        mu_lin, cnt_lin, mu_log, cnt_log = got

        npz_path = os.path.join(hist_out, f"{rid}.npz")
        data = {}
        if mu_lin is not None:
            data["mu_linear"] = mu_lin.astype(np.float32)
            data["cnt_linear"] = cnt_lin.astype(np.int32)
        # if mu_log is not None:
        #     data["mu_log"] = mu_log.astype(np.float32)
        #     data["cnt_log"] = cnt_log.astype(np.int32)
        if mu_log is not None:
            data["logmu_mid"] = mu_log.astype(np.float32)   # log μ bin 中点
            data["cnt_log"] = cnt_log.astype(np.int32)      # 对应 bin 的计数

        if not data:
            continue
        np.savez_compressed(npz_path, **data)

        manifest[rid] = {}
        if mu_lin is not None:
            manifest[rid]["linear_bins"] = int(len(mu_lin))
        if mu_log is not None:
            manifest[rid]["log_bins"] = int(len(mu_log))
        

    # 写清单
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump({"count": len(manifest), "items": manifest}, f, indent=2)

    print(f"[OK] wrote {parquet_path}")
    print(f"[OK] cached {len(manifest)} hist(s) under {hist_out}")

if __name__ == "__main__":
    main()
