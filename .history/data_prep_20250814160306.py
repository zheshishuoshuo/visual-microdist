#!/usr/bin/env python3
import os, json, argparse
import numpy as np
import pandas as pd

def load_hist_txt_to_mu_cnt(prefix: str):
    """优先线性直方图；失败则尝试 log 直方图（ln μ, ×100）→ 返回 (mu, cnt, src) 或 None"""
    mags_path = prefix + "ipm_mags_numpixels.txt"
    log_path  = prefix + "ipm_log_mags_numpixels.txt"

    # 1) 线性
    if os.path.exists(mags_path):
        arr = np.genfromtxt(mags_path, usecols=(0,1))
        if arr.size:
            arr = np.atleast_2d(arr)
            mu  = arr[:,0].astype(np.float64) / 1000.0
            cnt = arr[:,1].astype(np.int64)
            order = np.argsort(mu)
            return mu[order], cnt[order], "linear"

    # 2) log → μ 中点
    if os.path.exists(log_path):
        arr = np.genfromtxt(log_path, usecols=(0,1))
        if arr.size:
            arr = np.atleast_2d(arr)
            b   = arr[:,0].astype(np.float64)
            cnt = arr[:,1].astype(np.int64)
            L   = b / 100.0
            dL  = 0.01
            mu_mid = np.exp(L + 0.5*dL)
            order  = np.argsort(mu_mid)
            return mu_mid[order], cnt[order], "log"

    return None

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
    for _, row in df.iterrows():
        rid = row["rid"]
        prefix_txt = os.path.join(args.src, f"{rid}_")
        got = load_hist_txt_to_mu_cnt(prefix_txt)
        if got is None:
            continue
        mu, cnt, src = got
        # 存为 NPZ（float32/int32 足够 & 更小）
        npz_path = os.path.join(hist_out, f"{rid}.npz")
        np.savez_compressed(npz_path,
                            mu=mu.astype(np.float32),
                            cnt=cnt.astype(np.int32),
                            src=np.array([0 if src=="linear" else 1], dtype=np.int8))
        manifest[rid] = {"bins": int(len(mu)), "src": src}

    # 写清单
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump({"count": len(manifest), "items": manifest}, f, indent=2)

    print(f"[OK] wrote {parquet_path}")
    print(f"[OK] cached {len(manifest)} hist(s) under {hist_out}")

if __name__ == "__main__":
    main()
