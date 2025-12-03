from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def build_daily_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df["date"] = df["time"].dt.date
    df["hour"] = df["time"].dt.hour
    hours = list(range(24))
    mats = []
    cols = []
    for feat, prefix in [("volume", "vol"), ("occupancy", "occ"), ("duration", "dur"), ("volume-11kW", "vol11"), ("s_price", "sp"), ("e_price", "ep")]:
        pv = df.pivot_table(index="date", columns="hour", values=feat, aggfunc="mean")
        pv = pv.reindex(columns=hours)
        pv = pv.fillna(pv.mean())
        pv.columns = [f"{prefix}_h{h:02d}" for h in pv.columns]
        mats.append(pv)
        cols.extend(list(pv.columns))
    mat = pd.concat(mats, axis=1)
    return mat, cols

def choose_k(X: np.ndarray, k_min: int = 2, k_max: int = 6) -> int:
    best_k, best_score = k_min, -1.0
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        try:
            score = silhouette_score(X, labels)
        except Exception:
            score = -1.0
        if score > best_score:
            best_k, best_score = k, score
    return best_k

def analyze(data_path: Path, out_path: Path, region: str):
    if not data_path.exists():
        candidates = [
            Path(f"analyze/result/region-{region}/region-{region}-all.csv"),
            Path(f"analyze/result/region-{region}/region-{region}.csv"),
            Path(f"result/region/region-{region}-all.csv"),
            Path(f"result/region/region-{region}.csv"),
            Path(f"data/region-{region}-all.csv"),
            Path(f"data/region-{region}.csv"),
        ]
        for c in candidates:
            if c.exists():
                data_path = c
                break

    df = pd.read_csv(data_path, parse_dates=["time"]) 
    if not {"volume", "occupancy", "duration", "volume-11kW", "s_price", "e_price"}.issubset(df.columns):
        raise ValueError("CSV 需包含列: volume, occupancy, duration, volume-11kW, s_price, e_price")

    mat, cols = build_daily_matrix(df)
    dates = mat.index.values

    scaler = StandardScaler()
    X = scaler.fit_transform(mat.values)

    k = choose_k(X, 2, 6)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    outdir = out_path
    if outdir.suffix:
        outdir = outdir.parent
    if outdir.exists() and outdir.is_file():
        outdir = outdir.parent
    outdir.mkdir(parents=True, exist_ok=True)

    df_feat = mat.copy()
    df_feat.insert(0, "date", dates)
    df_feat.to_csv(outdir / f"region-{region}-daily-features.csv", index=False)

    pd.DataFrame({"date": dates, "cluster": labels}).to_csv(outdir / f"region-{region}-cluster-assignments.csv", index=False)

    prof = []
    for c in range(k):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        sub = mat.iloc[idx]
        vol = sub[[c for c in cols if c.startswith("vol_")]].mean().values
        occ = sub[[c for c in cols if c.startswith("occ_")]].mean().values
        dur = sub[[c for c in cols if c.startswith("dur_")]].mean().values
        vol11 = sub[[c for c in cols if c.startswith("vol11_")]].mean().values
        sp = sub[[c for c in cols if c.startswith("sp_")]].mean().values
        ep = sub[[c for c in cols if c.startswith("ep_")]].mean().values
        for h in range(24):
            prof.append({"cluster": c, "hour": h, "volume": vol[h], "occupancy": occ[h], "duration": dur[h], "volume-11kW": vol11[h], "s_price": sp[h], "e_price": ep[h]})
    df_prof = pd.DataFrame(prof)
    df_prof.to_csv(outdir / f"region-{region}-cluster-profiles.csv", index=False)

    fig, axes = plt.subplots(6, 1, figsize=(12, 20), sharex=True)
    hours = np.arange(24)
    colors = plt.cm.tab10(np.arange(k))
    for c in range(k):
        sub = df_prof[df_prof["cluster"] == c]
        axes[0].plot(hours, sub["volume"].values, color=colors[c], label=f"cluster {c}")
        axes[1].plot(hours, sub["occupancy"].values, color=colors[c], label=f"cluster {c}")
        axes[2].plot(hours, sub["duration"].values, color=colors[c], label=f"cluster {c}")
        axes[3].plot(hours, sub["volume-11kW"].values, color=colors[c], label=f"cluster {c}")
        axes[4].plot(hours, sub["s_price"].values, color=colors[c], label=f"cluster {c}")
        axes[5].plot(hours, sub["e_price"].values, color=colors[c], label=f"cluster {c}")
    axes[0].set_ylabel("volume")
    axes[0].set_title(f"Region {region} Daily Profile Clusters: Volume")
    axes[1].set_ylabel("occupancy")
    axes[1].set_title(f"Region {region} Daily Profile Clusters: Occupancy")
    axes[2].set_ylabel("duration")
    axes[2].set_title(f"Region {region} Daily Profile Clusters: Duration")
    axes[3].set_ylabel("volume-11kW")
    axes[3].set_title(f"Region {region} Daily Profile Clusters: Volume-11kW")
    axes[4].set_ylabel("s_price")
    axes[4].set_title(f"Region {region} Daily Profile Clusters: s_price")
    axes[5].set_ylabel("e_price")
    axes[5].set_xlabel("hour of day")
    axes[5].set_title(f"Region {region} Daily Profile Clusters: e_price")
    for ax in axes:
        ax.set_xticks(range(0, 24, 2))
        ax.grid(True, linestyle="--", alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(outdir / f"region-{region}-cluster-profiles.png", dpi=200)
    plt.close(fig)

    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)
    C2 = pca.transform(km.cluster_centers_)
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
    colors = plt.cm.tab10(np.arange(k))
    for c in range(k):
        idx = np.where(labels == c)[0]
        ax2.scatter(X2[idx, 0], X2[idx, 1], s=20, color=colors[c], alpha=0.6, label=f"cluster {c}")
    ax2.scatter(C2[:, 0], C2[:, 1], s=160, color="black", marker="*", label="centroid")
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(outdir / f"region-{region}-cluster-scatter.png", dpi=200)
    plt.close(fig2)

    ks = list(range(2, 7))
    scores = []
    for kk in ks:
        m = KMeans(n_clusters=kk, random_state=42, n_init=10)
        lab = m.fit_predict(X)
        try:
            s = silhouette_score(X, lab)
        except Exception:
            s = np.nan
        scores.append(s)
    fig3, ax3 = plt.subplots(1, 1, figsize=(8, 4))
    ax3.plot(ks, scores, marker="o")
    ax3.set_xticks(ks)
    ax3.set_xlabel("k")
    ax3.set_ylabel("silhouette")
    ax3.grid(True, linestyle="--", alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(outdir / f"region-{region}-cluster-silhouette.png", dpi=200)
    plt.close(fig3)

    print(f"聚类簇数k={k}")
    print("示例簇均值预览：")
    print(df_prof.groupby(["cluster", "hour"])[["volume", "occupancy", "duration", "volume-11kW", "s_price", "e_price"]].mean().reset_index().head())

if __name__ == "__main__":
    r = "102"
    analyze(Path(f"analyze/result/region-{r}/region-{r}-all.csv"), Path(f"analyze/result/region-{r}/cluster/region-{r}-cluster-profiles.png"), r)

def analyze_by_hour(data_path: Path, out_path: Path, region: str):
    if not data_path.exists():
        candidates = [
            Path(f"analyze/result/region-{region}/region-{region}-all.csv"),
            Path(f"analyze/result/region-{region}/region-{region}.csv"),
            Path(f"result/region/region-{region}-all.csv"),
            Path(f"result/region/region-{region}.csv"),
            Path(f"data/region-{region}-all.csv"),
            Path(f"data/region-{region}.csv"),
        ]
        for c in candidates:
            if c.exists():
                data_path = c
                break

    df = pd.read_csv(data_path, parse_dates=["time"]) 
    if not {"volume", "occupancy", "duration", "volume-11kW", "s_price", "e_price"}.issubset(df.columns):
        raise ValueError("CSV 需包含列: volume, occupancy, duration, volume-11kW, s_price, e_price")

    df["hour"] = df["time"].dt.hour
    feats = ["volume", "occupancy", "duration", "volume-11kW", "s_price", "e_price"]
    mat = df[feats].copy()

    scaler = StandardScaler()
    X = scaler.fit_transform(mat.values)

    k = choose_k(X, 2, 6)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    outdir = out_path
    if outdir.suffix:
        outdir = outdir.parent
    if outdir.exists() and outdir.is_file():
        outdir = outdir.parent
    outdir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"time": df["time"], "hour": df["hour"], "cluster": labels}).to_csv(outdir / f"region-{region}-hour-assignments.csv", index=False)

    prof = df.copy()
    prof["cluster"] = labels
    g1 = prof.groupby("cluster")[feats].mean().reset_index()
    g1.to_csv(outdir / f"region-{region}-hour-cluster-profiles.csv", index=False)

    g2 = prof.groupby(["cluster", "hour"])[feats].mean().reset_index()
    g2.to_csv(outdir / f"region-{region}-hour-cluster-hourly.csv", index=False)

    hours = np.arange(24)
    fig, axes = plt.subplots(6, 1, figsize=(12, 20), sharex=True)
    colors = plt.cm.tab10(np.arange(k))
    for c in range(k):
        sub = g2[g2["cluster"] == c]
        axes[0].plot(hours, sub.set_index("hour")["volume"].reindex(hours).values, color=colors[c], label=f"cluster {c}")
        axes[1].plot(hours, sub.set_index("hour")["occupancy"].reindex(hours).values, color=colors[c], label=f"cluster {c}")
        axes[2].plot(hours, sub.set_index("hour")["duration"].reindex(hours).values, color=colors[c], label=f"cluster {c}")
        axes[3].plot(hours, sub.set_index("hour")["volume-11kW"].reindex(hours).values, color=colors[c], label=f"cluster {c}")
        axes[4].plot(hours, sub.set_index("hour")["s_price"].reindex(hours).values, color=colors[c], label=f"cluster {c}")
        axes[5].plot(hours, sub.set_index("hour")["e_price"].reindex(hours).values, color=colors[c], label=f"cluster {c}")
    axes[0].set_ylabel("volume")
    axes[0].set_title(f"Region {region} Hour Clusters: Volume")
    axes[1].set_ylabel("occupancy")
    axes[1].set_title(f"Region {region} Hour Clusters: Occupancy")
    axes[2].set_ylabel("duration")
    axes[2].set_title(f"Region {region} Hour Clusters: Duration")
    axes[3].set_ylabel("volume-11kW")
    axes[3].set_title(f"Region {region} Hour Clusters: Volume-11kW")
    axes[4].set_ylabel("s_price")
    axes[4].set_title(f"Region {region} Hour Clusters: s_price")
    axes[5].set_ylabel("e_price")
    axes[5].set_xlabel("hour of day")
    axes[5].set_title(f"Region {region} Hour Clusters: e_price")
    for ax in axes:
        ax.set_xticks(range(0, 24, 2))
        ax.grid(True, linestyle="--", alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(outdir / f"region-{region}-hour-cluster-hourly.png", dpi=200)
    plt.close(fig)

    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)
    C2 = pca.transform(km.cluster_centers_)
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
    colors = plt.cm.tab10(np.arange(k))
    for c in range(k):
        idx = np.where(labels == c)[0]
        ax2.scatter(X2[idx, 0], X2[idx, 1], s=10, color=colors[c], alpha=0.5, label=f"cluster {c}")
    ax2.scatter(C2[:, 0], C2[:, 1], s=160, color="black", marker="*", label="centroid")
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(outdir / f"region-{region}-hour-cluster-scatter.png", dpi=200)
    plt.close(fig2)

    print(f"按小时聚类簇数k={k}")
    print(g1.head())
