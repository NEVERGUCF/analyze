from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def analyze(data_path: Path, out_path: Path,region:str):


    df = pd.read_csv(data_path, parse_dates=["time"]) 
    if not {"volume", "occupancy", "duration"}.issubset(df.columns):
        raise ValueError("CSV 需包含列: volume, occupancy, duration")

    df["date"] = df["time"].dt.date
    df["dow"] = df["time"].dt.dayofweek

    days = df["date"].nunique()

    daily = df.groupby("date").agg({
        "volume": ["sum"],
        "occupancy": ["mean"],
        "duration": ["mean", "sum"],
        "dow": ["first"],
    }).reset_index()

    daily.columns = [
        "date",
        "vol_sum",
        "occ_mean",
        "dur_mean",
        "dur_sum",
        "dow",
    ]

    def q25(x):
        return x.quantile(0.25)

    def q75(x):
        return x.quantile(0.75)

    weekly = (
        daily.groupby("dow")[["vol_sum", "occ_mean", "dur_mean", "dur_sum"]]
        .agg(["mean", "median", q25, q75])
        .reset_index()
    )
    flat_cols = ["dow"] + [f"{c[0]}_{c[1]}" for c in weekly.columns[1:]]
    weekly.columns = flat_cols

    names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    weekly["name"] = weekly["dow"].map(names)

    outdir = out_path
    if outdir.suffix:
        outdir = outdir.parent
    if outdir.exists() and outdir.is_file():
        outdir = outdir.parent
    outdir.mkdir(parents=True, exist_ok=True)

    weekly.to_csv(outdir / f"region-{region}-weekly-stats.csv", index=False)

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    x = weekly["dow"].values
    xticks = [names[i] for i in x]

    axes[0].plot(x, weekly["vol_sum_mean"], color="#1f77b4", label="mean")
    axes[0].fill_between(x, weekly["vol_sum_q25"], weekly["vol_sum_q75"], color="#1f77b4", alpha=0.2, label="IQR")
    axes[0].set_ylabel("volume (daily sum)")
    axes[0].set_title(f"Region {region} Weekly Pattern: Volume")
    axes[0].legend()

    axes[1].plot(x, weekly["occ_mean_mean"], color="#ff7f0e", label="mean")
    axes[1].fill_between(x, weekly["occ_mean_q25"], weekly["occ_mean_q75"], color="#ff7f0e", alpha=0.2, label="IQR")
    axes[1].set_ylabel("occupancy (daily mean)")
    axes[1].set_title(f"Region {region} Weekly Pattern: Occupancy")
    axes[1].legend()

    axes[2].plot(x, weekly["dur_mean_mean"], color="#2ca02c", label="mean")
    axes[2].fill_between(x, weekly["dur_mean_q25"], weekly["dur_mean_q75"], color="#2ca02c", alpha=0.2, label="IQR")
    axes[2].set_ylabel("duration (daily mean)")
    axes[2].set_xlabel("day of week")
    axes[2].set_title(f"Region {region} Weekly Pattern: Duration")
    axes[2].legend()

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(xticks)
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / f"region-{region}-weekly.png", dpi=200)
    plt.close(fig)

    mon = weekly.loc[weekly["dow"] == 0]
    print(f"总计T天={days}")
    print("周统计预览:")
    print(weekly[["name", "vol_sum_mean", "occ_mean_mean", "dur_mean_mean"]])
    if not mon.empty:
        print(
            f"周一均值: volume日总={mon['vol_sum_mean'].values[0]:.2f}, "
            f"occupancy日均={mon['occ_mean_mean'].values[0]:.2f}, "
            f"duration日均={mon['dur_mean_mean'].values[0]:.2f}"
        )

if __name__ == "__main__":
    analyze(Path("data/region/region-102.csv"), Path("analyze/result/region-102/region-102-weekly.png"),"102")
