from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def analyze(data_path:Path,Out_Path:Path,region:str):

    df = pd.read_csv(data_path, parse_dates=["time"]) 

    if not {"volume", "occupancy", "duration"}.issubset(df.columns):
        raise ValueError("CSV 需包含列: volume, occupancy, duration")

    df["date"] = df["time"].dt.date
    df["hour"] = df["time"].dt.hour
    df["weekday"] = df["time"].dt.dayofweek < 5

    T_days = df["date"].nunique()

    def q25(x):
        return x.quantile(0.25)

    def q75(x):
        return x.quantile(0.75)

    hourly = (
        df.groupby("hour")[ ["volume", "occupancy", "duration"] ]
        .agg(["mean", "median", q25, q75])
        .reset_index()
    )
    # flatten multi-index columns
    flat_cols = ["hour"] + [f"{c[0]}_{c[1]}" for c in hourly.columns[1:]]
    hourly.columns = flat_cols

    outdir = Out_Path
    if outdir.suffix:
        outdir = outdir.parent
    if outdir.exists() and outdir.is_file():
        outdir = outdir.parent
    try:
        outdir.mkdir(parents=True, exist_ok=True)
    except Exception:
        outdir = outdir.parent
        outdir.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(outdir / f"region-{region}-hourly-stats.csv", index=False)

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    hours = hourly["hour"].values

    # volume
    axes[0].plot(hours, hourly["volume_mean"], label="mean", color="#1f77b4")
    axes[0].fill_between(hours, hourly["volume_q25"], hourly["volume_q75"], alpha=0.2, color="#1f77b4", label="IQR")
    axes[0].set_ylabel("volume")
    axes[0].set_title(f"Region {region} Weekly Pattern: Volume")
    axes[0].legend()

    # occupancy
    axes[1].plot(hours, hourly["occupancy_mean"], label="mean", color="#ff7f0e")
    axes[1].fill_between(hours, hourly["occupancy_q25"], hourly["occupancy_q75"], alpha=0.2, color="#ff7f0e", label="IQR")
    axes[1].set_ylabel("occupancy")
    axes[1].set_title(f"Region {region} Weekly Pattern: Occupancy")

    axes[1].legend()

    # duration
    axes[2].plot(hours, hourly["duration_mean"], label="mean", color="#2ca02c")
    axes[2].fill_between(hours, hourly["duration_q25"], hourly["duration_q75"], alpha=0.2, color="#2ca02c", label="IQR")
    axes[2].set_ylabel("duration")
    axes[2].set_xlabel("hour of day")
    axes[2].set_title(f"Region {region} Weekly Pattern: Duration")
    axes[2].legend()

    for ax in axes:
        ax.set_xticks(range(0, 24, 2))
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.axvspan(12, 14, color="gray", alpha=0.1, label="noon")

    fig.tight_layout()
    fig.savefig(outdir / f"region-{region}-hourly.png", dpi=200)
    plt.close(fig)

    # Midday comparison
    noon_mask = df["hour"].isin([12, 13])
    workday = df["weekday"]
    vol_noon_weekday = df.loc[noon_mask & workday, "volume"].mean()
    vol_noon_weekend = df.loc[noon_mask & (~workday), "volume"].mean()
    occ_noon_weekday = df.loc[noon_mask & workday, "occupancy"].mean()
    occ_noon_weekend = df.loc[noon_mask & (~workday), "occupancy"].mean()
    dur_noon_weekday = df.loc[noon_mask & workday, "duration"].mean()
    dur_noon_weekend = df.loc[noon_mask & (~workday), "duration"].mean()

    print(f"总计T天={T_days}")
    print("小时均值(部分预览):")
    print(hourly[["hour", "volume_mean", "occupancy_mean", "duration_mean"]].head())
    print(f"工作日中午(12-13) volume均值={vol_noon_weekday:.2f}, 周末={vol_noon_weekend:.2f}")
    print(f"工作日中午(12-13) occupancy均值={occ_noon_weekday:.2f}, 周末={occ_noon_weekend:.2f}")
    print(f"工作日中午(12-13) duration均值={dur_noon_weekday:.2f}, 周末={dur_noon_weekend:.2f}")

if __name__ == "__main__":
    analyze(Path("data/region/region-102.csv"),Path("analyze/result/region"),"102")
