from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def analyze(data_path: Path, out_path: Path, holiday_path: Path,region:str):

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

    holidays = None
    if holiday_path is None:
        hp_candidates = [Path("analyze/data/holidays.csv"), Path("data/holidays.csv")] 
        for c in hp_candidates:
            if c.exists():
                holiday_path = c
                break
    if holiday_path and Path(holiday_path).exists():
        hp = pd.read_csv(holiday_path, parse_dates=["date"]) 
        holidays = set(hp["date"].dt.date.tolist())
    daily["is_holiday"] = daily["dow"].isin([5, 6]) if holidays is None else daily["date"].isin(holidays)

    def q25(x):
        return x.quantile(0.25)

    def q75(x):
        return x.quantile(0.75)

    grp = (
        daily.groupby("is_holiday")[ ["vol_sum", "occ_mean", "dur_mean", "dur_sum"] ]
        .agg(["mean", "median", q25, q75, "count"])
        .reset_index()
    )
    flat_cols = ["is_holiday"] + [f"{c[0]}_{c[1]}" for c in grp.columns[1:]]
    grp.columns = flat_cols

    outdir = out_path
    if outdir.suffix:
        outdir = outdir.parent
    if outdir.exists() and outdir.is_file():
        outdir = outdir.parent
    outdir.mkdir(parents=True, exist_ok=True)
    grp.to_csv(outdir / f"region-{region}-holiday-stats.csv", index=False)

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=False)
    wd_vals = daily.loc[~daily["is_holiday"], ["vol_sum", "occ_mean", "dur_mean"]]
    hd_vals = daily.loc[daily["is_holiday"], ["vol_sum", "occ_mean", "dur_mean"]]

    axes[0].boxplot([wd_vals["vol_sum"].values, hd_vals["vol_sum"].values], tick_labels=["Workday", "Holiday"])
    axes[0].set_ylabel("volume (daily sum)")
    axes[0].set_title(f"Region {region} Holiday Impact: Volume")

    axes[1].boxplot([wd_vals["occ_mean"].values, hd_vals["occ_mean"].values], tick_labels=["Workday", "Holiday"])
    axes[1].set_ylabel("occupancy (daily mean)")
    axes[1].set_title(f"Region {region} Holiday Impact: Occupancy")

    axes[2].boxplot([wd_vals["dur_mean"].values, hd_vals["dur_mean"].values], tick_labels=["Workday", "Holiday"])
    axes[2].set_ylabel("duration (daily mean)")
    axes[2].set_title(f"Region {region} Holiday Impact: Duration")
    axes[2].set_xlabel("group")

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / f"region-{region}-holiday.png", dpi=200)       
    plt.close(fig)

    wd_cnt = int((~daily["is_holiday"]).sum())
    hd_cnt = int(daily["is_holiday"].sum())
    print(f"总计T天={days}")
    print(f"工作日天数={wd_cnt}, 节假日天数={hd_cnt}")
    print(grp[["is_holiday", "vol_sum_mean", "occ_mean_mean", "dur_mean_mean"]])

if __name__ == "__main__":
    analyze(Path("analyze/result/region-102/region-102.csv"), Path("analyze/result/region-102/region-102-holiday.png"),region="102" )
