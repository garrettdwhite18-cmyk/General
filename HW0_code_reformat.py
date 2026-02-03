from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path("/Users/garrettwh/Downloads/Data").expanduser().resolve()
OUT_DIR = (DATA_DIR / "output").resolve()
TZ = "America/Chicago"
MONTH_MAP = {"F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6, "N": 7, "Q": 8,
             "U": 9, "V": 10, "X": 11, "Z": 12}
CONTRACT_RE = re.compile(r"^(?P<root>[A-Z0-9]+?)(?P<month>[FGHJKMNQUVXZ])(?P<year>\d{1,2})$")


@dataclass(frozen=True)
class ContractKey:
    root: str
    month_code: str
    year_code: str

    @property
    def ym(self) -> Tuple[int, int]:
        y = self.year_code
        if len(y) == 1:
            year = 2020 + int(y)
        else:
            yy = int(y)
            year = 2000 + yy if yy < 100 else yy
        month = MONTH_MAP[self.month_code]
        return (year, month)

    @property
    def code(self) -> str:
        return f"{self.root}{self.month_code}{self.year_code}"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def list_dbn_files(data_dir: Path) -> List[Path]:
    files = sorted([p for p in data_dir.rglob("*.dbn") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No .dbn files found under {data_dir}")
    return files


def read_dbn_to_df(path: Path) -> pd.DataFrame:
    errs = []
    try:
        import databento as db
        try:
            store = db.DBNStore.from_file(str(path))
            return store.to_df()
        except Exception as e:
            errs.append(("databento.DBNStore", str(e)))
    except Exception as e:
        errs.append(("import databento", str(e)))

    try:
        from databento_dbn import DBNStore
        store = DBNStore.from_file(str(path))
        return store.to_df()
    except Exception as e:
        errs.append(("databento_dbn.DBNStore", str(e)))

    msg = "Could not read DBN. Tried:\n" + "\n".join([f"- {k}: {v}" for k, v in errs])
    raise RuntimeError(msg)


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {c.lower(): c for c in df.columns}

    sym_col = None
    for n in ("symbol", "instrument", "ticker", "raw_symbol"):
        if n in colmap:
            sym_col = colmap[n]
            break
    if sym_col is None:
        raise ValueError(f"Missing symbol column. Have: {list(df.columns)}")

    ts_col = None
    for n in ("ts_event", "ts_recv", "ts", "timestamp", "time", "datetime", "date_time"):
        if n in colmap:
            ts_col = colmap[n]
            break
    if ts_col is None:
        if isinstance(df.index, pd.DatetimeIndex):
            out = df.reset_index()
            if "index" in out.columns:
                out = out.rename(columns={"index": "ts"})
            else:
                out = out.rename(columns={out.columns[0]: "ts"})
            df = out
            ts_col = "ts"
        else:
            raise ValueError(f"Missing timestamp column and index is not DatetimeIndex. Have: {list(df.columns)}")

    close_col = colmap.get("close") if "close" not in df.columns else "close"
    if close_col is None:
        raise ValueError(f"Missing close column. Have: {list(df.columns)}")

    out = df[[ts_col, sym_col, close_col]].copy()
    out = out.rename(columns={ts_col: "ts", sym_col: "symbol", close_col: "close"})
    return out


def parse_contract(symbol: str) -> Optional[ContractKey]:
    m = CONTRACT_RE.match(symbol)
    if not m:
        return None
    root = m.group("root")
    month = m.group("month")
    year = m.group("year")
    if month not in MONTH_MAP:
        return None
    return ContractKey(root=root, month_code=month, year_code=year)


def load_all(files: List[Path], tz: str) -> pd.DataFrame:
    chunks = []
    for p in files:
        df = read_dbn_to_df(p)
        df = normalize_cols(df)
        chunks.append(df)

    df = pd.concat(chunks, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts", "symbol", "close"])
    df["ts"] = df["ts"].dt.tz_convert(tz)
    df["symbol"] = df["symbol"].astype(str)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    return df


def pick_top2_by_rows(df: pd.DataFrame, root: str) -> Tuple[ContractKey, ContractKey]:
    sub = df[df["root"] == root].copy()
    if sub.empty:
        raise ValueError(f"No rows for root {root}")
    counts = sub.groupby("symbol", observed=True).size().sort_values(ascending=False)

    chosen: List[ContractKey] = []
    for s in list(counts.index):
        ck = parse_contract(s)
        if ck is not None and ck.root == root:
            chosen.append(ck)
        if len(chosen) >= 2:
            break

    if len(chosen) < 2:
        raise ValueError(
            f"Could not find 2 parsable contracts for root {root}. Symbols seen include: {list(counts.index[:30])}"
        )

    chosen = sorted(chosen[:2], key=lambda x: x.ym)
    return chosen[0], chosen[1]


def minute_index_active_span(day_df: pd.DataFrame, tz: str) -> Optional[pd.DatetimeIndex]:
    if day_df.empty:
        return None
    tmin = day_df.index.min()
    tmax = day_df.index.max()
    if pd.isna(tmin) or pd.isna(tmax):
        return None
    start = tmin.floor("min")
    end = tmax.ceil("min")
    if end < start:
        return None
    return pd.date_range(start=start, end=end, freq="1min", tz=tz)


def ffill_with_bounds(series: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    if series.empty:
        return pd.Series(index=idx, data=np.nan)
    s = series[~series.index.duplicated(keep="last")].reindex(idx)
    first_obs = series.index.min().floor("min")
    last_obs = series.index.max().ceil("min")
    s = s.ffill()
    mask_before = idx < first_obs
    mask_after = idx > last_obs
    if mask_before.any():
        s.loc[mask_before] = np.nan
    if mask_after.any():
        s.loc[mask_after] = np.nan
    return s


def build_spread_panel(df: pd.DataFrame, root: str, front: ContractKey, second: ContractKey, tz: str) -> pd.DataFrame:
    sub = df[(df["root"] == root) & (df["symbol"].isin([front.code, second.code]))].copy()
    if sub.empty:
        raise ValueError(f"No data for {root} among {front.code}, {second.code}")
    sub["date"] = sub["ts"].dt.normalize()
    out_rows = []

    for d, g in sub.groupby("date", observed=True):
        g2 = g.set_index("ts").sort_index()
        idx = minute_index_active_span(g2, tz)
        if idx is None or len(idx) == 0:
            continue
        f_raw = g2[g2["symbol"] == front.code]["close"]
        s_raw = g2[g2["symbol"] == second.code]["close"]
        f = ffill_with_bounds(f_raw, idx) if not f_raw.empty else pd.Series(index=idx, data=np.nan)
        s = ffill_with_bounds(s_raw, idx) if not s_raw.empty else pd.Series(index=idx, data=np.nan)
        panel = pd.DataFrame({f"{root}_front": f, f"{root}_second": s}, index=idx)
        panel[f"{root}_spread"] = panel[f"{root}_second"] - panel[f"{root}_front"]
        panel["date"] = panel.index.normalize()
        out_rows.append(panel)

    if not out_rows:
        raise ValueError(f"No usable day panels built for {root}")
    out = pd.concat(out_rows).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def trading_day_rolling_d(spread: pd.Series, date_index: pd.Series, Ns: Iterable[int]) -> Dict[int, pd.Series]:
    df = pd.DataFrame({"spread": spread, "date": date_index})
    daily = df.dropna(subset=["spread"]).groupby("date", observed=True)["spread"].mean().sort_index()
    out: Dict[int, pd.Series] = {}
    for N in Ns:
        daily_roll = daily.rolling(window=int(N), min_periods=1).mean()
        bench = date_index.map(daily_roll)
        out[int(N)] = spread - bench
    return out


def quantiles(x: pd.Series) -> pd.Series:
    qs = [0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999]
    return x.quantile(qs, interpolation="linear")


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def corr_pair(a: pd.Series, b: pd.Series) -> float:
    z = pd.concat([a, b], axis=1).dropna()
    if z.shape[0] < 2:
        return np.nan
    return float(z.corr().iloc[0, 1])


def intraday_profile(spread: pd.Series) -> pd.Series:
    x = spread.dropna()
    if x.empty:
        return pd.Series(dtype=float)
    mod = x.index.hour * 60 + x.index.minute
    return x.groupby(mod).mean().sort_index()


def run():
    ensure_dir(OUT_DIR)
    files = list_dbn_files(DATA_DIR)
    raw = load_all(files, TZ)
    raw["contract"] = raw["symbol"].map(parse_contract)
    raw = raw.dropna(subset=["contract"]).copy()
    raw["root"] = raw["contract"].map(lambda x: x.root)
    raw = raw[raw["root"].isin(["GC", "SIL"])].copy()

    gc_front, gc_second = pick_top2_by_rows(raw, "GC")
    sil_front, sil_second = pick_top2_by_rows(raw, "SIL")

    gc_panel = build_spread_panel(raw, "GC", gc_front, gc_second, TZ)
    sil_panel = build_spread_panel(raw, "SIL", sil_front, sil_second, TZ)
    gc_panel = gc_panel.drop(columns=["date"], errors="ignore")
    sil_panel = sil_panel.drop(columns=["date"], errors="ignore")
    panel = gc_panel.join(sil_panel, how="outer").sort_index()
    panel["date"] = panel.index.normalize()
    panel.to_parquet(OUT_DIR / "spreads_1m.parquet")

    Ns = [2, 3, 5]
    gc_d = trading_day_rolling_d(panel["GC_spread"], panel["date"], Ns)
    sil_d = trading_day_rolling_d(panel["SIL_spread"], panel["date"], Ns)

    qrows = []
    srows = []

    for name in ["GC_spread", "SIL_spread"]:
        x = panel[name].dropna()
        if x.empty:
            continue
        quantiles(x).to_frame("value").to_csv(OUT_DIR / f"{name}_quantiles.csv", index_label="q")
        srows.append({
            "series": name,
            "n": int(x.shape[0]),
            "mean": float(x.mean()),
            "std": float(x.std(ddof=1)) if x.shape[0] > 1 else np.nan,
            "min": float(x.min()),
            "max": float(x.max()),
            "median": float(x.median()),
        })

    for N in Ns:
        for root, dct in [("GC", gc_d), ("SIL", sil_d)]:
            nm = f"{root}_d_{N}D"
            x = dct[N].dropna()
            if x.empty:
                continue
            quantiles(x).to_frame("value").to_csv(OUT_DIR / f"{nm}_quantiles.csv", index_label="q")
            srows.append({
                "series": nm,
                "n": int(x.shape[0]),
                "mean": float(x.mean()),
                "std": float(x.std(ddof=1)) if x.shape[0] > 1 else np.nan,
                "min": float(x.min()),
                "max": float(x.max()),
                "median": float(x.median()),
            })

    pd.DataFrame(srows).to_csv(OUT_DIR / "summary_stats.csv", index=False)

    cor = {"corr_spreads": corr_pair(panel["GC_spread"], panel["SIL_spread"])}
    for N in Ns:
        cor[f"corr_d_{N}D"] = corr_pair(gc_d[N], sil_d[N])
    pd.DataFrame([cor]).to_csv(OUT_DIR / "correlations.csv", index=False)

    plt.figure()
    panel[["GC_spread", "SIL_spread"]].plot(ax=plt.gca())
    plt.xlabel("time")
    plt.ylabel("second - front")
    save_plot(OUT_DIR / "spreads_timeseries.png")

    plt.figure()
    panel["GC_spread"].dropna().hist(bins=200)
    plt.xlabel("GC spread")
    plt.ylabel("count")
    save_plot(OUT_DIR / "GC_spread_hist.png")

    plt.figure()
    panel["SIL_spread"].dropna().hist(bins=200)
    plt.xlabel("SIL spread")
    plt.ylabel("count")
    save_plot(OUT_DIR / "SIL_spread_hist.png")

    for N in Ns:
        plt.figure()
        gc_d[N].plot(ax=plt.gca())
        plt.xlabel("time")
        plt.ylabel(f"GC d ({N} trading-day mean)")
        save_plot(OUT_DIR / f"GC_d_{N}D_timeseries.png")

        plt.figure()
        sil_d[N].plot(ax=plt.gca())
        plt.xlabel("time")
        plt.ylabel(f"SIL d ({N} trading-day mean)")
        save_plot(OUT_DIR / f"SIL_d_{N}D_timeseries.png")

    gc_prof = intraday_profile(panel["GC_spread"])
    if not gc_prof.empty:
        plt.figure()
        gc_prof.plot(ax=plt.gca())
        plt.xlabel("minute of day (0=00:00)")
        plt.ylabel("avg GC spread")
        save_plot(OUT_DIR / "GC_spread_intraday_profile.png")

    sil_prof = intraday_profile(panel["SIL_spread"])
    if not sil_prof.empty:
        plt.figure()
        sil_prof.plot(ax=plt.gca())
        plt.xlabel("minute of day (0=00:00)")
        plt.ylabel("avg SIL spread")
        save_plot(OUT_DIR / "SIL_spread_intraday_profile.png")

    meta = {
        "pair": "GC vs SIL",
        "GC_front": gc_front.code,
        "GC_second": gc_second.code,
        "SIL_front": sil_front.code,
        "SIL_second": sil_second.code,
        "files_count": len(files),
        "tz": TZ,
        "data_dir": str(DATA_DIR),
        "out_dir": str(OUT_DIR),
        "Ns": Ns,
        "minute_grid": "active_span_union_of_legs_per_day",
        "ffill": "within_active_span_only; masked before first and after last observation",
        "d_definition": "minute spread minus N-trading-day rolling mean of daily average spread",
    }
    pd.Series(meta).to_json(OUT_DIR / "run_metadata.json", indent=2)


if __name__ == "__main__":
    run()
