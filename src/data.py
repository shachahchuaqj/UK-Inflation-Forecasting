from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "ukcpi.xls"


def load_inflation_data(path=DATA_PATH) -> pd.Series:
    """
    Load monthly UK CPI inflation data into columns [Year, Month, Inflation].
    Inflation data expressed as percentage change from the same month in the previous year.
    """

    df = pd.read_excel(path, header=None)
    df.columns = ["Year", "Month", "Inflation"]

    df["Date"] = pd.to_datetime(
        dict(
            year=df["Year"],
            month=df["Month"],
            day=1,
        )
    )

    df = df.sort_values("Date").set_index("Date")

    if df.index.has_duplicates:
        raise ValueError("Duplicate monthly observations")

    series = df["Inflation"].astype(float).asfreq("MS")

    if series.isna().any():
        raise ValueError("Missing monthly inflation observations")

    return series