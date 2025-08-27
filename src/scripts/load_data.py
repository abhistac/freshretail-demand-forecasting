# path shim so script works no matter where it's run from
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from datasets import load_dataset
import pandas as pd
from src.config import RAW_DIR, INTERIM_DIR, HF_DATASET, SAMPLE_ROWS

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        "Dingdong-Inc/FreshRetailNet-50K",
        cache_dir="data/hf_cache",
    )
    print(ds)  # confirmation

    # Tiny CSV head for sanity
    tiny = ds["train"].to_pandas().head(10)
    (RAW_DIR / "sample_head.csv").write_text(tiny.to_csv(index=False))
    print(f"Wrote {RAW_DIR/'sample_head.csv'}")

    # Sample a subset for EDA and save as parquet (fastparquet engine)
    df = ds["train"].to_pandas().sample(SAMPLE_ROWS, random_state=42)
    df["dt"] = pd.to_datetime(df["dt"])
    out = INTERIM_DIR / f"train_sample_{SAMPLE_ROWS}.parquet"
    df.to_parquet(out, index=False, engine="pyarrow")
    print(f"Saved sample parquet to {out}")

if __name__ == "__main__":
    main()