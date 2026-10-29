"""Small command-line EDA summary for the Telco churn CSV."""

from pathlib import Path

from churn_project.modeling import TARGET, load_and_clean_data

DATA_PATH = Path(__file__).resolve().parent / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def main() -> None:
    data = load_and_clean_data(DATA_PATH)
    print(f"Rows: {len(data):,}")
    print(f"Overall churn rate: {data[TARGET].mean():.1%}")
    for column in ("Contract", "InternetService", "PaymentMethod"):
        summary = data.groupby(column, observed=True)[TARGET].agg(
            customers="size", churn_rate="mean"
        )
        summary["churn_rate"] = summary["churn_rate"].map(lambda value: f"{value:.1%}")
        print(f"\nChurn by {column}:\n{summary.to_string()}")


if __name__ == "__main__":
    main()

\n