from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    # ── 1. Loading a Real Dataset ─────────────────────────────────────────────
    df = sns.load_dataset("titanic")

    # Alternatively, load any CSV the same way:
    # df = pd.read_csv("titanic.csv")

    # ── 2. Inspecting the Data ────────────────────────────────────────────────
    print("=== Shape ===")
    print(df.shape)  # (891, 15) — rows, columns

    print("\n=== First 5 Rows ===")
    print(df.head())

    print("\n=== Column Types ===")
    print(df.dtypes)

    print("\n=== Info (names + non-null counts + types) ===")
    df.info()

    print("\n=== Null Counts (sorted) ===")
    print(df.isnull().sum().sort_values(ascending=False))

    print("\n=== Null Percentages ===")
    print((df.isnull().sum() / len(df) * 100).round(1).sort_values(ascending=False))

    # ── 3. Handling Missing Values ────────────────────────────────────────────

    # deck is missing 77 % of values — preserve the signal, then drop the column
    df["deck_known"] = df["deck"].notnull().astype(int)
    df = df.drop(columns=["deck"])

    # Drop the 2 rows where `embarked` is null
    df = df.dropna(subset=["embarked"])

    # Fill missing `age` values with the median (robust to outliers)
    median_age = df["age"].median()
    df["age"] = df["age"].fillna(median_age)

    # Fill missing `embark_town` with the mode
    mode_embark_town = df["embark_town"].mode()[0]
    df["embark_town"] = df["embark_town"].fillna(mode_embark_town)

    print("\n=== Null Counts After Cleaning ===")
    print(df[["age", "embarked", "embark_town"]].isnull().sum())

    # ── 4. Fixing Data Types ──────────────────────────────────────────────────

    # String columns with a small number of distinct values → category
    for col in ["sex", "class", "embarked", "who"]:
        df[col] = df[col].astype("category")

    # survived is 0/1 → bool makes the intent explicit
    df["survived"] = df["survived"].astype(bool)

    # Parsing dates — Titanic doesn't have one, but the pattern is:
    # df["date"] = pd.to_datetime(df["date"])
    # df["year"]        = df["date"].dt.year
    # df["month"]       = df["date"].dt.month
    # df["day_of_week"] = df["date"].dt.day_name()

    print("\n=== Types After Fixes ===")
    print(df.dtypes)

    # ── 5. Removing Duplicates ────────────────────────────────────────────────
    print("\n=== Duplicate Row Count ===")
    print(df.duplicated().sum())

    # View all copies of any duplicated rows
    print(df[df.duplicated(keep=False)])

    # Drop duplicates (no-op here, but the pattern is always the same)
    df = df.drop_duplicates()

    # ── 6. Exploratory Data Analysis ─────────────────────────────────────────

    # Five-number summary for numeric columns
    print("\n=== describe() ===")
    print(df.describe())

    # Include categorical columns too
    print("\n=== describe(include='all') ===")
    print(df.describe(include="all"))

    # Distribution of categorical columns
    print("\n=== Sex Distribution ===")
    print(df["sex"].value_counts())

    print("\n=== Class Distribution ===")
    print(df["class"].value_counts())

    # Proportions (great for spotting class imbalance)
    print("\n=== Survival Rate (proportion) ===")
    print(df["survived"].value_counts(normalize=True).round(2))

    # groupby reveals how a metric varies across categories
    print("\n=== Survival Rate by Sex ===")
    print(df.groupby("sex")["survived"].mean().round(2))

    print("\n=== Survival Rate by Class ===")
    print(df.groupby("class")["survived"].mean().round(2))

    # Cross both dimensions at once
    print("\n=== Survival Rate by Class × Sex ===")
    print(df.groupby(["class", "sex"])["survived"].mean().round(2).unstack())

    # Correlation between numeric columns
    print("\n=== Correlation Matrix ===")
    print(df[["age", "fare", "pclass", "sibsp", "parch"]].corr().round(2))

    # ── 7. Visualizing with Matplotlib ───────────────────────────────────────

    # Distribution of a numeric column
    df["age"].hist(bins=20, edgecolor="black")
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "age_distribution.png")
    plt.show()

    # Bar chart from value_counts
    df["class"].value_counts().plot(kind="bar", edgecolor="black")
    plt.title("Passengers by Class")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution.png")
    plt.show()

    # Survival rate by class
    survival_by_class = df.groupby("class")["survived"].mean()
    survival_by_class.plot(kind="bar", edgecolor="black")
    plt.title("Survival Rate by Class")
    plt.xlabel("Class")
    plt.ylabel("Survival Rate")
    plt.xticks(rotation=0)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "survival_by_class.png")
    plt.show()

    # Scatter plot — alpha=0.4 keeps overlapping points visible
    plt.scatter(df["age"], df["fare"], alpha=0.4)
    plt.title("Age vs Fare")
    plt.xlabel("Age")
    plt.ylabel("Fare")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "age_vs_fare.png")
    plt.show()

    # Side-by-side subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    df["age"].hist(bins=20, ax=axes[0], edgecolor="black")
    axes[0].set_title("Age Distribution")

    df["fare"].hist(bins=30, ax=axes[1], edgecolor="black")
    axes[1].set_title("Fare Distribution")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "distributions.png")
    plt.show()


if __name__ == "__main__":
    main()
