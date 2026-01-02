from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    to_date,
    year,
    regexp_replace,
    when,
    count,
    avg,
    round as spark_round,
)
import kagglehub
import os
import shutil
from datetime import datetime
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATASET_NAME = "artermiloff/steam-games-dataset"
TARGET_FILENAME = "games_march2025_full.csv"
TARGET_FILENAME_PARQUET = "games_march2025_full.parquet"

# Output directory inside container (bind-mounted to host)
OUTPUT_DIR = "/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data directory inside container (bind-mounted to host)
DATA_DIR = "/data"
os.makedirs(DATA_DIR, exist_ok=True)


def format_summary(df, value_col):
    rows = df.collect()
    lines = []
    for r in rows:
        label = r["summary"]
        value = r[value_col]
        lines.append(f"{label}: {value}")
    return "\n".join(lines)


def main():
    started = datetime.now()
    started_str = f"Spark analysis started: " + f"{started.strftime('%Y-%m-%d %H:%M:%S')}\n"
    spark = SparkSession.builder.appName("Spark_Analysis_Application").getOrCreate()

    # Βήμα 1: Φόρτωση Dataset (CSV ή ήδη Parquet) και μετατροπή σε Parquet αν χρειάζεται
    print("Φόρτωση του dataset...")
    path = ensure_dataset_file()

    try:
        df_raw = spark.read.parquet(path)
        print("Το αρχείο .parquet φορτώθηκε επιτυχώς!")
    except Exception:
        print("Το αρχείο .parquet δεν βρέθηκε – γίνεται μετατροπή από CSV σε Parquet...")
        df_csv = spark.read.csv(path, header=True, inferSchema=True)
        path_parquet = os.path.join(DATA_DIR, TARGET_FILENAME_PARQUET)
        df_csv.write.mode("overwrite").parquet(path_parquet)
        print("Η μετατροπή σε Parquet έγινε επιτυχώς")
        df_raw = spark.read.parquet(path_parquet)

    # Βήμα 2: Επισκόπηση και Βαθμίδα Καθαρισμού με Spark DataFrame API
    print("\n--- ΕΠΙΣΚΟΠΗΣΗ ΔΕΔΟΜΕΝΩΝ ---")
    row_count = df_raw.count()
    col_count = len(df_raw.columns)
    print(f"Διαστάσεις Dataset: {row_count} γραμμές, {col_count} στήλες")

    # Επιλογή σχετικών στηλών για ανάλυση & μετατροπές τύπων
    df = df_raw.select(
        "name",
        regexp_replace(col("price"), "[^0-9.]", "").cast("double").alias("price"),
        regexp_replace(col("user_score"), "[^0-9.]", "").cast("double").alias("user_score"),
        regexp_replace(col("metacritic_score"), "[^0-9.]", "").cast("double").alias("metacritic_score"),
        "positive",
        "negative",
        "pct_pos_total",
        "categories",
        "genres",
        "tags",
        to_date(col("release_date"), "yyyy-MM-dd").alias("release_date"),
    ).withColumn("release_year", year(col("release_date")))

    print(f"\nΧρησιμοποιούμε {len(df.columns)} στήλες: {df.columns}")

    # Υπολογισμός total_reviews & positive_ratio (μόνο με Spark)
    if "positive" in df.columns and "negative" in df.columns:
        df = df.withColumn("total_reviews", col("positive") + col("negative"))
        df = df.withColumn(
            "positive_ratio",
            when(col("total_reviews") > 0, (col("positive") / col("total_reviews") * 100)).otherwise(None),
        )

    df.printSchema()

    # Αποθήκευση καθαρισμένου dataset σε μορφή Parquet (για Spark)
    output_parquet_path = os.path.join(OUTPUT_DIR, "cleaned_games.parquet")
    print(f"\nΑποθήκευση Parquet στο {output_parquet_path} ...")
    df.write.mode("overwrite").parquet(output_parquet_path)

    # Αποθήκευση και σε CSV για "ανθρώπινη" ανάγνωση (π.χ. Excel)
    csv_temp_dir = os.path.join(OUTPUT_DIR, "cleaned_games_csv_tmp")
    final_csv_path = os.path.join(OUTPUT_DIR, "cleaned_games.csv")

    print(f"Αποθήκευση CSV προσωρινά στο {csv_temp_dir} ...")
    (
        df.coalesce(1)
        .write.mode("overwrite")
        .option("header", True)
        .csv(csv_temp_dir)
    )

    csv_files = [
        f for f in os.listdir(csv_temp_dir)
        if f.startswith("part-") and f.endswith(".csv")
    ]
    if csv_files:
        src = os.path.join(csv_temp_dir, csv_files[0])
        shutil.move(src, final_csv_path)
        for f in os.listdir(csv_temp_dir):
            os.remove(os.path.join(csv_temp_dir, f))
        os.rmdir(csv_temp_dir)
        print(f"Η CSV έξοδος αποθηκεύτηκε στο {final_csv_path} .")
    else:
        print("Προειδοποίηση: Δεν βρέθηκε part-*.csv στον προσωρινό φάκελο.")

    # Βήμα 3: Βασικά Analytics με Spark
    print("\n--- ΒΑΣΙΚΑ ΑΝΑΛΥΤΙΚΑ ΣΤΟΙΧΕΙΑ (SPARK) ---")

    # 1. Καταμέτρηση παιχνιδιών ανά έτος κυκλοφορίας (Top 10)
    games_per_year_df = (
        df.where(col("release_year").isNotNull())
        .groupBy("release_year")
        .agg(count("name").alias("game_count"))
        .orderBy("release_year")
    )
    games_per_year_rows = games_per_year_df.collect()
    games_per_year_last10 = games_per_year_rows[-10:]
    print("\n1. Καταμέτρηση παιχνιδιών ανά έτος κυκλοφορίας (Top 10):")
    for r in games_per_year_last10:
        print(f"{r['release_year']}: {r['game_count']}")

    # 2. Περιγραφική στατιστική για τιμές
    print("\n2. Περιγραφική στατιστική για τιμές (price):")
    # Χρήση όλων των τιμών για max, αλλά αγνόηση ακραίων τιμών (>1000) για τον μέσο όρο
    price_stats_df = df.select("price").summary()

    # Στατιστικά μόνο για τιμές <= 1000 (για τον μέσο όρο χωρίς ακραίες τιμές)
    price_stats_filtered_df = (
        df.where((col("price").isNotNull()) & (col("price") <= 1000))
        .select("price")
        .summary()
    )

    # Συνδυασμός: κρατάμε τον mean από το φιλτραρισμένο DF, τα υπόλοιπα από το πλήρες
    full_rows = {r["summary"]: r for r in price_stats_df.collect()}
    filtered_rows = {r["summary"]: r for r in price_stats_filtered_df.collect()}

    ordered_summaries = [
        "count",
        "mean",
        "stddev",
        "min",
        "25%",
        "50%",
        "75%",
        "max",
    ]

    lines = []
    for key in ordered_summaries:
        row_dict = filtered_rows if key == "mean" and key in filtered_rows else full_rows
        if key in row_dict:
            value = row_dict[key]["price"]
            lines.append(f"{key}: {value}")

    price_stats_str = "\n".join(lines)
    print(price_stats_str)

    # 3. Περιγραφική στατιστική για βαθμολογίες
    print("\n3. Περιγραφική στατιστική για user_score:")
    # Υπολογισμός περιγραφικών στατιστικών μόνο για "πραγματικές" user scores (> 0)
    user_score_stats_df = (
        df.where(col("user_score") > 0)
        .select("user_score")
        .summary()
    )
    user_score_stats_str = format_summary(user_score_stats_df, "user_score")
    print(user_score_stats_str)

    print("\n4. Περιγραφική στατιστική για metacritic_score:")
    # Υπολογισμός περιγραφικών στατιστικών μόνο για "πραγματικά" metacritic scores (> 0)
    metacritic_stats_df = (
        df.where(col("metacritic_score") > 0)
        .select("metacritic_score")
        .summary()
    )
    metacritic_stats_str = format_summary(metacritic_stats_df, "metacritic_score")
    print(metacritic_stats_str)

    # 5. Μέση τιμή και μέση βαθμολογία ανά έτος κυκλοφορίας (για έτη με >30 παιχνίδια)
    print("\n5. Μέση τιμή και μέση βαθμολογία ανά έτος κυκλοφορίας (για έτη με >30 παιχνίδια):")

    yearly_stats_df = (
        df.where(col("release_year").isNotNull())
        .groupBy("release_year")
        .agg(
            count("name").alias("game_count"),
            avg("price").alias("avg_price"),
            avg("user_score").alias("avg_user_score"),
            avg("metacritic_score").alias("avg_metacritic"),
            avg("pct_pos_total").alias("avg_pct_pos"),
        )
    )

    yearly_stats_df = (
        yearly_stats_df
        .withColumn("avg_price", spark_round(col("avg_price"), 2))
        .withColumn("avg_user_score", spark_round(col("avg_user_score"), 2))
        .withColumn("avg_metacritic", spark_round(col("avg_metacritic"), 2))
        .withColumn("avg_pct_pos", spark_round(col("avg_pct_pos"), 2))
        .where(col("game_count") > 30)
        .orderBy("release_year")
    )

    yearly_stats_rows = yearly_stats_df.collect()
    yearly_stats_last15 = yearly_stats_rows[-15:]
    for r in yearly_stats_last15:
        print(
            f"{r['release_year']}: count={r['game_count']}, "
            f"avg_price={r['avg_price']}, "
            f"avg_user_score={r['avg_user_score']}, "
            f"avg_metacritic={r['avg_metacritic']}, "
            f"avg_pct_pos={r['avg_pct_pos']}"
        )

    # Βήμα 4: Δημιουργία Γραφημάτων (με δεδομένα από Spark)
    print("\n--- ΔΗΜΙΟΥΡΓΙΑ ΓΡΑΦΗΜΑΤΩΝ (SPARK) ---")

    # Γράφημα 1: Μέση τιμή ανά έτος (έτη με >30 παιχνίδια)
    if yearly_stats_last15:
        years = [int(r["release_year"]) for r in yearly_stats_last15 if r["avg_price"] is not None]
        avg_prices = [float(r["avg_price"]) for r in yearly_stats_last15 if r["avg_price"] is not None]
        if years and avg_prices:
            plt.figure(figsize=(12, 6))
            ax = plt.gca()
            ax.bar(range(len(years)), avg_prices, color="skyblue")
            ax.set_title(
                "Μέση Τιμή Παιχνιδιών Steam ανά Έτος Κυκλοφορίας\n(Για έτη με πάνω από 30 παιχνίδια)",
                fontsize=14,
            )
            ax.set_xlabel("Έτος Κυκλοφορίας")
            ax.set_ylabel("Μέση Τιμή (USD)")
            ax.set_xticks(range(len(years)))
            ax.set_xticklabels(years, rotation=45)
            for i, v in enumerate(avg_prices):
                ax.text(i, v + 0.5, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
            plt.tight_layout()
            out_path = os.path.join(OUTPUT_DIR, "avg_price_per_year.png")
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"Το γράφημα 'avg_price_per_year.png' αποθηκεύτηκε στο {out_path}.")
        else:
            print("Δεν υπάρχουν επαρκή δεδομένα για το γράφημα τιμών ανά έτος")
    else:
        print("Δεν υπάρχουν δεδομένα yearly_stats για το γράφημα τιμών ανά έτος")

    # Γράφημα 2: Σύγκριση user_score vs metacritic_score
    scores_df = df.select("metacritic_score", "user_score").where(
        col("metacritic_score").isNotNull() & col("user_score").isNotNull()
    )
    scores_count = scores_df.count()
    if scores_count > 10:
        scores_sample = scores_df.limit(5000)
        rows = scores_sample.collect()
        x = [float(r["metacritic_score"]) for r in rows]
        y = [float(r["user_score"]) for r in rows]

        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, alpha=0.5)
        plt.title(
            "Σύγκριση Βαθμολογιών: User Score vs Metacritic Score\n(Γνώμη Παικτών vs Γνώμη Κριτικών)",
            fontsize=14,
        )
        plt.xlabel("Metacritic Score (Κριτικοί)")
        plt.ylabel("User Score (Παίκτες)")

        if len(x) > 1:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            xs_line = np.linspace(min(x), max(x), 100)
            plt.plot(xs_line, p(xs_line), "r--", alpha=0.8)

        correlation = scores_df.stat.corr("metacritic_score", "user_score")
        plt.text(
            0.05,
            0.95,
            f"Συσχέτιση: {correlation:.3f}",
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, "user_vs_metacritic_score.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Το γράφημα 'user_vs_metacritic_score.png' αποθηκεύτηκε στο {out_path}.")
    else:
        print("Δεν υπάρχουν αρκετά δεδομένα για σύγκριση user_score vs metacritic_score")

    # Γράφημα 3: Κατανομή Τιμών (Παλιά vs Καινούρια)
    df_era = df.withColumn(
        "era",
        when(col("release_year") <= 2014, "Παλιά (μέχρι 2014)")
        .when(col("release_year") > 2014, "Καινούρια (2015+)")
    )

    price_data = df_era.where(
        (col("price").isNotNull())
        & (col("price") >= 0)
        & (col("price") <= 100)
        & col("era").isNotNull()
    )

    old_rows = price_data.where(col("era") == "Παλιά (μέχρι 2014)").select("price").collect()
    new_rows = price_data.where(col("era") == "Καινούρια (2015+)").select("price").collect()

    old_prices = [float(r["price"]) for r in old_rows]
    new_prices = [float(r["price"]) for r in new_rows]

    if old_prices and new_prices:
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        box = ax.boxplot([old_prices, new_prices], labels=["Παλιά (μέχρι 2014)", "Καινούρια (2015+)"])
        ax.set_title(
            "Σύγκριση Κατανομής Τιμών για Παλιά vs Καινούρια Παιχνίδια Steam\n(Παλιά = μέχρι 2014, Καινούρια = 2015 και μετά)",
            fontsize=14,
        )
        ax.set_xlabel("Εποχή Παιχνιδιού")
        ax.set_ylabel("Τιμή (USD)")

        old_mean = float(np.mean(old_prices))
        old_median = float(np.median(old_prices))
        new_mean = float(np.mean(new_prices))
        new_median = float(np.median(new_prices))

        stats_text = (
            f"Παλιά: Μέση τιμή = ${old_mean:.2f}, Διάμεσος = ${old_median:.2f}\n"
            f"Καινούρια: Μέση τιμή = ${new_mean:.2f}, Διάμεσος = ${new_median:.2f}"
        )

        plt.text(
            0.02,
            0.98,
            stats_text,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, "price_comparison_old_vs_new.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(
            "Το γράφημα 'price_comparison_old_vs_new.png' αποθηκεύτηκε στο "
            f"{out_path}."
        )
    else:
        print("Δεν υπάρχουν αρκετά δεδομένα για boxplot τιμών")

    # Βήμα 5: Εγγραφή Αποτελεσμάτων σε Αρχείο .txt (Spark έκδοση)
    print("\n--- ΕΓΓΡΑΦΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ (SPARK) ---")
    output_filename = os.path.join(OUTPUT_DIR, "analysis_results.txt")

    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write("STEAM DATASET ANALYSIS RESULTS (March 2025) - SPARK/PARQUET\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Total number of games in the dataset: {row_count}\n")
            f.write(f"Number of columns used: {len(df.columns)}\n")
            f.write(f"Columns: {', '.join(df.columns)}\n\n")

            f.write("1. GAME COUNT PER RELEASE YEAR (Top 10)\n")
            f.write("-" * 50 + "\n")
            for r in games_per_year_last10:
                f.write(f"{r['release_year']}: {r['game_count']}\n")
            f.write("\n")

            f.write("2. BASIC PRICE STATISTICS (price) [Spark summary()]\n")
            f.write("-" * 50 + "\n")
            f.write(price_stats_str + "\n\n")

            f.write("3. SCORE STATISTICS (Spark summary())\n")
            f.write("-" * 50 + "\n")
            f.write("User Score (Players' Opinion):\n")
            f.write(user_score_stats_str + "\n\n")
            f.write("Metacritic Score (Critics' Opinion):\n")
            f.write(metacritic_stats_str + "\n\n")

            f.write("4. PRICE AND SCORE SUMMARY PER YEAR (Spark)\n")
            f.write("-" * 50 + "\n")
            for r in yearly_stats_last15:
                f.write(
                    f"{r['release_year']}: count={r['game_count']}, "
                    f"avg_price={r['avg_price']}, avg_user_score={r['avg_user_score']}, "
                    f"avg_metacritic={r['avg_metacritic']}, avg_pct_pos={r['avg_pct_pos']}\n"
                )
            f.write("\n")

            f.write("5. KEY INSIGHTS (based on Spark analysis)\n")
            f.write("-" * 50 + "\n")
            f.write("- Most Steam games have been released in recent years.\n")

            years_non_null = [
                int(r["release_year"])
                for r in games_per_year_rows
                if r["release_year"] is not None
            ]
            if years_non_null:
                min_year = min(years_non_null)
                max_year = max(years_non_null)
                f.write(
                    f"- The dataset covers the period from {min_year} to {max_year}.\n"
                )

            if games_per_year_rows:
                peak_row = max(
                    games_per_year_rows,
                    key=lambda r: r["game_count"] if r["game_count"] is not None else 0,
                )
                f.write(
                    f"- The year with the most game releases is {peak_row['release_year']} "
                    f"with {peak_row['game_count']} games.\n"
                )

            # Υπολογισμός μέσης και διάμεσης τιμής αγνοώντας ακραίες τιμές (>1000)
            df_price_no_outliers = df.where((col("price").isNotNull()) & (col("price") <= 1000))

            avg_price_overall = df_price_no_outliers.select(
                avg("price").alias("avg_price")
            ).collect()[0]["avg_price"]

            median_approx = df_price_no_outliers.approxQuantile("price", [0.5], 0.01)[0]
            f.write(
                f"- The average game price (Spark) is {avg_price_overall:.2f} USD "
                f"and the approximate median price is {median_approx:.2f} USD.\n"
            )

            if scores_count > 10:
                correlation = scores_df.stat.corr("metacritic_score", "user_score")
                f.write(
                    f"- The correlation between user score (players) and metacritic score "
                    f"(critics) is {correlation:.3f}.\n"
                )

            ended = datetime.now()

            f.write(started_str)
            f.write(
                f"Spark analysis completed: "
                f"{ended.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(
                f"Total execution time: {ended - started}"
            )

    except Exception as e:
        print(f"Σφάλμα κατά την εγγραφή του αρχείου: {e}")

    print("\n" + "=" * 50)
    print("Spark analysis was completed successfully!")
    print("=" * 50)

    spark.stop()






def ensure_dataset_file():
    """Use the mounted /data folder as the persistent cache for the Kaggle CSV.

    If TARGET_FILENAME exists in DATA_DIR, reuse it.
    Otherwise download the dataset once via KaggleHub and copy the first CSV
    found in the downloaded folder into DATA_DIR under TARGET_FILENAME.
    """

    local_path = os.path.join(DATA_DIR, TARGET_FILENAME)

    local_path_parquet = os.path.join(DATA_DIR,TARGET_FILENAME_PARQUET)

    # 1) If the file is already in the mounted /data folder, just reuse it
    if os.path.exists(local_path_parquet):
        print(f"Βρέθηκε ήδη τοπικό αρχείο dataset στο {local_path_parquet}")
        return local_path_parquet

    if os.path.exists(local_path):
        print(f"Βρέθηκε ήδη τοπικό αρχείο dataset στο {local_path}")
        return local_path

    # 2) Otherwise, download (or reuse KaggleHub cache) and copy once into /data
    print("Το αρχείο δεν βρέθηκε στο /data. Γίνεται λήψη από KaggleHub...")
    try:
        dataset_path = kagglehub.dataset_download(DATASET_NAME)
        print("Path to dataset files:", dataset_path)
    except Exception as exc:
        print(f"Αποτυχία λήψης από KaggleHub: {exc}")
        raise

    # take the first CSV directly under dataset_path
    csv_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"Δεν βρέθηκαν CSV αρχεία στον φάκελο: {dataset_path}")

    src = os.path.join(dataset_path, csv_files[0])
    shutil.copy(src, local_path)
    print(f"Αντιγράφηκε το αρχείο {os.path.basename(src)} στο {local_path} (mounted /data)")

    return local_path




if __name__ == "__main__":
    main()