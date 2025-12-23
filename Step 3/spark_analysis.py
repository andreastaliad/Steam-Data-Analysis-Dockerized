from pyspark.sql import SparkSession
import kagglehub
import os
import shutil
from pyspark.sql.functions import col, to_date, year, regexp_replace, when



DATASET_NAME = "artermiloff/steam-games-dataset"
TARGET_FILENAME = "games_march2025_full.csv"
TARGET_FILENAME_PARQUET = "games_march2025_full.parquet"

# Output directory inside container (bind-mounted to host)
OUTPUT_DIR = "/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data directory inside container (bind-mounted to host)
DATA_DIR = "/data"
os.makedirs(DATA_DIR, exist_ok=True)



def main():
    spark = SparkSession.builder.appName("Spark_Analysis_Application").getOrCreate()

    # Βήμα 1: Φόρτωση Dataset
    print("Φόρτωση του dataset...")
    path = ensure_dataset_file()
    parquet = None
    try:
        parquet = spark.read.parquet(path)
        print("Το αρχείο .parquet φορτώθηκε επίτυχως!")
    except Exception as e:
        print("Το αρχείο .parquet δεν βρέθηκε γίνεται μετατροπή απο csv...")
        # header = True parameter to treat the first row of the CSV as a header 
        # inferSchema = True parameter to infer the data types of the columns
        df = spark.read.csv(path, header=True, inferSchema=True)

        # Create the parquet destination directory and name
        pathParquet = os.path.join(DATA_DIR, TARGET_FILENAME_PARQUET)

        # Create the parquet file
        df.write.parquet(pathParquet)
        print("Η μετατροπή σε Parquet έγινε επιτυχώς")
        parquet = spark.read.parquet(pathParquet)

    # Βημα 2: Επισκόπηση και Βαθμίδα Καθαρισμού
    # Print the shape of the parquet data frame
    print("\n--- ΕΠΙΣΚΟΠΗΣΗ ΔΕΔΟΜΕΝΩΝ ---")  
    print(f"Shape: {parquet.count()} γραμμες, {len(parquet.columns)} στυλες")

    # Επιλογή σχετικών στηλών για ανάλυση (με βάση τις ΠΡΑΓΜΑΤΙΚΕΣ στήλες)

    parquet = parquet.select(
        'name', 
        # Επεξεργασία τιμών: μετατροπή 'price' σε αριθμητικό
        regexp_replace(col('price'), '[^0-9.]', '').cast('double').alias('price'),  # Βασικά στοιχεία (price_final -> price)
        # Επεξεργασία βαθμολογιών: μετατροπή σε αριθμητικές
        regexp_replace(col('user_score'), '[^0-9.]', '').cast('int').alias('user_score'),
        regexp_replace(col('metacritic_score'), '[^0-9.]', '').cast('int').alias('metacritic_score'),  # Βαθμολογίες (rating -> user_score/metacritic_score)
        'positive', 'negative', 'pct_pos_total',  # Κριτικές χρηστών
        'categories', 'genres', 'tags' ,        # Κατηγορίες
        to_date(col('release_date'),'yyyy-MM-dd').alias('release_date') # Convert to datetime using spark functions
    ).withColumn('release_year',year(col('release_date')))

    print(f"\nΧρησιμοποιούμε {len(parquet.columns)} στήλες: {parquet.columns}")

    #1. Check if columns exist
    if 'positive' in parquet.columns and 'negative' in parquet.columns:
        #2. Add total_reviews
        parquet = parquet.withColumn('total_reviews', col('positive')+col('negative'))

        #3 Calculate ratio 
        parquet = parquet.withColumn('positive_ratio',when(col('total_reviews') > 0, (col('positive')/col('total_reviews')*100)).otherwise(None))

    parquet.printSchema()

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