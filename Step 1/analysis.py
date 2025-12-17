import os
import glob
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np  
import kagglehub


DATASET_NAME = "artermiloff/steam-games-dataset"
TARGET_FILENAME = "games_march2025_full.csv"


def ensure_dataset_file():
    """Ensure the target CSV exists locally; download from KaggleHub if missing."""
    local_path = os.path.join(os.getcwd(), TARGET_FILENAME)
    if os.path.exists(local_path):
        return local_path

    print("Το αρχείο δεν βρέθηκε τοπικά. Γίνεται λήψη από KaggleHub...")
    try:
        dataset_path = kagglehub.dataset_download(DATASET_NAME)
        print("Path to dataset files:", dataset_path)
    except Exception as exc:
        print(f"Αποτυχία λήψης από KaggleHub: {exc}")
        return local_path

    candidate = os.path.join(dataset_path, TARGET_FILENAME)
    if os.path.exists(candidate):
        shutil.copy(candidate, local_path)
        print(f"Αντιγράφηκε το αρχείο από KaggleHub στο {local_path}")
        return local_path

    csv_files = glob.glob(os.path.join(dataset_path, "*.csv"))
    if csv_files:
        shutil.copy(csv_files[0], local_path)
        print(f"Χρήση αρχείου {os.path.basename(csv_files[0])} από KaggleHub")
        return local_path

    print("Δεν βρέθηκε κατάλληλο CSV στο KaggleHub download.")
    return local_path

# Βήμα 1: Φόρτωση Dataset
print("Φόρτωση του dataset...")
csv_path = ensure_dataset_file()
try:
    df = pd.read_csv(csv_path, low_memory=False)
    print("Το dataset φορτώθηκε επιτυχώς!")
except FileNotFoundError:
    print("Σφάλμα: Το αρχείο 'games_march2025_full.csv' δεν βρέθηκε στον τρέχοντα φάκελο.")
    print("Βεβαιώσου ότι το έχεις κατεβάσει από το Kaggle και ότι βρίσκεται στο σωστό path.")
    exit()

# Βήμα 2: Επισκόπηση και Βαθμίδα Καθαρισμού
print("\n--- ΕΠΙΣΚΟΠΗΣΗ ΔΕΔΟΜΕΝΩΝ ---")
print(f"Διαστάσεις Dataset: {df.shape[0]} γραμμές, {df.shape[1]} στήλες")

# Επιλογή σχετικών στηλών για ανάλυση (με βάση τις ΠΡΑΓΜΑΤΙΚΕΣ στήλες)
columns_of_interest = [
    'name', 'release_date', 'price',  # Βασικά στοιχεία (price_final -> price)
    'user_score', 'metacritic_score',  # Βαθμολογίες (rating -> user_score/metacritic_score)
    'positive', 'negative', 'pct_pos_total',  # Κριτικές χρηστών
    'categories', 'genres', 'tags'  # Κατηγορίες
]

# Επιλέγω μόνο τις στήλες που υπάρχουν στο DataFrame
existing_columns = [col for col in columns_of_interest if col in df.columns]
df = df[existing_columns].copy()

print(f"\nΧρησιμοποιούμε {len(existing_columns)} στήλες: {existing_columns}")

# Μετατροπή της στήλης 'release_date' σε datetime
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_year'] = df['release_date'].dt.year

# Επεξεργασία τιμών: μετατροπή 'price' σε αριθμητικό
if 'price' in df.columns:
    df['price'] = pd.to_numeric(df['price'], errors='coerce')

# Επεξεργασία βαθμολογιών: μετατροπή σε αριθμητικές
if 'user_score' in df.columns:
    df['user_score'] = pd.to_numeric(df['user_score'], errors='coerce')
if 'metacritic_score' in df.columns:
    df['metacritic_score'] = pd.to_numeric(df['metacritic_score'], errors='coerce')

# Υπολογισμός ποσοστού θετικών κριτικών (αν δεν υπάρχει ήδη η στήλη pct_pos_total)
if all(col in df.columns for col in ['positive', 'negative']):
    df['total_reviews'] = df['positive'] + df['negative']
    # Υπολογισμός ποσοστού θετικών κριτικών, αποφεύγοντας διαίρεση με το μηδέν
    df['positive_ratio_calc'] = df.apply(
        lambda row: (row['positive'] / row['total_reviews']) * 100 if row['total_reviews'] > 0 else None,
        axis=1
    )

# Βήμα 3: Βασικά Analytics
print("\n--- ΒΑΣΙΚΑ ΑΝΑΛΥΤΙΚΑ ΣΤΟΙΧΕΙΑ (ANALYTICS) ---")

# 1. Καταμέτρηση παιχνιδιών ανά έτος κυκλοφορίας (Top 10)
print("\n1. Καταμέτρηση παιχνιδιών ανά έτος κυκλοφορίας (Top 10):")
if 'release_year' in df.columns:
    games_per_year = df['release_year'].value_counts().sort_index().tail(10)
    print(games_per_year)
else:
    print("Δεν υπάρχει στήλη release_year")

# 2. Βασική στατιστική για τις τιμές
print("\n2. Περιγραφική στατιστική για τιμές (price):")
if 'price' in df.columns:
    price_stats = df['price'].describe()
    print(price_stats)
else:
    print("Δεν υπάρχει στήλη price")

# 3. Βασική στατιστική για βαθμολογίες
print("\n3. Περιγραφική στατιστική για user_score:")
if 'user_score' in df.columns:
    user_score_stats = df['user_score'].describe()
    print(user_score_stats)

print("\n4. Περιγραφική στατιστική για metacritic_score:")
if 'metacritic_score' in df.columns:
    metacritic_stats = df['metacritic_score'].describe()
    print(metacritic_stats)

# 4. Σύνδεση τιμών και βαθμολογιών ανά έτος
print("\n5. Μέση τιμή και μέση βαθμολογία ανά έτος κυκλοφορίας (για έτη με >30 παιχνίδια):")
stats_cols = {}
if 'price' in df.columns:
    stats_cols['avg_price'] = ('price', 'mean')
if 'user_score' in df.columns:
    stats_cols['avg_user_score'] = ('user_score', 'mean')
if 'metacritic_score' in df.columns:
    stats_cols['avg_metacritic'] = ('metacritic_score', 'mean')
if 'pct_pos_total' in df.columns:
    stats_cols['avg_pct_pos'] = ('pct_pos_total', 'mean')

if 'release_year' in df.columns and stats_cols:
    yearly_stats = df.groupby('release_year').agg(
        game_count=('name', 'count'),
        **stats_cols
    ).round(2)
    # Φιλτράρουμε για έτη με επαρκή δεδομένα
    yearly_stats_filtered = yearly_stats[yearly_stats['game_count'] > 30].tail(15)
    print(yearly_stats_filtered)
else:
    yearly_stats_filtered = None
    print("Δεν υπάρχουν αρκετές στήλες για να γίνει ομαδοποίηση")

# Βήμα 4: Δημιουργία Γραφημάτων
print("\n--- ΔΗΜΙΟΥΡΓΙΑ ΓΡΑΦΗΜΑΤΩΝ ---")

# Γράφημα 1: Μέση τιμή ανά έτος
if yearly_stats_filtered is not None and 'avg_price' in yearly_stats_filtered.columns:
    plt.figure(figsize=(12, 6))
    ax = yearly_stats_filtered['avg_price'].plot(kind='bar', color='skyblue')
    plt.title('Μέση Τιμή Παιχνιδιών Steam ανά Έτος Κυκλοφορίας\n(Για έτη με πάνω από 30 παιχνίδια)', fontsize=14)
    plt.xlabel('Έτος Κυκλοφορίας')
    plt.ylabel('Μέση Τιμή (USD)')
    plt.xticks(rotation=45)
    # Προσθήκη των τιμών πάνω από τα bars
    for i, v in enumerate(yearly_stats_filtered['avg_price']):
        if not pd.isna(v):
            ax.text(i, v + 0.5, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig('avg_price_per_year.png', dpi=150)
    print("Το γράφημα 'avg_price_per_year.png' αποθηκεύτηκε.")
else:
    print("Δεν υπάρχουν δεδομένα για το γράφημα τιμών ανά έτος")

# Γράφημα 2: Σύγκριση βαθμολογιών user_score vs metacritic_score (αν υπάρχουν)
if all(col in df.columns for col in ['user_score', 'metacritic_score']):
    # Φιλτράρουμε τις γραμμές που έχουν και τις δύο βαθμολογίες
    scores_df = df[['user_score', 'metacritic_score']].dropna()
    if len(scores_df) > 10:
        plt.figure(figsize=(10, 6))
        plt.scatter(scores_df['metacritic_score'], scores_df['user_score'], alpha=0.5)
        plt.title('Σύγκριση Βαθμολογιών: User Score vs Metacritic Score\n(Γνώμη Παικτών vs Γνώμη Κριτικών)', fontsize=14)
        plt.xlabel('Metacritic Score (Κριτικοί)')
        plt.ylabel('User Score (Παίκτες)')
        
        # Προσθήκη γραμμής συσχέτισης
        z = np.polyfit(scores_df['metacritic_score'], scores_df['user_score'], 1)
        p = np.poly1d(z)
        plt.plot(scores_df['metacritic_score'], p(scores_df['metacritic_score']), "r--", alpha=0.8)
        
        # Προσθήκη συντελεστή συσχέτισης
        correlation = scores_df['user_score'].corr(scores_df['metacritic_score'])
        plt.text(0.05, 0.95, f'Συσχέτιση: {correlation:.3f}', transform=plt.gca().transAxes, 
                fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('user_vs_metacritic_score.png', dpi=150)
        print("Το γράφημα 'user_vs_metacritic_score.png' αποθηκεύτηκε.")
    else:
        print("Δεν υπάρχουν αρκετά δεδομένα για σύγκριση user_score vs metacritic_score")
else:
    print("Δεν υπάρχουν και οι δύο στήλες για σύγκριση user_score vs metacritic_score")

# Γράφημα 3: Κατανομή Τιμών (Παλιά vs Καινούρια)
if 'price' in df.columns and 'release_year' in df.columns:
    plt.figure(figsize=(12, 6))
    # Δημιουργία κατηγοριών για παλιά (πριν το 2015) και καινούρια (2015 και μετά) παιχνίδια
    df['era'] = pd.cut(df['release_year'],
                       bins=[0, 2014, df['release_year'].max()],
                       labels=['Παλιά (μέχρι 2014)', 'Καινούρια (2015+)'])
    # Boxplot για σύγκριση των τιμών ανά εποχή
    # Περιορίζουμε τις τιμές μέχρι 100 για καλύτερη οπτικοποίηση
    price_data = df[(df['price'] >= 0) & (df['price'] <= 100) & (df['era'].notna())]
    if not price_data.empty:
        sns.boxplot(x='era', y='price', data=price_data, palette='Set2')
        plt.title('Σύγκριση Κατανομής Τιμών για Παλιά vs Καινούρια Παιχνίδια Steam\n(Παλιά = μέχρι 2014, Καινούρια = 2015 και μετά)', fontsize=14)
        plt.xlabel('Εποχή Παιχνιδιού')
        plt.ylabel('Τιμή (USD)')
        
        # Προσθήκη στατιστικών πληροφοριών
        old_games = price_data[price_data['era'] == 'Παλιά (μέχρι 2014)']['price']
        new_games = price_data[price_data['era'] == 'Καινούρια (2015+)']['price']
        
        stats_text = f"Παλιά: Μέση τιμή = ${old_games.mean():.2f}, Διάμεσος = ${old_games.median():.2f}\n"
        stats_text += f"Καινούρια: Μέση τιμή = ${new_games.mean():.2f}, Διάμεσος = ${new_games.median():.2f}"
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('price_comparison_old_vs_new.png', dpi=150)
        print("Το γράφημα 'price_comparison_old_vs_new.png' αποθηκεύτηκε.")
    else:
        print("Δεν υπάρχουν αρκετά δεδομένα για boxplot τιμών")
else:
    print("Δεν υπάρχουν στήλες price ή release_year για το boxplot")

# Βήμα 5: Εγγραφή Αποτελεσμάτων σε Αρχείο .txt
print("\n--- ΕΓΓΡΑΦΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ ---")
output_filename = 'analysis_results.txt'
try:
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("ΑΠΟΤΕΛΕΣΜΑΤΑ ΑΝΑΛΥΣΗΣ STEAM DATASET (Μάρτιος 2025)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Συνολικός αριθμός παιχνιδιών στο dataset: {df.shape[0]}\n")
        f.write(f"Αριθμός στηλών που χρησιμοποιήθηκαν: {df.shape[1]}\n")
        f.write(f"Στήλες: {', '.join(df.columns.tolist())}\n\n")

        f.write("1. ΚΑΤΑΜΕΤΡΗΣΗ ΠΑΙΧΝΙΔΙΩΝ ΑΝΑ ΕΤΟΣ (Top 10)\n")
        f.write("-"*50 + "\n")
        if 'release_year' in df.columns:
            f.write(games_per_year.to_string())
        else:
            f.write("Δεν υπάρχει στήλη release_year")
        f.write("\n\n")

        f.write("2. ΒΑΣΙΚΑ ΣΤΑΤΙΣΤΙΚΑ ΓΙΑ ΤΙΜΕΣ (price)\n")
        f.write("-"*50 + "\n")
        if 'price' in df.columns:
            f.write(price_stats.to_string())
        else:
            f.write("Δεν υπάρχει στήλη price")
        f.write("\n\n")

        f.write("3. ΣΤΑΤΙΣΤΙΚΑ ΓΙΑ ΒΑΘΜΟΛΟΓΙΕΣ\n")
        f.write("-"*50 + "\n")
        if 'user_score' in df.columns:
            f.write("User Score (Γνώμη Παικτών):\n")
            f.write(user_score_stats.to_string())
            f.write("\n\n")
        if 'metacritic_score' in df.columns:
            f.write("Metacritic Score (Γνώμη Κριτικών):\n")
            f.write(metacritic_stats.to_string())
            f.write("\n\n")

        f.write("4. ΣΥΝΟΨΗ ΤΙΜΩΝ ΚΑΙ ΒΑΘΜΟΛΟΓΙΩΝ ΑΝΑ ΕΤΟΣ\n")
        f.write("-"*50 + "\n")
        if yearly_stats_filtered is not None:
            f.write(yearly_stats_filtered.to_string())
        else:
            f.write("Δεν υπάρχουν αρκετά δεδομένα για ομαδοποίηση")
        f.write("\n\n")

        f.write("5. ΚΥΡΙΑ ΣΥΜΠΕΡΑΣΜΑΤΑ\n")
        f.write("-"*50 + "\n")
        f.write("- Τα περισσότερα παιχνίδια στο Steam έχουν κυκλοφορήσει τα τελευταία χρόνια.\n")
        
        if 'release_year' in df.columns and not df['release_year'].isna().all():
            min_year = int(df['release_year'].min())
            max_year = int(df['release_year'].max())
            f.write(f"- Τα δεδομένα καλύπτουν την περίοδο από {min_year} έως {max_year}.\n")
            
            if games_per_year is not None and not games_per_year.empty:
                peak_year = games_per_year.idxmax()
                peak_count = games_per_year.max()
                f.write(f"- Το έτος με τα περισσότερα παιχνίδια είναι το {peak_year} με {peak_count} παιχνίδια.\n")
        
        if 'price' in df.columns:
            avg_price = df['price'].mean()
            median_price = df['price'].median()
            f.write(f"- Η μέση τιμή των παιχνιδιών είναι {avg_price:.2f} USD και η διάμεσος τιμή είναι {median_price:.2f} USD.\n")
        
        if all(col in df.columns for col in ['user_score', 'metacritic_score']):
            # Υπολογισμός συσχέτισης αν υπάρχουν αρκετά δεδομένα
            scores_df = df[['user_score', 'metacritic_score']].dropna()
            if len(scores_df) > 10:
                correlation = scores_df['user_score'].corr(scores_df['metacritic_score'])
                f.write(f"- Η συσχέτιση μεταξύ user score (παίκτες) και metacritic score (κριτικοί) είναι {correlation:.3f}.\n")
                if correlation > 0.7:
                    f.write("  (Υψηλή θετική συσχέτιση: Παίκτες και κριτικοί τείνουν να συμφωνούν)\n")
                elif correlation > 0.3:
                    f.write("  (Μέτρια θετική συσχέτιση: Υπάρχει κάποια συμφωνία)\n")
                else:
                    f.write("  (Χαμηλή συσχέτιση: Παίκτες και κριτικών έχουν διαφορετικές απόψεις)\n")
        
        f.write("\nΓΡΑΦΗΜΑΤΑ ΠΟΥ ΔΗΜΙΟΥΡΓΗΘΗΚΑΝ:\n")
        if os.path.exists('avg_price_per_year.png'):
            f.write("  - avg_price_per_year.png: Μέση τιμή ανά έτος κυκλοφορίας\n")
        if os.path.exists('user_vs_metacritic_score.png'):
            f.write("  - user_vs_metacritic_score.png: Σύγκριση user score vs metacritic score\n")
        if os.path.exists('price_comparison_old_vs_new.png'):
            f.write("  - price_comparison_old_vs_new.png: Σύγκριση τιμών παλιών vs καινούριων παιχνιδιών\n")
        
        f.write(f"\nΗ ανάλυση ολοκληρώθηκε: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"Τα αποτελέσματα αποθηκεύτηκαν στο αρχείο '{output_filename}'.")
except Exception as e:
    print(f"Σφάλμα κατά την εγγραφή του αρχείου: {e}")

print("\n" + "="*50)
print("Η ΑΝΑΛΥΣΗ ΟΛΟΚΛΗΡΩΘΗΚΕ ΕΠΙΤΥΧΩΣ!")
print("="*50)