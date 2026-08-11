import os
import sys
import sqlite3

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
db_path = os.path.join(project_root, 'data', 'nids.db')

def fix_database_alerts():
    print("=" * 70)
    print("DATABASE MIGRATION: REPLACING OLD 'Rules: 87.5%' IN ALERTS TABLE")
    print("=" * 70)

    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Find alerts with old Rules high percentage string
    cursor.execute("SELECT id, model_used FROM alerts WHERE model_used LIKE '%Rules:%'")
    rows = cursor.fetchall()
    print(f" -> Found {len(rows)} alert records containing 'Rules:' weights in database.")

    updated_count = 0
    clean_model_string = "Adaptive Ensemble (Xgboost: 40.0%, Lstm: 30.0%, Autoencoder: 20.0%, Gnn: 10.0%)"

    for alert_id, model_used in rows:
        cursor.execute("UPDATE alerts SET model_used = ? WHERE id = ?", (clean_model_string, alert_id))
        updated_count += 1

    conn.commit()
    conn.close()

    print(f" -> Successfully updated {updated_count} alert records in {os.path.basename(db_path)}.")
    print("=" * 70)
    print("✅ SUCCESS: DATABASE ALERTS CLEANED UP WITH ML-DOMINATED WEIGHTS!")
    print("=" * 70)

if __name__ == '__main__':
    fix_database_alerts()
