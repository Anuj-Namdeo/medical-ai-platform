import sqlite3
import pandas as pd
import os

db_path = r"database\medical_ai.db"

conn = sqlite3.connect(db_path)

tables = [
    "patients",
    "predictions",
    "model_versions",
    "audit_logs",
    "image_metadata"
]

output_folder = "csv_exports"
os.makedirs(output_folder, exist_ok=True)

for table in tables:
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)

    csv_path = os.path.join(output_folder, f"{table}.csv")

    df.to_csv(csv_path, index=False)

    print(f"Exported: {csv_path}")

conn.close()

print("\nAll CSV files exported successfully!")