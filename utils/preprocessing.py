import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_scale_data(file_path):
    df = pd.read_csv(file_path)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    if 'category' not in df.columns or 'family' not in df.columns:
        raise ValueError("Expected columns: 'category' for benign/ransomware and 'family' for family labels.")

    # Binary labels: 0 = Benign, 1 = Ransomware
    binary_labels = df['category'].apply(lambda x: 0 if str(x).strip().lower() == 'benign' else 1).values

    # Family label encoding
    family_labels = LabelEncoder().fit_transform(df['family'].astype(str).str.strip())

    # Keep numeric features only
    X = df.drop(columns=['category', 'family']).select_dtypes(include=['number']).fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, binary_labels, family_labels
