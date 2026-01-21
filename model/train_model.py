"""
Wine Cultivar Prediction - Debug Version
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("="*60)
print("WINE CULTIVAR PREDICTION - MODEL TRAINING")
print("="*60)

# Load with standard column names
print("\n1. Loading wine.csv...")
df = pd.read_csv('wine.csv', header=None, names=[
    'cultivar', 'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash',
    'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
    'proanthocyanins', 'color_intensity', 'hue', 'od280_od315', 'proline'
])

print(f"   ✓ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Show all columns
print("\n2. All columns in dataset:")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2}. {col}")

# Convert cultivar
df['cultivar'] = df['cultivar'] - 1
print(f"\n3. Target column ready")
print(f"   Classes: {sorted(df['cultivar'].unique())}")
print(f"   Distribution: Cultivar 0={sum(df['cultivar']==0)}, Cultivar 1={sum(df['cultivar']==1)}, Cultivar 2={sum(df['cultivar']==2)}")

# Select 6 features - THESE EXACT NAMES
selected_features = [
    'alcohol',
    'malic_acid', 
    'ash',
    'magnesium',
    'flavanoids',
    'color_intensity'
]

print(f"\n4. Selecting 6 features:")
for i, feat in enumerate(selected_features, 1):
    print(f"   {i}. {feat}")

# Verify all features exist
missing = [f for f in selected_features if f not in df.columns]
if missing:
    print(f"\n   ✗ ERROR: Features not found: {missing}")
    print(f"   Available columns: {list(df.columns)}")
    exit()

X = df[selected_features]
y = df['cultivar']

print(f"\n5. Data prepared")
print(f"   Features shape: {X.shape}")
print(f"   Target shape: {y.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n6. Data split")
print(f"   Training: {len(X_train)} samples")
print(f"   Testing: {len(X_test)} samples")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n7. Features scaled ✓")

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

print("\n8. Model trained ✓")

# Evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)
print(f"\nAccuracy: {accuracy*100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, 
                          target_names=['Cultivar 0', 'Cultivar 1', 'Cultivar 2']))
print("="*60)

# Save model
print("\n9. Saving model files...")
joblib.dump(model, 'wine_cultivar_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(selected_features, 'feature_names.pkl')

print("   ✓ wine_cultivar_model.pkl")
print("   ✓ scaler.pkl")
print("   ✓ feature_names.pkl")

# Verify files
import os
if os.path.exists('wine_cultivar_model.pkl'):
    print(f"\n   Model file size: {os.path.getsize('wine_cultivar_model.pkl')} bytes")
    print("   ✓ Files created successfully!")
else:
    print("\n   ✗ ERROR: Files not created!")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print("\nNext step: Run 'cd ..' then 'python app.py' to test the web app")