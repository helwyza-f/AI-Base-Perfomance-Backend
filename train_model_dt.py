import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

# Seed agar hasil konsisten
np.random.seed(42)

# Jumlah sampel
n_samples = 5000

# Pendidikan: 1 (SMP), ..., 6 (S3)
pendidikan = np.random.choice(
    [1, 2, 3, 4, 5, 6], 
    size=n_samples, 
    p=[0.1, 0.2, 0.25, 0.25, 0.15, 0.05]
)

# Pengalaman kerja: 0 - 5 tahun
pengalaman_kerja = np.random.randint(0, 6, size=n_samples)

# Skor psikotes: 30 - 90
skor_psikotes = np.random.randint(30, 91, size=n_samples)

# Rumus performa
performa = (
    (pendidikan / 6) * 35 +           # max 35 poin
    (pengalaman_kerja / 5) * 25 +     # max 25 poin
    (skor_psikotes / 90) * 40         # max 40 poin
)

# Tambahkan noise kecil
performa += np.random.normal(0, 3, size=n_samples)
performa = np.clip(performa, 0, 100)

# Buat DataFrame
df = pd.DataFrame({
    'pendidikan': pendidikan,
    'pengalaman_kerja': pengalaman_kerja,
    'skor_psikotes': skor_psikotes,
    'performa': performa
})

# Tambah fitur interaksi
df['edu_x_exp'] = df['pendidikan'] * df['pengalaman_kerja']
df['exp_x_score'] = df['pengalaman_kerja'] * df['skor_psikotes']

# Split train-test
X = df[['pendidikan', 'pengalaman_kerja', 'skor_psikotes', 'edu_x_exp', 'exp_x_score']]
y = df['performa']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Buat dan latih model Decision Tree
model_dt = DecisionTreeRegressor(
    max_depth=5,
    min_samples_split=20,
    random_state=42
)
model_dt.fit(X_train, y_train)

# Evaluasi
y_pred = model_dt.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"✅ Model trained")
print(f"R² Score       : {r2:.4f}")
print(f"Mean Abs Error : {mae:.2f}")

# Simpan model
joblib.dump(model_dt, 'model_dt.pkl')
print("🧠 Model saved to model_dt.pkl")
