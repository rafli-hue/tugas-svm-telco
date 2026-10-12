import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

folder = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(folder, "WA_Fn-UseC_-Telco-Customer-Churn.csv"))

print("Jumlah baris dan kolom:", df.shape)
print(df.head(3).T)

# info dan tipe kolom
print("\nINFO KOLOM")
df.info()
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")
df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

# missing values
print("\nMISSING VALUES")
kosong = df.isna().sum()
print(kosong[kosong > 0])
print("Yang kosong dan tenure = 0:", (df[df["TotalCharges"].isna()]["tenure"] == 0).sum())
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# duplikasi
print("\nDUPLIKASI")
print("Baris duplikat:", df.duplicated().sum())
print("customerID duplikat:", df["customerID"].duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)

# kategori
print("\nKATEGORI")
numerik = ["tenure", "MonthlyCharges", "TotalCharges"]
kategori = [c for c in df.columns if c not in numerik + ["customerID"]]
for c in kategori:
    df[c] = df[c].str.strip()
    print(c, sorted(df[c].unique()))
for c in ["MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
          "TechSupport", "StreamingTV", "StreamingMovies"]:
    df[c] = df[c].replace({"No internet service": "No", "No phone service": "No"})

# outlier dengan IQR
print("\nOUTLIER (IQR)")
for kol in numerik:
    q1 = df[kol].quantile(0.25)
    q3 = df[kol].quantile(0.75)
    iqr = q3 - q1
    bawah = q1 - 1.5 * iqr
    atas = q3 + 1.5 * iqr
    jumlah = ((df[kol] < bawah) | (df[kol] > atas)).sum()
    print(kol, "batas:", round(bawah, 1), "sampai", round(atas, 1), "outlier:", jumlah)

# kelas
print("\nKELAS TARGET (Churn)")
print(df["Churn"].value_counts())
print(df["Churn"].value_counts(normalize=True).round(3))

# leakage
print("\nCEK LEAKAGE")
y = (df["Churn"] == "Yes").astype(int)
print("customerID unik semua:", df["customerID"].is_unique)
print(df[numerik].corrwith(y).abs().round(3))
print("TotalCharges vs tenure x MonthlyCharges:",
      round(df["TotalCharges"].corr(df["tenure"] * df["MonthlyCharges"]), 3))

# encoding kategori
biner = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
         "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
         "StreamingMovies", "PaperlessBilling"]
multi = ["InternetService", "Contract", "PaymentMethod"]

X = df[numerik].copy()
for c in biner:
    nilai = "Female" if c == "gender" else "Yes"
    X[c] = (df[c] == nilai).astype(int)
X = X.rename(columns={"gender": "gender_Female"})
X = pd.concat([X, pd.get_dummies(df[multi], drop_first=True, dtype=int)], axis=1)
print("\nJumlah fitur:", len(numerik) + len(biner) + len(multi))
print("Jumlah kolom setelah encoding:", X.shape[1])

# bagi data lalu standarisasi
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
X_train = X_train.copy()
X_test = X_test.copy()

scaler = StandardScaler()
X_train[numerik] = scaler.fit_transform(X_train[numerik])
X_test[numerik] = scaler.transform(X_test[numerik])
print("Data latih:", X_train.shape, "Data uji:", X_test.shape)
print("Rata-rata data latih setelah standarisasi:", X_train[numerik].mean().abs().max().round(6))
print("Std data latih setelah standarisasi:", X_train[numerik].std(ddof=0).mean().round(4))

# hasil pemrosesan
hasil = X_train.copy()
hasil["Churn"] = y_train.map({0: "No", 1: "Yes"})
hasil = hasil.sort_index()

print("\n5 DATA TERATAS HASIL PEMROSESAN")
print(hasil.head(5).round(4).T)
print("\n5 DATA TERBAWAH HASIL PEMROSESAN")
print(hasil.tail(5).round(4).T)

hasil.to_csv(os.path.join(folder, "telco_hasil_praproses.csv"))
