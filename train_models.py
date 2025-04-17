import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib

# Veriyi oku
df = pd.read_csv("processed_dataset.csv")

# Ortak input özellikleri
feature_columns = ["goal_diff", "goal_total"]

# Tahmin hedefleri
targets = {
    "match_result": "MS1 / MS0 / MS2",
    "score_label": "Tam skor",
    "kg_var": "KG Var / Yok",
    "over_2_5": "2.5 Üst / Alt",
    "over_3_5": "3.5 Üst / Alt"
}

# Model eğitimi
for target_col, description in targets.items():
    print(f"\n🎯 Eğitim: {description} → {target_col}")

    df_target = df.dropna(subset=feature_columns + [target_col])
    X = df_target[feature_columns]
    y = df_target[target_col]

    # Eğer etiketler string ise sayıya çevir
    label_encoder = None
    if y.dtype == 'object' or y.dtype.name == 'category':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    # Eğitim / test böl
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Doğruluk: {acc:.2%}")

    # Modeli kaydet
    model_path = f"model_{target_col}.pkl"
    joblib.dump(model, model_path)
    print(f"📦 Model kaydedildi: {model_path}")

    # Etiket dönüşümünü de kaydet
    if label_encoder:
        encoder_path = f"label_encoder_{target_col}.pkl"
        joblib.dump(label_encoder, encoder_path)
        print(f"🔤 LabelEncoder kaydedildi: {encoder_path}")
