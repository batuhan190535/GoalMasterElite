import pandas as pd

# CSV'yi yükle
df = pd.read_csv("xg_dataset_2024.csv")

# Boş skorları sıfırla
df["home_goals"] = df["home_goals"].fillna(0).astype(int)
df["away_goals"] = df["away_goals"].fillna(0).astype(int)

# Özellikler
df["goal_diff"] = df["home_goals"] - df["away_goals"]
df["goal_total"] = df["home_goals"] + df["away_goals"]

# Maç sonucu
def get_result(row):
    if row["home_goals"] > row["away_goals"]:
        return "MS1"
    elif row["home_goals"] < row["away_goals"]:
        return "MS2"
    else:
        return "MS0"

df["match_result"] = df.apply(get_result, axis=1)

# KG Var / Yok
df["kg_var"] = ((df["home_goals"] > 0) & (df["away_goals"] > 0)).astype(int)

# 2.5 üst
df["over_2_5"] = (df["goal_total"] > 2.5).astype(int)

# 3.5 üst
df["over_3_5"] = (df["goal_total"] > 3.5).astype(int)

# Skor etiketi
df["score_label"] = df["home_goals"].astype(str) + "-" + df["away_goals"].astype(str)

# Hazır tabloyu görelim
df[["home_team", "away_team", "home_goals", "away_goals", "goal_diff", "goal_total", "match_result", "kg_var", "over_2_5", "over_3_5", "score_label"]].head()

df.to_csv("processed_dataset.csv", index=False)
print("Veri işlendi ve 'processed_dataset.csv' olarak kaydedildi.")
