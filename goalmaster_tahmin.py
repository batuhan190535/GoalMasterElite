import requests
import joblib
import csv
from datetime import datetime

# === API AYARLARI ===
API_KEY = "6a72c47692bd418b2965ff2cfe6e9b2d"
HEADERS = {"x-rapidapi-key": API_KEY}
BASE_URL = "https://v3.football.api-sports.io"

# === MODEL YÜKLE ===
model_match_result = joblib.load("model_match_result.pkl")
model_score_label = joblib.load("model_score_label.pkl")
model_kg_var = joblib.load("model_kg_var.pkl")
model_over_2_5 = joblib.load("model_over_2_5.pkl")
model_over_3_5 = joblib.load("model_over_3_5.pkl")

encoder_match_result = joblib.load("label_encoder_match_result.pkl")
encoder_score_label = joblib.load("label_encoder_score_label.pkl")

# === MAÇLARI GETİR ===
def get_today_fixtures():
    today = datetime.now().strftime("%Y-%m-%d")
    url = f"{BASE_URL}/fixtures?date={today}&status=NS"
    res = requests.get(url, headers=HEADERS)
    return res.json().get("response", [])

# === GİRİŞ VERİSİNİ HAZIRLA ===
def hazirla_input(home_id, away_id):
    input_data = {
        "avg_goals_home": 1.2,
        "avg_goals_away": 1.0,
        "home_form": 3,
        "away_form": 2
    }
    return list(input_data.values())

# === TAHMİN YAP ===
def make_predictions(input_data):
    input_data_reshaped = [input_data]

    match_result_prediction = encoder_match_result.inverse_transform(
        model_match_result.predict(input_data_reshaped)
    )[0]

    score_label_prediction = encoder_score_label.inverse_transform(
        model_score_label.predict(input_data_reshaped)
    )[0]

    kg_var_prediction = "Var" if model_kg_var.predict(input_data_reshaped)[0] == 1 else "Yok"
    over_2_5_prediction = "Üst" if model_over_2_5.predict(input_data_reshaped)[0] == 1 else "Alt"
    over_3_5_prediction = "Üst" if model_over_3_5.predict(input_data_reshaped)[0] == 1 else "Alt"

    return {
        "match_result": match_result_prediction,
        "score_label": score_label_prediction,
        "kg_var": kg_var_prediction,
        "over_2_5": over_2_5_prediction,
        "over_3_5": over_3_5_prediction
    }

# === CSV'YE KAYDET ===
def save_predictions_to_csv(home, away, predictions):
    with open("goalmaster_tahminler.csv", mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            home, away,
            predictions["match_result"],
            predictions["score_label"],
            predictions["kg_var"],
            predictions["over_2_5"],
            predictions["over_3_5"]
        ])

# === GÜNLÜK TAHMİN ÇALIŞTIRICI ===
def run_daily_prediction():
    fixtures = get_today_fixtures()
    print(f"📅 Bugün {len(fixtures)} maç bulundu.")

    for fixture in fixtures:
        home = fixture["teams"]["home"]["name"]
        away = fixture["teams"]["away"]["name"]
        print(f"⚽ {home} vs {away}")

        try:
            input_data = hazirla_input(fixture["teams"]["home"]["id"], fixture["teams"]["away"]["id"])
            predictions = make_predictions(input_data)
            save_predictions_to_csv(home, away, predictions)
        except Exception as e:
            print(f"❌ Hata: {e}")

# === MANUEL TEST İÇİN ===
if __name__ == "__main__":
    run_daily_prediction()
