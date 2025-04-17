import requests
import joblib
import pandas as pd
from datetime import datetime
import time

# API anahtarını buraya gir
API_KEY = "6a72c47692bd418b2965ff2cfe6e9b2d"
HEADERS = {"x-apisports-key": API_KEY}
BASE_URL = "https://v3.football.api-sports.io"

# Model yükleme fonksiyonu
def load_model(model_filename):
    return joblib.load(model_filename)

# Bugün oynanacak maçları çeken fonksiyon (kesin düzeltildi!)
def get_today_fixtures():
    today = datetime.today().strftime('%Y-%m-%d')
    url = f"{BASE_URL}/fixtures?date={today}&status=NS"
    res = requests.get(url, headers=HEADERS)
    return res.json().get("response", [])

# Takımların son 3 maçını getiren fonksiyon
def get_last_matches(team_id, n=3):
    url = f"{BASE_URL}/fixtures?team={team_id}&last={n}"
    res = requests.get(url, headers=HEADERS)
    return res.json().get("response", [])

# Ortalama gol hesaplama fonksiyonu
def calculate_goal_avg(matches, team_id):
    total_goals = 0
    for match in matches:
        goals = match['goals']
        home_id = match['teams']['home']['id']
        is_home = (team_id == home_id)
        goals_for = goals['home'] if is_home else goals['away']
        goals_for = goals_for if goals_for is not None else 0
        total_goals += goals_for
    return round(total_goals / len(matches), 2) if matches else 0

# Tahmin girdisini hazırlayan fonksiyon
def hazirla_input(home_id, away_id):
    last_home_matches = get_last_matches(home_id)
    last_away_matches = get_last_matches(away_id)
    goal_avg_home = calculate_goal_avg(last_home_matches, home_id)
    goal_avg_away = calculate_goal_avg(last_away_matches, away_id)
    goal_diff = round(goal_avg_home - goal_avg_away, 2)
    goal_total = round(goal_avg_home + goal_avg_away, 2)
    return [[goal_diff, goal_total]]

# Tahminleri yapan ve okunaklı hale getiren fonksiyon
def make_predictions(input_data):
    base_path = "C:/Users/batuh/OneDrive/Masaüstü/GOALMASTER/"
    # Modelleri yükle
    model_match_result = load_model(base_path + "model_match_result.pkl")
    model_score_label = load_model(base_path + "model_score_label.pkl")
    model_kg_var = load_model(base_path + "model_kg_var.pkl")
    model_over_2_5 = load_model(base_path + "model_over_2_5.pkl")
    model_over_3_5 = load_model(base_path + "model_over_3_5.pkl")

    # Encoder'ları yükle (okunaklı sonuçlar için)
    encoder_match_result = load_model(base_path + "label_encoder_match_result.pkl")
    encoder_score_label = load_model(base_path + "label_encoder_score_label.pkl")

    # Tahminleri yap
    match_result_prediction = model_match_result.predict(input_data)[0]
    score_label_prediction = model_score_label.predict(input_data)[0]
    kg_var_prediction = model_kg_var.predict(input_data)[0]
    over_2_5_prediction = model_over_2_5.predict(input_data)[0]
    over_3_5_prediction = model_over_3_5.predict(input_data)[0]

    # Sonuçları okunaklı hale getir
    return {
        "Maç Sonucu": encoder_match_result.inverse_transform([match_result_prediction])[0],
        "Tahmini Skor": encoder_score_label.inverse_transform([score_label_prediction])[0],
        "Karşılıklı Gol": "Var" if kg_var_prediction else "Yok",
        "2.5 Üst": "Üst" if over_2_5_prediction else "Alt",
        "3.5 Üst": "Üst" if over_3_5_prediction else "Alt"
    }

# Ana çalışma bölümü
fixtures = get_today_fixtures()
print(f"Bugün {len(fixtures)} maç bulundu.\n")

for fixture in fixtures:
    home = fixture['teams']['home']
    away = fixture['teams']['away']
    home_id, away_id = home['id'], away['id']
    
    print(f"⚽ {home['name']} vs {away['name']}")

    input_data = hazirla_input(home_id, away_id)
    predictions = make_predictions(input_data)

    for tahmin_tipi, sonuc in predictions.items():
        print(f" - {tahmin_tipi}: {sonuc}")

    print("\n" + "-"*30 + "\n")
    time.sleep(1.2)
