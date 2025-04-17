from fastapi import FastAPI
import requests
import joblib
from datetime import datetime

app = FastAPI()

# API bilgileri
API_KEY = "6a72c47692bd418b2965ff2cfe6e9b2d"
HEADERS = {"x-apisports-key": API_KEY}
BASE_URL = "https://v3.football.api-sports.io"

# Model yükleme fonksiyonu
def load_model(model_filename):
    return joblib.load(model_filename)

# Maçları çekme
def get_today_fixtures():
    today = datetime.today().strftime('%Y-%m-%d')
    url = f"{BASE_URL}/fixtures?date={today}&status=NS"
    res = requests.get(url, headers=HEADERS)
    return res.json().get("response", [])

# Son 3 maç verisi
def get_last_matches(team_id, n=3):
    url = f"{BASE_URL}/fixtures?team={team_id}&last={n}"
    res = requests.get(url, headers=HEADERS)
    return res.json().get("response", [])

def calculate_goal_avg(matches, team_id):
    total_goals = sum(
        match['goals']['home'] if match['teams']['home']['id'] == team_id else match['goals']['away']
        for match in matches if match['goals']['home'] is not None and match['goals']['away'] is not None
    )
    return round(total_goals / len(matches), 2) if matches else 0

def hazirla_input(home_id, away_id):
    goal_avg_home = calculate_goal_avg(get_last_matches(home_id), home_id)
    goal_avg_away = calculate_goal_avg(get_last_matches(away_id), away_id)
    return [[round(goal_avg_home - goal_avg_away, 2), round(goal_avg_home + goal_avg_away, 2)]]

def make_predictions(input_data):
    model_path = "./"
    model_match_result = load_model(model_path + "model_match_result.pkl")
    model_score_label = load_model(model_path + "model_score_label.pkl")
    model_kg_var = load_model(model_path + "model_kg_var.pkl")
    model_over_2_5 = load_model(model_path + "model_over_2_5.pkl")
    model_over_3_5 = load_model(model_path + "model_over_3_5.pkl")

    encoder_match_result = load_model(model_path + "label_encoder_match_result.pkl")
    encoder_score_label = load_model(model_path + "label_encoder_score_label.pkl")

    return {
        "match_result": encoder_match_result.inverse_transform([model_match_result.predict(input_data)[0]])[0],
        "score_label": encoder_score_label.inverse_transform([model_score_label.predict(input_data)[0]])[0],
        "kg_var": "Var" if model_kg_var.predict(input_data)[0] else "Yok",
        "over_2_5": "Üst" if model_over_2_5.predict(input_data)[0] else "Alt",
        "over_3_5": "Üst" if model_over_3_5.predict(input_data)[0] else "Alt"
    }

@app.get("/")
def home():
    return {"mesaj": "GOALMASTER API çalışıyor!"}

@app.get("/tahminler")
def tahminleri_getir():
    fixtures = get_today_fixtures()
    tahminler = []
    for fixture in fixtures:
        home = fixture['teams']['home']
        away = fixture['teams']['away']
        input_data = hazirla_input(home['id'], away['id'])
        predictions = make_predictions(input_data)
        tahminler.append({
            "ev_sahibi": home['name'],
            "deplasman": away['name'],
            "tahminler": predictions
        })
    return tahminler
