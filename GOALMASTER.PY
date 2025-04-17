import requests
import time
import pandas as pd

API_KEY = "6a72c47692bd418b2965ff2cfe6e9b2d"
HEADERS = {"x-apisports-key": API_KEY}
BASE_URL = "https://v3.football.api-sports.io"

# Lig ID'leri ve adları
LEAGUES = {
    39: "Premier League",
    140: "La Liga",
    78: "Bundesliga",
    135: "Serie A",
    61: "Ligue 1",
    203: "Süper Lig",
    88: "Eredivisie",
    94: "Primeira Liga",
    235: "Rusya Premier Lig",
    144: "Belçika Pro League"
}

SEASON = 2024
match_data = []

def get_fixtures(league_id):
    url = f"{BASE_URL}/fixtures?league={league_id}&season={SEASON}"
    response = requests.get(url, headers=HEADERS)
    return response.json().get('response', [])

def get_statistics(fixture_id):
    url = f"{BASE_URL}/fixtures/statistics?fixture={fixture_id}"
    response = requests.get(url, headers=HEADERS)
    return response.json().get('response', [])

# Başla
for league_id, league_name in LEAGUES.items():
    print(f"\n⏳ {league_name} ({league_id}) liginden maçlar çekiliyor...")
    fixtures = get_fixtures(league_id)
    print(f"✅ {len(fixtures)} maç bulundu.")

    for fixture in fixtures:
        fixture_id = fixture['fixture']['id']
        home = fixture['teams']['home']['name']
        away = fixture['teams']['away']['name']
        g_home = fixture['goals']['home']
        g_away = fixture['goals']['away']

        # Varsayılan değer
        xg_home, xg_away = 0.0, 0.0

        try:
            stats = get_statistics(fixture_id)
            for team_stat in stats:
                team = team_stat['team']['name']
                for stat in team_stat['statistics']:
                    if stat['type'] == "Expected Goals":
                        val = stat['value'] if stat['value'] is not None else 0.0
                        if team == home:
                            xg_home = val
                        elif team == away:
                            xg_away = val

        except Exception as e:
            print(f"⚠️ HATA: {fixture_id} - {e} (Geçiliyor)")

        match_data.append({
            "fixture_id": fixture_id,
            "league_id": league_id,
            "league_name": league_name,
            "home_team": home,
            "away_team": away,
            "home_goals": g_home,
            "away_goals": g_away,
            "xg_home": xg_home,
            "xg_away": xg_away
        })

        time.sleep(1.2)  # rate limit için

# Kaydet
df = pd.DataFrame(match_data)
df.to_csv("xg_dataset_2024.csv", index=False)
print("\n🎯 Tüm veriler çekildi ve 'xg_dataset_2024.csv' olarak kaydedildi.")
