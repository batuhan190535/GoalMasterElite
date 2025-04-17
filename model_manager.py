import joblib

class ModelManager:
    def __init__(self):
        # Model dosyalarını yükleyelim
        self.models = {
            'match_result': joblib.load('C:/Users/batuh/OneDrive/Masaüstü/GOALMASTER/model_match_result.pkl'),
            'score_label': joblib.load('C:/Users/batuh/OneDrive/Masaüstü/GOALMASTER/model_score_label.pkl'),
            'kg_var': joblib.load('C:/Users/batuh/OneDrive/Masaüstü/GOALMASTER/model_kg_var.pkl'),
            'over_2_5': joblib.load('C:/Users/batuh/OneDrive/Masaüstü/GOALMASTER/model_over_2_5.pkl'),
            'over_3_5': joblib.load('C:/Users/batuh/OneDrive/Masaüstü/GOALMASTER/model_over_3_5.pkl')
        }

    # Modeli yükle ve tahmin yap
    def make_predictions(self, input_data):
        input_data_reshaped = [list(input_data.values())]  # input verisini 2D hale getir

        # Her modelin tahminini alalım
        match_result_prediction = self.models['match_result'].predict(input_data_reshaped)
        score_label_prediction = self.models['score_label'].predict(input_data_reshaped)
        kg_var_prediction = self.models['kg_var'].predict(input_data_reshaped)
        over_2_5_prediction = self.models['over_2_5'].predict(input_data_reshaped)
        over_3_5_prediction = self.models['over_3_5'].predict(input_data_reshaped)

        return {
            "match_result": match_result_prediction[0],
            "score_label": score_label_prediction[0],
            "kg_var": kg_var_prediction[0],
            "over_2_5": over_2_5_prediction[0],
            "over_3_5": over_3_5_prediction[0],
        }

# Test verisi
input_data = {
    "goal_diff": 1.5,
    "goal_total": 3.0,
    "form_rate": 0.75,
    "title_chase": 1,
    "relegation_risk": 0
}

# ModelManager objesi oluştur
model_manager = ModelManager()

# Tahmin yapalım
predictions = model_manager.make_predictions(input_data)
print(predictions)
