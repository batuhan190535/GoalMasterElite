# daily_task.py

from datetime import datetime
from goalmaster_tahmin import run_daily_prediction

def main():
    print("🚀 Günlük GOALMASTER tahmin otomasyonu başlatıldı.")
    print("🕘 Tarih:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    run_daily_prediction()
    print("✅ Tahminler başarıyla tamamlandı.")

if __name__ == "__main__":
    main()
