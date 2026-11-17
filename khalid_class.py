import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Reproducible environment simulation
np.random.seed(42)
n_rows = 10500
start_time = datetime(2026, 1, 1, 0, 0, 0)
timestamps = [start_time + timedelta(minutes=i*5) for i in range(n_rows)]

data = {
    "Timestamp": [t.strftime("%Y-%m-%d %H:%M:%S") for t in timestamps],
    "PM2.5_ug_m3": np.random.exponential(scale=18.0, size=n_rows) + 5.0,
    "MQ135_CO2_ppm": np.random.normal(loc=550.0, scale=150.0, size=n_rows),
    "MQ7_CO_ppm": np.random.exponential(scale=2.5, size=n_rows) + 0.2,
    "MQ131_O3_ppm": np.random.uniform(0.01, 0.09, size=n_rows),
    "DHT22_Temp_C": np.random.normal(loc=26.0, scale=5.0, size=n_rows),
    "DHT22_Humidity_Pct": np.random.normal(loc=55.0, scale=15.0, size=n_rows)
}

df = pd.DataFrame(data)

# Real-world boundary clipping for low-cost breakout boards
df["PM2.5_ug_m3"] = df["PM2.5_ug_m3"].clip(2.0, 150.0).round(1)
df["MQ135_CO2_ppm"] = df["MQ135_CO2_ppm"].clip(350.0, 2000.0).round(0).astype(int)
df["MQ7_CO_ppm"] = df["MQ7_CO_ppm"].clip(0.05, 35.0).round(2)
df["MQ131_O3_ppm"] = df["MQ131_O3_ppm"].clip(0.005, 0.25).round(3)
df["DHT22_Temp_C"] = df["DHT22_Temp_C"].clip(10.0, 45.0).round(1)
df["DHT22_Humidity_Pct"] = df["DHT22_Humidity_Pct"].clip(10.0, 100.0).round(1)

# Overlay logical external variables (Rush hours & midday thermal inversions)
for i in range(n_rows):
    t = timestamps[i]
    if t.hour in [8, 9, 17, 18, 19]: # Traffic spikes
        df.at[i, "PM2.5_ug_m3"] += np.random.uniform(15, 45)
        df.at[i, "MQ7_CO_ppm"] += np.random.uniform(4, 12)
    if t.hour in [12, 13, 14, 15]: # Ozone formation via heat
        df.at[i, "MQ131_O3_ppm"] += np.random.uniform(0.02, 0.06)

df["PM2.5_ug_m3"] = df["PM2.5_ug_m3"].round(1)
df["MQ7_CO_ppm"] = df["MQ7_CO_ppm"].round(2)
df["MQ131_O3_ppm"] = df["MQ131_O3_ppm"].round(3)

# Clinical label mapping logic (Asthma Trigger Targets)
conditions = [
    (df["PM2.5_ug_m3"] > 35.0) |
    (df["MQ131_O3_ppm"] > 0.070) |
    (df["MQ7_CO_ppm"] > 9.0) |
    ((df["DHT22_Humidity_Pct"] > 75.0) & (df["DHT22_Temp_C"] > 31.0)) |
    ((df["DHT22_Humidity_Pct"] < 25.0) & (df["DHT22_Temp_C"] < 15.0)) # Extreme dry/cold
]

df["Asthma_Danger"] = np.select(conditions, [1], default=0)
df.to_csv("esp32_asthma_dataset_10k.csv", index=False)
print("File successfully generated as 'esp32_asthma_dataset_10k.csv'")



# File successfully generated as 'esp32_asthma_dataset_10k.csv'

# df = pd.read_csv("/content/esp32_asthma_dataset_10k.csv")