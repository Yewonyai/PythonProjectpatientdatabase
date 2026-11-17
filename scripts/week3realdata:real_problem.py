import numpy as np


np.random.seed(7)
temps = np.random.uniform(low=15.0, high=45.0, size=(30, 5))
print(temps)

indices = np.random.choice(temps.size, size = 10, replace = False)
temps.flat[indices] = np.nan
print(np.isnan(temps).sum())

for Day, readings in enumerate(temps, start = 1):
    print(f"\nDay {Day}")
    for sensor, value in enumerate(readings, start = 1):
        print(f"\nsensor {sensor}: {value}")
nan_per_sensor = np.isnan(temps).sum(axis =  0)
for sensor, count in enumerate(nan_per_sensor, start = 1):
    print(f"\nsensor {sensor}: {count} NaN value(s)")

days_with_nan = np.where(np.isnan(temps).any(axis = 1))[0]
print("Days with at least one Nan readings:")
for  day in days_with_nan:
    print(f"\nDay {day+1}")

column_means = np.nanmean(temps, axis = 0)
rows, cols = np.where (np.isnan(temps))
# Correct approach
temps_clean = temps.copy()
column_means = np.nanmean(temps, axis=0)
rows, cols = np.where(np.isnan(temps_clean))
# Now temps still has its original NaNs, temps_clean is the imputed version
print(np.isnan(temps).sum())

#hottest sensor
sensor_means = np.mean(temps, axis = 0)
for sensor, avg in  enumerate(sensor_means, start = 1):
    print(f"\nsensor {sensor}: {avg:.2f}°C")
hottest_sensor = np.argmax(sensor_means)
print(f"\nsensor {hottest_sensor + 1} runs hottest on average.")

day_means = np.mean(temps, axis = 1)
hottest_day = np.argmax(day_means)
print(f"\nDay {hottest_day+1} had the highest average temperature.")
print(f"\naverage temperature: {day_means[hottest_day]:.2f}°C")

heat_alerts = temps > 40
alerts_per_sensor = np.sum(heat_alerts, axis=0)
for sensor, alerts in enumerate(alerts_per_sensor, start = 1):
    print(f"\nsensor {sensor}: {alerts} heat alerts(s)")
 # questions and comment section
print(f"\nreplacing a NaN with   the columns mean allows us to keep all available data. "
      f"\n if we delete an entire row because one sensor reading is missing,"
      f"\n we also lose the other valid sensor reading from that day"
      f"\n deleting rows can also reduce the dataset size and may affect the accuracy of the analysis"
      f"Second question"
      f"\n Repacing NaN  values with the global mean can distort the data because "
      f"\n different sensors may operate at different average temperatures."
      f"\n the global mean  may not accurately represent a particular sensor,"
      f"\n leading to biased results and less reliable analysis."

      )

sensor1 = temps[:,0]
rolling_avg = np.array([
np.mean(sensor1[i:i+7])
    for i in range(len(sensor1) - 6)])
print(f"\n7 days rolling average for sensor 1:", rolling_avg)
