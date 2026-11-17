

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(7)
temps = np.random.uniform(low=15.0, high=45.0, size=(30, 5))


indices = np.random.choice(temps.size, size = 10, replace = False)
temps.flat[indices] = np.nan

# df = pd.DataFrame(temps)


def clean_data(arr):
    """
    Convert a raw NumPy temperature array into a cleaned Pandas DataFrame.

    Parameters:
        arr (np.ndarray): Raw temperature data with possible NaN values.

    Returns:
        pd.DataFrame: Cleaned DataFrame with:
            - named rows and columns
            - NaNs filled with column means
            - Daily_Avg column
            - Status column
    """
    # Work on a copy so the input is never modified
    data = arr.copy()

    # Create DataFrame
    df = pd.DataFrame(
        data,
        index=[f"Day_{i+1}" for i in range(data.shape[0])],
        columns=[f"Sensor_{i+1}" for i in range(data.shape[1])]

    )
    # df.columns = [f"sensor_{i}]" for i, sensor in enumerate(df.columns, start=1)]
    # df.index = (f"day_{i}" for i, day in enumerate(df.index, start=1))
    # print(f"", df)
    #

    # Fill NaNs with column means
    df = df.fillna(df.mean())

    # Add daily average
    # df["Daily_Avg"] = df.mean(axis=1)
    sensor_cols = ["Sensor_1", "Sensor_2", "Sensor_3", "Sensor_4", "Sensor_5"]
    df["Daily_Avg"] = df[sensor_cols].mean(axis=1)
    # Add status based on daily average

    ALERT_THRESHOLD = 35  # define once

    df["Status"] = np.where(df["Daily_Avg"] >= 35, "ALERT", "NORMAL")

    return df


print(clean_df)

plt.figure(figsize=(10,5))

plt.plot(clean_df.index, clean_df["Daily_Avg"], marker="o")

plt.title("Daily Average Temperature Across 30 Days")
plt.xlabel("Days")
plt.ylabel("Average Temperature (°C)")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("daily_average_temperature.png")

plt.show()


# Correct — all 5 sensors
sensor_cols = ["Sensor_1", "Sensor_2", "Sensor_3", "Sensor_4", "Sensor_5"]
heat_alerts = (clean_df[sensor_cols] >= 35).sum()

plt.figure(figsize=(6,5))

plt.bar(heat_alerts.index, heat_alerts.values)

plt.title("Heat Alerts per Sensor")
plt.xlabel("Sensors")
plt.ylabel("Number of Heat Alerts")

plt.tight_layout()

plt.savefig("heat_alerts_per_sensor.png")

plt.show()

print(f"\nWrapping the data-cleaning pipeline in a function makes the code reusable,"
      f" organized, and easier to maintain. "
      f"\nInstead of rewriting the same steps for every new dataset, "
      f"\nyou can simply call the function with a different input array."
      f"\n It also makes the code easier to test, debug,"
      f"\n and modify because all the cleaning logic is contained in one place. "
      f"\nAdditionally, "
      f"\nthe function avoids changing the original data by working on a copy of the input array."
      f"\nplt.savefig() saves the current plot as an image file (such as a .png)"
      f" \non your computer."

)



plt.figure(figsize=(10, 5))

# Plot the Daily Average
plt.plot(
    clean_df.index,
    clean_df["Daily_Avg"],
    marker="o",
    label="Daily Average"
)

# Plot Sensor_1 readings
plt.plot(
    clean_df.index,
    clean_df["Sensor_1"],
    marker="s",
    label="Sensor_1"
)

# Add a horizontal alert threshold line at 35°C
plt.axhline(
    y=35,
    color="red",
    linestyle="--",
    label="Alert Threshold (35°C)"
)

# Add title and labels
plt.title("Daily Average and Sensor_1 Temperature Readings")
plt.xlabel("Days")
plt.ylabel("Temperature (°C)")

# Rotate day labels
plt.xticks(rotation=45)

# Display the legend
plt.legend()

# Adjust spacing
plt.tight_layout()

# Save the plot
plt.savefig("temperature_trend_with_threshold.png")

# Display the plot
plt.show()

#end