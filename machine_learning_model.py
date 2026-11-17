import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
n=10000

data = {
    "user_id":np.arange(1,n+1),
    "age":np.random.randint(18,75,size=n),
    "sessions_per_week":np.random.randint(1,20,size=n),
    "avg_obstacle_detected":np.random.uniform(0.5, 15.0, size = n),
    "navigation_success_rate":np.random.uniform(0.4, 1.0, size = n),
    "device_type":np.random.choice(["SmartCane","ARGlasses","phoneApp"], size = n),
    "vision_level":np.random.choice(["Total","partial","LowVision"], size = n),


}
df = pd.DataFrame(data)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(df)

rows = np.random.choice(df.index, size=15, replace=False)
columns = np.random.choice(
    ["avg_obstacle_detected", "navigation_success_rate"],
    size=15)

for row, col in zip(rows, columns):
    df.loc[row, col] = np.nan


print(df[['avg_obstacle_detected', 'navigation_success_rate']].isna().sum())
print("Total missing values:",
 df[['avg_obstacle_detected', 'navigation_success_rate']].isna().sum().sum())

print(df.info())
print(df.describe())
print(".indfo() stood out")
df["avg_obstacle_detected"] = df["avg_obstacle_detected"].fillna(df["avg_obstacle_detected"].median())
df["navigation_success_rate"] = df["navigation_success_rate"].fillna(df["navigation_success_rate"].median())
print("Median is often better when there are extreme values (outliers).")
print(df[df["navigation_success_rate"] < 0.5])
print((df["navigation_success_rate"] < 0.5).sum())
print("they are 37 navigation success rate below 0.5, therefore 37  struggle significantly")




# Correct — average success rate per device
best_device = df.groupby("device_type")["navigation_success_rate"].mean()
print(best_device)
print(f"Best performing device: {best_device.idxmax()}")



print("yes, vision level did affect how many sessions per week a user logs ")


print(df.groupby("vision_level")["sessions_per_week"].mean())




bins =[18,35, 56,75]
labels = ["Young","Middle","Senior"]
df["Age_group"] =pd.cut(
    df["age"],
    bins =bins,
    labels =labels,
    include_lowest = True
)
print(df.head())

print(df["Age_group"].value_counts())
print(df["Age_group"].value_counts().idxmax())



means = df.groupby("Age_group")["navigation_success_rate"].mean()

print(means)
print(means.idxmax())
print("the age group that has the highest success rate is Young")

navigation_success_rate= df.groupby("device_type")["navigation_success_rate"].mean()

navigation_success_rate.plot(kind="bar")


plt.title("Average navigation success rate by device type")
plt.xlabel("device type")
plt.ylabel("average navigation success rate")

plt.savefig("average_navigation_success_rate_by_device_type.png")
plt.show()

average_session_per_week = df.groupby("vision_level")["sessions_per_week"].mean()
average_session_per_week.plot(kind = "line")
plt.title("average sessions per week by vision level")
plt.xlabel("vision level")
plt.ylabel("average sessions per week")
plt.savefig("average_sesssions_per_week_by_vision_level.png")
plt.show()



df["navigation_success_rate"].plot(kind="hist", bins=10)

plt.title("Distribution of navigation success rate")
plt.xlabel("navigation success rate")
plt.ylabel(" all users")

plt.savefig("navigation_success_rate_histogram.png")

plt.show()

print("phoneApp is currently serving user best,"
      "it might be because individuals are always on their phones in 2026. "
      "and the world is currently a global village.")
print("the device that seems to be underserved is the ARGlasses,"
      " the research team should whether the glasses fit the users eyes condition,"
      "and they also ensure that the ARGlasses are smart")
print("Understanding data is essential before building AI because it ensures data quality,"
      " identifies missing values and patterns, "
      "and helps select relevant features. "
      "Proper data analysis improves model accuracy, reduces errors, "
      "and supports informed decision-making. High-quality, "
      "well-understood data enables AI systems to produce reliable, meaningful, and trustworthy predictions.")

df["engagement_score"] = (
    df["sessions_per_week"] *
    df["navigation_success_rate"]
)
print(df.head())




top_10 = df.sort_values(by="engagement_score", ascending=False)[
    ["user_id", "sessions_per_week", "navigation_success_rate",
     "engagement_score", "device_type"]
].head(10)
print(top_10.to_string(index=False))

df["high_success"] = (df["navigation_success_rate"] >= 0.75).astype(int)
# 1 = high success, 0 = low success

# Create features
X = df.drop(columns=["user_id", "navigation_success_rate", "high_success"])

# Create target
y = df["high_success"]

# One-hot encode categorical features
# NOTE: Age_group is categorical too (created via pd.cut), so it must be
# included here — otherwise LogisticRegression chokes on string values
# like "Young" when it tries to fit.
X = pd.get_dummies(X, columns=["device_type", "vision_level", "Age_group"])
print(X.columns)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(
    y_test,
    predictions
)

print(accuracy)