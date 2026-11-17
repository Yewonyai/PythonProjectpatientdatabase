import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
n=200

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
X = X.drop(columns=["avg_obstacle_detected"])

# One-hot encode categorical features
X = pd.get_dummies(X, columns=["device_type", "vision_level", "Age_group"])
print(X.columns )

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression()

logistic_model.fit(X_train, y_train)

predictions = logistic_model.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(
    y_test,
    predictions
)

print(accuracy)

from sklearn.metrics import accuracy_score, classification_report

y_pred = logistic_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print(classification_report(y_test, y_pred))

print("Precision measures how accurate your positive predictions are."
"Suppose your model predicts 100 success rate."
"But when we check their actual results, only 20 actual success rate."
"Then:"
      "Precision = 20 / 100 = 20%",
      "in the context of my navigation assistant precision is 1.00,"
   
      "this is a high precision rate, mneaning the model did not make any false positive prediction ")

print("recall:Recall = How many of the actual positive cases did my model find?"
"High recall means the model doesn't miss many false negatives."
      "class 0 recal is 1.00 thats 100% recall and class 1 recall is 0.79 thats is 79% recall"
      "High recall means the model doesn't miss many false negatives."
      "79% means fairly high and the model mostly did not miss many false negatives in the navigation success rate b")
print("F1-score combines precision and recall into one measurement."
      "the f1-score is 0.88 that is 88% this is a balanced model ")

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# ORIGINAL MODEL
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

# Predictions from original model
y_pred_lr = logistic_model.predict(X_test)

# SECOND MODEL
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree .fit(X_train, y_train)

# Predictions from Decision Tree
y_pred_dt = decision_tree .predict(X_test)


print("Logistic Regression Accuracy:",
      accuracy_score(y_test, y_pred_lr))

print("\nLogistic Regression Report:")
print(classification_report(y_test, y_pred_lr))


print("Decision Tree Accuracy:",
      accuracy_score(y_test, y_pred_dt))

print("\nDecision Tree Report:")
print(classification_report(y_test, y_pred_dt))

print("Logistic Regression Accuracy: 0.925"
      "and,"
      "Decision Tree Accuracy: 0.85"
      "Logistic Regression Accuracy performed better, and was more accurate with a 93% accuracy"
      "than decision tree with accuracy report of 85% accuracy "
      "Decision Trees tend to overfit — they memorize the training data too precisely and struggle with new data."
      " With only 200 users and relatively few features, Logistic Regression's simpler linear boundary generalised better."
      " If we had tens of thousands of users and more complex patterns, the result might flip.")


print("Classification predicts a category — high success or low success, yes or no."
      " Regression predicts a continuous number — like predicting the exact navigation success rate value (0.73, 0.91 etc). "
      "This week we used classification because we only needed to know which group a user falls into, "
      "not the exact number.")

print("Q2: Why do we split into training and testing sets?"

"If you train and test on the same data," 
""
      "your model memorizes the answers instead of learning patterns."
      " It would score perfectly on data it has seen but fail completely on new users. "
      "The test set simulates new users the model has never encountered — giving you an honest measure of real performance.")

print("Q3: High precision but low recall — what does it mean for our users?"

"This is the most important question for a visually impaired individual  specifically."
      " High precision means when the model says a user will succeed, "
      "it's almost always right. "
      "Low recall means the model misses many users who would actually benefit — it labels them as low success and they never get properly supported."
      " In assistive technology, low recall is dangerous. It means real people who need help are being overlooked by the system.")


#smarter model
# Separate features and target
X = df.drop(
    columns=["user_id", "navigation_success_rate", "high_success"]
)

y = df["high_success"]

# Find categorical columns
categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

# Encode categorical columns
X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True
)

X.head()
# Separate features and target





from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42

)

logistic_model= LogisticRegression(max_iter=1000)
decision_tree  = DecisionTreeClassifier(random_state=42)
random_forest = RandomForestClassifier(random_state =42)
logistic_model.fit(X_train, y_train)
decision_tree .fit(X_train, y_train)
random_forest.fit(X_train, y_train)

#make predictions
y_pred_lr = logistic_model.predict(X_test)
y_pred_dt = decision_tree .predict(X_test)
forest_pred = random_forest.predict(X_test)


#accuracy


print("logistic regression accuracy:")
print(accuracy_score,(y_test,y_pred_lr ))

print("\n decision tree accuracy :")
print(accuracy_score,(y_test,y_pred_dt))

print("Random forest tree accuracy:")
print(accuracy_score,(y_test,forest_pred))

#classification

print("\n===== Logistic Regression =====")
print(classification_report(y_test, y_pred_lr))

print("\n======  Decision Tree accuracy=====")
print(classification_report,(y_test, y_pred_dt))

print("\n===== Random Forest=====")
print(classification_report,(y_test,forest_pred))