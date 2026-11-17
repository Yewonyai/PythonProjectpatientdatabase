

import numpy as np
import pandas as pd

np.random.seed(7)
temps = np.random.uniform(low=15.0, high=45.0, size=(30, 5))


indices = np.random.choice(temps.size, size = 10, replace = False)
temps.flat[indices] = np.nan

df = pd.DataFrame(temps)

df.columns = [f"sensor_{i}]" for i, sensor in enumerate (df.columns, start = 1)]
df.index = (f"day_{i}" for i, day in enumerate(df.index, start = 1))
print(f"", df)

print(f"" ,df.info())

print(f"", df.describe())
print(f"numpy did not show which data type we were working with without being asked directly using functions "
      f"numpy could not reveal the mean, count, std,min, percentile, "
      f"max, without each individual function being used "
      f"numpy could not reveal the memory usage"
      f"numpy did not show the rows and columns we were working with and which sensors had null and,"
      f"the data types."
      )

print(f"",df.isnull().sum())

df  = df.fillna(df.mean(numeric_only = True))
print(df)
print(f"Only two lines of codes compared to numpy, which needed multiple lines of codes ")

new_df = df.dropna()
days_lost = len(df) - len(new_df)
print(f" Days lost:", days_lost)
print(f" the days i lost are :")
df["daily_avg"] = df.mean(axis = 1)
print(df)

df["status"] = np.where(
    df["daily_avg"] > 35,
    "ALERT",
    "NORMAL"
)
print(df)

df_sorted = df.sort_values(by="daily_avg", ascending=False)
print(df_sorted)
print(f"the overall hostest day was day_28, with sensor reading 36.156204")

print(f"it felt easier to be able to use shorter lines of codes in pandas compared to numpy,"
      f"one pandas method could perform many operations and it is way faster than numpy"
      f"pandas has easier readability as well. these are so far my observations "
      f"it felt harder to grasp some concepts and how to use some methods and functions, "
      f"i look forward to different experiences and challenges. since that the only way to grow.")



print(f"labels location, lets say i want to change the label from day_1 to DAY_1,"
      f"  there will be need to use i.loc")



print(f" for df.iloc, it is used to find integer location, "
      f"for example if i want to get the row at post 1, i use the above code")


df["day_num"] = range(1, len(df)+1)
print(df[["day_num", "daily_avg"]].head())
df["week"] = pd.cut(
    df["day_num"],
    bins=[0,7,15,23,30],
    labels=["week 1","week 2","week 3", "week 4"]
)
print(df[["day_num", "week"]].head(10))

weekly_avg = df.groupby("week")["daily_avg"].mean()
print(f"",weekly_avg)
print(f"",weekly_avg.idxmax())
print(f"",weekly_avg.max())
