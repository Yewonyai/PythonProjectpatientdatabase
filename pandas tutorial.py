
import pandas as pd
import numpy as np
df = pd.DataFrame({
    "patient_names"
    :["Anita", "Fatimah","Maryam","Abdulrahman","Ashraf","Saratu U","AG Umar","Yisa T", "Oiza", "Nene M"],
    "age":[21,20,18,19,21,61,65,50,17,14],
    "status":["uni student","uni student","uni student", "uni student", "uni student","business woman","engineer","engineer", "A level student", "secondary school student"],
    "sex":["F","F","F","M","M","F","M","M","F","F"],
    "city":["Abuja","Kaduna","minna","Minna","Kogi","kogi","Lagos","Abuja","porthacourt","Abuja"],
    "eye_condition":["short_sight","myopia","astigmatism","hypermyopia","cataract",",myopia","cataract","myopia","astigmatism","short_sight"]

})
print(df)
print(df.age)
print(df["age"].max())
print(df["age"].min())
print(df.patient_names)
print(df["patient_names"].loc[1])
print(df.loc[1,["patient_names","age","sex"]])
print(df.iloc[3],[])