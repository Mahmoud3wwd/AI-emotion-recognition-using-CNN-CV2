import pandas as pd
import os
import matplotlib.pyplot as plt

df = {"name": [], "path": [], "length": [], "type": []}
path = r"images3/images"

for paths, subdir, files in os.walk(path):
    for directory in subdir:
        path = os.path.join(paths, directory)
        length = len(list(os.listdir(path)))
        df["name"].append(directory)
        df["path"].append(path)
        df["length"].append(length)
        if "validation" in paths:
            df["type"].append("validation")
        else:
            df["type"].append("train")

df = pd.DataFrame(data=df)
df = df.drop([0, 1])

train = df[df["type"] == "train"]
print(train)
validation = df[df["type"] == "validation"]
print(validation)

length = list(train["length"])
print(length)
print(sum(length))

# from this histogram there is bias in the dataset for the "happy" class
plt.hist(x=length, bins=10)
plt.show()
df.to_csv("info.csv")

