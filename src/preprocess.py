import pandas as pd

original = pd.read_csv("../data/original_diabetes.csv")
original["Verdict"] = (original["Outcome"] == 1)
original = original.drop(columns = ["Outcome"])
original.to_csv("../data/diabetes.csv", index = False)
