import pandas as pd

def preprocess(original):
	modified = original.copy()
	modified["Verdict"] = (modified["Outcome"] == 1)
	modified = modified.drop(columns = ["Outcome"])
	modified.to_csv("../data/diabetes.csv", index = False)
	return modified
