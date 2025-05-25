import pandas as pd

def preprocess(original):
    # Create a copy of the original DataFrame to avoid modifying it directly
    modified = original.copy()
    
    # Create a new boolean column 'Verdict' where True indicates diabetes (Outcome == 1)
    modified["Verdict"] = (modified["Outcome"] == 1)
    
    # Drop the original 'Outcome' column as it's replaced by 'Verdict'
    modified = modified.drop(columns=["Outcome"])
    
    # Save the processed DataFrame to a CSV file for later use
    modified.to_csv("../data/diabetes.csv", index=False)
    
    # Return the modified DataFrame
    return modified
