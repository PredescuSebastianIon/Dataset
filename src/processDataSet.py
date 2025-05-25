# Import OS to create director
import os
# Import libraries
import pandas as pd
import seaborn
import matplotlib.pyplot as plt

# Folder for saving figures
FIGURES_DIR = "../figures"
if not os.path.exists(FIGURES_DIR):
	os.makedirs(FIGURES_DIR)

# Analize missing value
def missing_value_report(df, name, threshold = 50):
	missing_data = pd.DataFrame({
		"missing_count": df.isna().sum(),
		"missing_pct":   df.isna().mean() * 100
	})
	print(f"\n----------- Missing Values: {name} -----------")
	print(missing_data)
	for col, row in missing_data.iterrows():
		if row["missing_pct"] >= threshold:
			df.drop(columns = col, inplace = True)

def statistics(df, name_of_subset):
	# Distributions
	columns_names = df.select_dtypes(include = ["int", "float"]).columns
	for i in columns_names:
		figure, ax = plt.subplots()
		seaborn.histplot(df[i], kde = True, ax = ax)
		ax.set_title(f"{i} Distribution {name_of_subset}")
		figure.savefig(f"{FIGURES_DIR}/{name_of_subset}_{i}_hist.png", bbox_inches="tight")
		plt.close(figure)

	verdict_column = ["Verdict"]
	for i in verdict_column:
		figure, ax = plt.subplots()
		seaborn.countplot(x = i, data = df, ax = ax)
		ax.set_title(f"{i} Counts {name_of_subset}")
		figure.savefig(f"{FIGURES_DIR}/{name_of_subset}_{i}_count.png", bbox_inches="tight")
		plt.close(figure)

	# Outlier detection (boxplots)
	for i in columns_names:
		figure, ax = plt.subplots()
		seaborn.boxplot(x = df[i], ax = ax)
		ax.set_title(f"{i} Boxplot {name_of_subset}")
		figure.savefig(f"{FIGURES_DIR}/{name_of_subset}_{i}_box.png", bbox_inches="tight")
		plt.close(figure)

	# Correlation matrix
	corr_columns = list(columns_names) + ["Verdict"]
	corr = df[corr_columns].corr()
	figure, ax = plt.subplots()
	seaborn.heatmap(corr, annot = True, fmt = ".2f", cmap = "coolwarm", ax = ax)
	ax.set_title("Correlation Matrix {name_of_subset}")
	figure.savefig(f"{FIGURES_DIR}/{name_of_subset}_correlation_heatmap.png", bbox_inches="tight")
	plt.close(figure)

	# Relationship to target (violin plots)
	for i in ["Glucose", "BMI", "Age", "Insulin"]:
		figure, ax = plt.subplots()
		seaborn.violinplot(x = "Verdict", y = i, data = df, inner = "quartile", ax = ax)
		ax.set_title(f"{i} by Verdict")
		figure.savefig(f"{FIGURES_DIR}/{name_of_subset}_{i}_violin.png", bbox_inches="tight")
		plt.close(figure)

# Load data and process it
def load_data(train_df, test_df):
	# Missing values
	missing_value_report(train_df, "Train")
	missing_value_report(test_df, "Test")

	# Descriptive statistics
	print("\n----------- Train.describe() -----------")
	print(train_df.describe(include="all"))
	print("\n----------- Test.describe() -----------")
	print(test_df.describe(include="all"))

	# Statistics for each subset
	statistics(train_df, "Train")
	statistics(test_df, "Test")
	
	print(f"\nAll figures are saved to the '{FIGURES_DIR}/' directory")
