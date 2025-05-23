from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
import matplotlib.pyplot as plt

trained_models = {}

# Compute metrics
def make_report(model_name, true_Y, predicted_Y):
    accuracy = accuracy_score(true_Y, predicted_Y)
    precision = precision_score(true_Y, predicted_Y)
    recall = recall_score(true_Y, predicted_Y)
    f1 = f1_score(true_Y, predicted_Y)
    # The report
    report = (
        f"{model_name}:\n"
        f"Accuracy: {accuracy:.2f}\n"
        f"Precision:{precision:.2f}\n"
        f"Recall: {recall:.2f}\n"
        f"F1-score: {f1:.2f}\n"
    )
    return report

# Build confusion-matrix figures
def build_cmat(name, true_Y, predicted_Y):
	cm = confusion_matrix(true_Y, predicted_Y)
	figure, ax = plt.subplots()
	disp = ConfusionMatrixDisplay(cm, display_labels=[False, True])
	disp.plot(ax=ax, cmap=plt.cm.Blues)
	ax.set_title(name)
	plt.close(figure)
	return figure

def evaluate(train_df, test_df):
	X_train = train_df.drop(columns=["Verdict"])
	Y_train = train_df["Verdict"]
	X_test = test_df.drop(columns=["Verdict"])
	Y_test = test_df["Verdict"]

	# Fit models
	logistic = LogisticRegression().fit(X_train, Y_train)
	forest = RandomForestClassifier(random_state = 42).fit(X_train, Y_train)

	# Save trained models
	trained_models["logistic"] = logistic
	trained_models["forest"] = forest

	# Predict
	y_pred_logic = logistic.predict(X_test)
	y_pred_forest = forest.predict(X_test)

	report_logic = make_report("Logistic Regression", Y_test, y_pred_logic)
	report_forest = make_report("Random Forest", Y_test, y_pred_forest)
	combined_report = report_logic + "\n" + report_forest

	# Create the figures
	figure_logic = build_cmat("LR Confusion Matrix", Y_test, y_pred_logic)
	figure_forest = build_cmat("RF Confusion Matrix", Y_test, y_pred_forest)

	# Return the text report and the two figures
	return combined_report, figure_logic, figure_forest
