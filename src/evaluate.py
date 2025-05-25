from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
import matplotlib.pyplot as plt

# Global dictionary to store trained models
trained_models  =  {}

# Compute classification metrics and format them into a report string
def make_report(model_name, true_Y, predicted_Y):
    # Calculate accuracy score
    accuracy  =  accuracy_score(true_Y, predicted_Y)
    # Calculate precision score
    precision  =  precision_score(true_Y, predicted_Y)
    # Calculate recall score
    recall  =  recall_score(true_Y, predicted_Y)
    # Calculate F1 score
    f1  =  f1_score(true_Y, predicted_Y)

    # Create formatted report string
    report  =  (
        f"{model_name}:\n"
        f"Accuracy: {accuracy:.2f}\n"
        f"Precision:{precision:.2f}\n"
        f"Recall: {recall:.2f}\n"
        f"F1-score: {f1:.2f}\n"
    )
    return report

# Create a confusion matrix figure for visualization
def build_cmat(name, true_Y, predicted_Y):
    # Compute confusion matrix from true and predicted labels
    cm  =  confusion_matrix(true_Y, predicted_Y)
    # Create a new matplotlib figure and axis
    figure, ax  =  plt.subplots()
    # Prepare confusion matrix display object with labels
    disp  =  ConfusionMatrixDisplay(cm, display_labels = [False, True])
    # Plot confusion matrix on the axis with a blue colormap
    disp.plot(ax = ax, cmap = plt.cm.Blues)
    # Set the plot title
    ax.set_title(name)
    # Close the figure to prevent automatic display (useful in notebooks)
    plt.close(figure)
    # Return the figure object
    return figure

# Train models on training data, evaluate on test data, and generate reports and plots
def evaluate(train_df, test_df):
    # Separate features and target from training set
    X_train  =  train_df.drop(columns = ["Verdict"])
    Y_train  =  train_df["Verdict"]
    # Separate features and target from test set
    X_test  =  test_df.drop(columns = ["Verdict"])
    Y_test  =  test_df["Verdict"]

    # Train Logistic Regression model on training data
    logistic  =  LogisticRegression().fit(X_train, Y_train)
    # Train Random Forest classifier with fixed random seed
    forest  =  RandomForestClassifier(random_state = 42).fit(X_train, Y_train)

    # Store trained models for later use
    trained_models["logistic"]  =  logistic
    trained_models["forest"]  =  forest

    # Predict test labels using Logistic Regression
    y_pred_logic  =  logistic.predict(X_test)
    # Predict test labels using Random Forest
    y_pred_forest  =  forest.predict(X_test)

    # Generate classification reports for both models
    report_logic  =  make_report("Logistic Regression", Y_test, y_pred_logic)
    report_forest  =  make_report("Random Forest", Y_test, y_pred_forest)
    # Combine reports into a single string
    combined_report  =  report_logic + "\n" + report_forest

    # Create confusion matrix plot for Logistic Regression
    figure_logic  =  build_cmat("LR Confusion Matrix", Y_test, y_pred_logic)
    # Create confusion matrix plot for Random Forest
    figure_forest  =  build_cmat("RF Confusion Matrix", Y_test, y_pred_forest)

    # Return combined report text and both confusion matrix figures
    return combined_report, figure_logic, figure_forest
