import pandas as pd
from sklearn.model_selection import train_test_split as train
import gradio as gr
# Import the functions
from preprocess import preprocess
from processDataSet import load_data
from evaluate import evaluate
# Import the models
from evaluate import trained_models

DATA_DIR = "../data"

# Function to load, preprocess, split data, save CSVs, generate EDA reports, and evaluate models
def load_and_split_data():
    # Load original data and preprocess it
    df = pd.read_csv(f"{DATA_DIR}/original_diabetes.csv")
    df = preprocess(df)

    # Split into train/test sets, 80/20 split
    train_df, test_df = train(df, test_size = 0.2, random_state = 42)

    # Save splits as CSV for reuse and reproducibility
    train_df.to_csv(f"{DATA_DIR}/train.csv", index = False)
    test_df.to_csv(f"{DATA_DIR}/test.csv", index = False)

    # Perform exploratory data analysis and generate figures
    load_data(train_df, test_df)

    # Train models, evaluate, and generate reports and confusion matrix plots
    report_text, figure_logic, figure_forest  =  evaluate(train_df, test_df)

    # Return data and evaluation outputs to be displayed in the UI
    return train_df, test_df, report_text, figure_logic, figure_forest

# Function to predict diabetes likelihood for new user input using logistic regression
def prediction(filed1, filed2, filed3, filed4, filed5, filed6, filed7, filed8):
    # Check if model has been trained
    if "logistic" not in trained_models:
        return "Please train the model first"

    logistic_model = trained_models["logistic"]

    # Construct input DataFrame from user fields
    data_about_patient = pd.DataFrame([{
        "Pregnancies": filed1,
        "Glucose": filed2,
        "BloodPressure": filed3,
        "SkinThickness": filed4,
        "Insulin": filed5,
        "BMI": filed6,
        "DiabetesPedigreeFunction": filed7,
        "Age": filed8
    }])

    # Predict verdict (0 or 1)
    verdict  =  logistic_model.predict(data_about_patient)[0]

    # Return user-friendly message
    if verdict == 0:
        return "Low chances of diabetes"
    else:
        return "High chances of diabetes. Please consult a doctor!"

# Build Gradio interface
with gr.Blocks() as interface:
    # Input fields for patient data
    with gr.Row():
        filed1 = gr.Number(label = "Pregnancies", value = 0)
        filed2 = gr.Number(label = "Glucose", value = 120)
        filed3 = gr.Number(label = "BloodPressure", value = 70)
        filed4 = gr.Number(label = "SkinThickness", value = 20)
        filed5 = gr.Number(label = "Insulin", value = 80)
        filed6 = gr.Number(label = "BodyMassIndex", value = 28.0)
        filed7 = gr.Number(label = "DiabetesPedigreeFunction", value = 0.0)
        filed8 = gr.Number(label = "Age", value = 20)

    # Button to trigger prediction
    predict_button  =  gr.Button("Enter your data")
    predict_button.click(
        fn = prediction,
        inputs = [
            filed1, 
            filed2, 
            filed3, 
            filed4, 
            filed5, 
            filed6, 
            filed7, 
            filed8
        ],
        outputs = [
            gr.Textbox(label = "Prediction", interactive = False)
        ]
    )

    gr.Markdown("----------------------------")
    gr.Markdown("Diabetes CSV Splitter")

    # Button to trigger loading, training and evaluation pipeline
    train_button  =  gr.Button("Train Model", variant = "primary")
    train_button.click(
        fn = load_and_split_data,
        inputs = [],
        outputs = [
            gr.Dataframe(label = "Train (80%)"),
            gr.Dataframe(label = "Test (20%)"),
            gr.Textbox(label = "Evaluation Metrics"),
            gr.Plot(label = "Logistic Regression Confusion"),
            gr.Plot(label = "Random Forest Confusion")
        ]
    )

# Launch Gradio app with public sharing enabled
interface.launch(share = True)
