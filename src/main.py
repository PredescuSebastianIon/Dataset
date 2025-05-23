import pandas as pd
import gradio as gr

def load_data():
	df = pd.read_csv("../data/diabetes.csv")
	df.drop(columns=["outcome"])
	return df

interface = gr.Interface(
	fn = load_data,
	inputs = [],
	outputs = "dataframe"
)

interface.launch()
