import pandas as pd
import gradio as gr
from sklearn.model_selection import train_test_split

def load_and_split_data():
	df = pd.read_csv("../data/diabetes.csv")
	train_df, test_df = train_test_split(
		df,
		test_size = 0.2,
		random_state = 42,
		shuffle=True
	)
	train_df.to_csv("../data/train.csv", index = False)
	test_df.to_csv("../data/test.csv", index = True)
	return train_df, test_df

interface = gr.Interface(
	fn = load_and_split_data,
	inputs = [],
	outputs = [
		gr.Dataframe(label = "Train (80%)"),
        gr.Dataframe(label = "Test  (20%)"),
	],
	title = "Diabetes CSV Splitter",
	description = "Click to load the initial csv and split it into " \
				"2 subsets, one for training and the other for testing"
)

interface.launch()
