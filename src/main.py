import pandas as pd
import gradio as gr
from sklearn.model_selection import train_test_split
from preprocess import preprocess
from processDataSet import load_data

def load_and_split_data():
	df = pd.read_csv("../data/original_diabetes.csv")
	df = preprocess(df)
	train_df, test_df = train_test_split(
		df,
		test_size = 0.2,
		random_state = 42,
		shuffle=True
	)
	train_df.to_csv("../data/train.csv", index = False)
	test_df.to_csv("../data/test.csv", index = True)
	load_data(train_df, test_df)
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

interface.launch(share = True)
