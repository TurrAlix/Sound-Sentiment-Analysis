import numpy as np
import matplotlib.pyplot as plt
import json
import whisper
import plotly.graph_objects as go
import librosa
import soundfile as sf
import random
from IPython.display import Audio
from plotly.subplots import make_subplots
import noisereduce as nr
import os

class Plots:
    @staticmethod
    def plt_freq_labels(train_df, val_df):
        train_counts = train_df['label'].value_counts().sort_index()
        val_counts = val_df['label'].value_counts().sort_index()

        # Ensure all labels (0,1,2) are present
        all_labels = [0, 1, 2]
        train_counts = train_counts.reindex(all_labels, fill_value=0)
        val_counts = val_counts.reindex(all_labels, fill_value=0)

        # Map labels to sentiment categories
        label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        label_names = [label_map[label] for label in all_labels]

        # Create subplots
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Training Set", "Validation Set"))
        # Training set bar chart
        fig.add_trace(go.Bar(
            x=label_names,
            y=train_counts,
            marker_color="skyblue",
            name="Training Set",
            width=[0.4] * len(label_names)  # Reduce bar width
        ), row=1, col=1)
        # Validation set bar chart
        fig.add_trace(go.Bar(
            x=label_names,
            y=val_counts,
            marker_color="salmon",
            name="Validation Set",
            width=[0.4] * len(label_names)  # Reduce bar width
        ), row=1, col=2)
        # Update layout
        fig.update_layout(
            title="Frequency of Labels in Training and Validation Data",
            xaxis_title="Labels",
            yaxis_title="Frequency",
            xaxis=dict(tickmode="array", tickvals=label_names),
            template="plotly_white", 
            showlegend=False, 
            bargap=0.01,  # Reduce space between bars, 
            bargroupgap=0.0, # Reduce spacing between bars inside each subplot
            height=400,
            width=900
        )   
        fig.show()

    @staticmethod
    def plot_count_words(df):
        word_counts = df['text'].apply(lambda x: len(x.split()))
        print(df[word_counts <= 3][['text', "wav_id"]])

        max_word_count_corpus = word_counts.max()
        average_word_count_corpus = round(word_counts.mean(), 2)
        print(f"Max word count: {max_word_count_corpus}")
        print(f"Average word count: {average_word_count_corpus}")

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=word_counts,
            nbinsx=50,
            marker_color='blue',
            name="Word Count"
        ))
        fig.update_layout(
            title_text="Word Count Distribution",
            showlegend=False,
            height=400,
            width=800
        )
        fig.update_xaxes(title_text="Word Count")
        fig.update_yaxes(title_text="Frequency")
        fig.show()

    @staticmethod
    def plot_duration(df_train, df_test, df_val):
        # Create subplots in a single row
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=["Train Dataset", "Test Dataset", "Validation Dataset"]
        )

        # Add histograms for each dataset in the subplots
        fig.add_trace(go.Histogram(x=df_train['duration'], nbinsx=50, opacity=0.7, name='Train', marker=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Histogram(x=df_test['duration'], nbinsx=50, opacity=0.7, name='Test', marker=dict(color='green')), row=1, col=2)
        fig.add_trace(go.Histogram(x=df_val['duration'], nbinsx=50, opacity=0.7, name='Validation', marker=dict(color='red')), row=1, col=3)

        # Update layout to squeeze them together
        fig.update_layout(
            title='Distribution of Audio Durations',
            showlegend=True,
            height=400,
            barmode='overlay',
            xaxis_title='Duration (seconds)',
            yaxis_title='Frequency',
            yaxis=dict(showgrid=True, range=[0, 2500], dtick=500),  # Adjust y-axis for the first plot (Train)
            xaxis=dict(tickmode='linear'),
        )

        # Update individual y-axes for each subplot
        max_freq_train = df_train['duration'].value_counts().max()
        print(df_train['duration'].value_counts())
        max_freq_test = df_test['duration'].value_counts().max()
        max_freq_val = df_val['duration'].value_counts().max()
        fig.update_yaxes(range=[0, max_freq_train], row=1, col=1)  # Train dataset
        fig.update_yaxes(range=[0, 40], row=1, col=2)  # Test dataset
        fig.update_yaxes(range=[0, 170], row=1, col=3)  # Validation dataset

        # Show the plot
        fig.show()

class Audio2Text:
    def __init__(self):
        """
        Initialize the dataset manager with the path to the data directory.
        """
        self.model = whisper.load_model("tiny", device="cuda")

    def transcribe_file(self, file_path):
        """
        Transcribe a single audio file using Whisper.
        """
        print("  Processing ", file_path)
        result = self.model.transcribe(file_path)
        transcription = result["text"]
        return {file_path: transcription}

    def save_checkpoint(self, data, checkpoint_file):
        """
        Save the transcriptions data to a checkpoint file.
        """
        with open(checkpoint_file, 'w') as f:
            json.dump(data, f)

    def load_checkpoint(self, checkpoint_file):
        """
        Load previous transcriptions from a checkpoint file, if it exists.
        """
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
            print(f"\tLoaded {len(data)} previous transcriptions.")
            return data
        except FileNotFoundError:
            print("\tNo previous checkpoint found, starting from scratch.")
            return {}

    def process_batch(self, batch_files, transcriptions):
        """
        Process a batch of audio files and return their transcriptions.
        """
        print(f"Processing batch:")  
        batch_transcriptions = [self.transcribe_file(file) for file in batch_files]
        
        for batch_dict in batch_transcriptions:
            transcriptions.update(batch_dict)
        return transcriptions

    def process_audio_files(self, audio_files, batch_size, checkpoint_file):
        """
        Process the audio files in batches, transcribing them and saving progress to a checkpoint.
        """
        transcriptions = self.load_checkpoint(checkpoint_file)
        for i in range(0, len(audio_files), batch_size):
            batch_files = audio_files[i:i + batch_size]
            batch_files = [file for file in batch_files if file not in transcriptions]

            if batch_files:
                transcriptions = self.process_batch(batch_files, transcriptions)
                # Save the checkpoint after each batch
                self.save_checkpoint(transcriptions, checkpoint_file)
                print(f"\tCheckpoint saved! It contains {len(transcriptions)} transcriptions so far.")
        
        return transcriptions

    def transcribe_audio_files(self, df, batch_size, checkpoint_file, name):
        """
        Transcribe the audio files in the training dataset and add the transcriptions to the DataFrame.
        """
        audio_files = df['audio_path'].tolist()
        transcriptions = self.process_audio_files(audio_files, batch_size, checkpoint_file)
        df['text'] = df['audio_path'].map(transcriptions)
        df = process_text(df)
        df.to_excel(f"../Dataset/{name}_dataset.xlsx", index=False)
        return df

# Function to remove illegal characters from the text
def clean_text(text):
    if isinstance(text, str):
        return "".join(c for c in text if c.isprintable())  
    return text  

def process_text(df):
    df['text'] = df['text'].apply(clean_text)
    df['text'] = df['text'].str.lower()
    df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True) 
    df['text'] = df['text'].str.replace('-', ' ') 
    df['text'] = df['text'].fillna('')
    return df


def data_get_info(df, wav_id=None):
    if wav_id == None:
        wav_id = random.randint(0, len(df) - 1)

    path_audio = df.loc[df['wav_id'] == wav_id, 'audio_path'].values[0]
    text = df.loc[df['wav_id'] == wav_id, 'text'].values[0]
    label = df.loc[df['wav_id'] == wav_id, 'label'].values[0]

    print(f"Audio {wav_id} \tLabel for the audio: {label} \n\tText: {text}")
    return Audio(filename=path_audio)

def get_audio_duration(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    return round(librosa.get_duration(y=y, sr=sr),3)

def compute_duration(df_train, df_val, df_test):
    df_train['duration'] = df_train['audio_path'].apply(get_audio_duration)
    df_val['duration'] = df_val['audio_path'].apply(get_audio_duration)
    df_test['duration'] = df_test['audio_path'].apply(get_audio_duration)

def calculate_duration_statistics(df):
    mean_duration = round(df['duration'].mean(),2)
    median_duration = df['duration'].median()
    min_duration = df['duration'].min()
    max_duration = df['duration'].max()
    return mean_duration, median_duration, min_duration, max_duration

def add_sample_rate(df):
    for idx, row in df.iterrows():
        audio_path = row['audio_path']
        _, sr = librosa.load(audio_path, sr=None)
        df.at[idx, 'sampling_rate'] = sr
        print(f"Sampling rate of {audio_path}: {sr} Hz - {df.at[idx, 'sampling_rate']} Hz")


def slow_down(df):
	count =0 
	for idx, row in df.iterrows():
		if row['duration'] < 2:
			count +=1
			print(f"{row['audio_path']} needs to slow down: \t(old duration: {row['duration']})")
			# Slow down the audio file
			y, sr = librosa.load(row['audio_path'], sr=None, mono=True)
			y_slow = librosa.effects.time_stretch(y, rate=0.75)
			# Overwrite the audio file with the new one slowed down
			sf.write(row['audio_path'], y_slow, sr)
			# Compute new duration and save it in the dataframe
			new_duration = get_audio_duration(row['audio_path'])
			df.at[idx, 'duration'] = new_duration
			print(f"\t new duration: {new_duration}")
	print(f"Total files slowed down: {count}")

def slow_down_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    y_slow = librosa.effects.time_stretch(y, rate=0.75)
    # Overwrite the audio file with the new one slowed down
    sf.write(audio_path, y_slow, sr)

def slow_down_df(df):
	count =0 
	for idx, row in df.iterrows():
		if row['duration'] < 2:
			count +=1
			print(f"{row['audio_path']} needs to slow down: \t(old duration: {row['duration']})")
			# Slow down the audio file
			slow_down_audio(row['audio_path'])
			# Compute new duration and save it in the dataframe
			new_duration = get_audio_duration(row['audio_path'])
			df.at[idx, 'duration'] = new_duration
			print(f"\t new duration: {new_duration}")
	print(f"Total files slowed down: {count}")

def denoise_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    # Apply basic noise reduction
    y_denoised = nr.reduce_noise(y=y, sr=sr)
    # Overwrite the audio file with the new one denoised
    sf.write(audio_path, y_denoised, sr)

def denoise_df(df):
    for _, row in df.iterrows():
        audio_path = row["audio_path"]
        # overwrite the audio file with the denoised one
        denoise_audio(audio_path)

def add_text(df, checkpoint_file):
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"The checkpoint file {checkpoint_file} does not exist.")
    else:   
        if 'text' not in df.columns:
            print("The DataFrame does not contain yet a 'text' column.")
        else:
             print("The DataFrame already contains a 'text' column. I'll delete it and add the new one.")
             df.drop(columns=['text'], inplace=True)          
        
        with open(checkpoint_file, 'r') as f:
            text_ckpt_data = json.load(f)
            df['text'] = df['audio_path'].map(text_ckpt_data)
    df = process_text(df)
    return df