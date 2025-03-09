# Importing the necessary Python libraries
import os
import json
import time
import yaml
import traceback
from datetime import datetime

import polars as pl

import mlx_whisper
import requests

def format_episode_number(episode_num, episode_type):
    '''
    Format episode number based on episode type

    Inputs:
        - episode_num: Episode number
        - episode_type: Type of episode ('main', 'patreon', 'movie-night')

    Returns:
        - Formatted episode number
    '''
    if episode_type == 'patreon':
        # Use zfill for patreon episodes which might have non-standard numbering
        return str(episode_num).zfill(3)
    else:
        # Use standard formatting for main and movie-night episodes
        return f"{episode_num:03d}"

def download_and_transcribe_episode(episode, transcript_dir, episode_type):
    '''
    Downloads and transcribes a single episode, handling errors and metadata creation.
    
    Inputs:
        - episode: Dictionary containing episode information
        - transcript_dir: Directory to save transcripts
        - episode_type: Type of episode ('main', 'patreon', 'movie-night')

    Returns:
        - N/A
    '''
    # Ensure the transcript directory exists
    os.makedirs(transcript_dir, exist_ok=True)
    
    # Format episode number based on episode type
    episode_num_formatted = format_episode_number(episode['episode_num'], episode_type)
    
    # Setting the file paths
    episode_transcript_filepath = os.path.join(transcript_dir, f"episode_{episode_num_formatted}.txt")
    episode_transcript_metadata_filepath = os.path.join(transcript_dir, f"episode_{episode_num_formatted}.txt.metadata.json")
    episode_audio_filepath = os.path.join(transcript_dir, f"episode_{episode_num_formatted}.mp3")

    # Checking if the episode transcript file exists
    if os.path.exists(episode_transcript_filepath):
        print(f"Skipping {episode_type} episode {episode_num_formatted} - transcript already exists")
        return
    
    print(f"Processing {episode_type} episode {episode_num_formatted}: {episode['title']}")

    # Attempting to download and transcribe the audio
    try:
        # Downloading the audio file for the episode
        response = requests.get(episode['link'], stream=True)

        # Writing the audio file to the disk
        with open(episode_audio_filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)

        # Transcribing the audio file with the MLX Whisper API
        transcribed_text = mlx_whisper.transcribe(
            episode_audio_filepath, 
            path_or_hf_repo='mlx-community/whisper-large-v3-turbo'
        )['text']

        # Writing the transcribed text to the episode transcript file
        with open(episode_transcript_filepath, 'w') as f:
            f.write(transcribed_text)

        # Deleting the audio file
        os.remove(episode_audio_filepath)
        
        # Forming the metadata content
        episode_metadata = {
            'metadataAttributes': {
                'episode_title': episode['title'],
                'episode_summary': episode['summary'],
                'episode_num': episode['episode_num'],
                'episode_upload_date': episode['timestamp']
            }
        }

        # Writing the metadata content to the episode transcript metadata file
        with open(episode_transcript_metadata_filepath, 'w') as f:
            json.dump(episode_metadata, f, indent=4)
            
        print(f"Successfully transcribed {episode_type} episode {episode_num_formatted}")

    except Exception as e:
        # Log the error details to a file
        error_log_path = '../data/logs/transcript_errors.log'
        os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
        
        with open(error_log_path, 'a') as log_file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_file.write(f"[{timestamp}] Error processing episode {episode['episode_num']}: {str(e)}\n")
            log_file.write(f"Episode link: {episode['link']}\n")
            log_file.write(f"Traceback: {traceback.format_exc()}\n\n")
        
        print(f"Error processing episode {episode['episode_num']}. See log file for details.")

        if os.path.exists(episode_audio_filepath):
            os.remove(episode_audio_filepath)

# Loading in the files containing information about the respective episode types
df_wc_public_episodes = pl.read_csv('../data/episode-metadata/wc_public_episodes.csv')
df_wc_patreon_episodes = pl.read_csv('../data/episode-metadata/wc_patreon_episodes.csv')
df_wc_movie_episodes = pl.read_csv('../data/episode-metadata/wc_movie_night_episodes.csv')

# Setting the transcription directory
wc_transcript_dir = '../data/transcripts/'

# Iterating over all the episodes in the public episode metadata DataFrame
for episode in df_wc_public_episodes.iter_rows(named=True):
    download_and_transcribe_episode(episode, os.path.join(wc_transcript_dir, 'main'), 'main')

# Iterating over all the episodes in the Patreon episode metadata DataFrame
for episode in df_wc_patreon_episodes.iter_rows(named=True):
    download_and_transcribe_episode(episode, os.path.join(wc_transcript_dir, 'patreon'), 'patreon')

# Iterating over all the episodes in the movie night episode metadata DataFrame
for episode in df_wc_movie_episodes.iter_rows(named=True):
    download_and_transcribe_episode(episode, os.path.join(wc_transcript_dir, 'movie-night'), 'movie-night')