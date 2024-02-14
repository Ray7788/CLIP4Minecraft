import yt_dlp
from yt_dlp.utils import download_range_func
import json
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, wait

import subprocess
import sys
# import ffmpeg


def download_video_clip(video_id, start_time, end_time, width, height, output_folder):
    """
    Download a video clip from YouTube based on the video ID and the start and end time.
    """
    os.makedirs(output_folder, exist_ok=True)

    ydl_opts = {
        'format': 'bestvideo[height<={}][width<={}][filesize<250MB]'.format(height+200,width+200),
        'quiet': False,
        'outtmpl': os.path.join(output_folder, '{}.mp4'.format(video_id)),
        'merge_output_format': 'mp4',
        'download_ranges': download_range_func(None, [(start_time, end_time)]),
        'force_keyframes_at_cuts': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        video_url = 'https://www.youtube.com/watch?v={}'.format(video_id)
        ydl.download([video_url])


def process_json(source_json_file, output_folder, sampled_json_file):
    """
    Extract video clips from the source JSON file and save them to the output folder.
    """
    with open(source_json_file, 'r') as file:
        video_data = json.load(file)
    
    successful_videos = []  # List of videos that were successfully downloaded
    for entry in video_data:
        video_id = entry['vid']
        start_time = entry['begin position']
        end_time = entry['end position']
        
        try:
            download_video_clip(video_id, start_time, end_time, width=256, height=160, output_folder=output_folder)
            successful_videos.append({
                'vid': video_id,
                'transcript clip': entry['transcript clip']
            })
        except Exception as e:
            print(f"Error downloading video {video_id}: {e}")
            continue

    # Write successful videos to corresponding output JSON file
    with open(sampled_json_file, 'w') as file:
        json.dump(successful_videos, file, indent=4)

    # return successful_videos


def main():
    # MCdata/train_1.json
    source_json_files = ['train_1.json', 'test_1.json']  # the path to source JSON file
    sampled_json_files = ['train.json', 'test.json']  # actual selected output JSON file
    output_folders = ['train_videos', 'test_videos']  # the path to the output folder
    
    all_successful_videos = []   # List of videos that were successfully downloaded

    with ThreadPoolExecutor(max_workers=len(source_json_files)) as executor:
        futures = [executor.submit(process_json, source_json_files[i], output_folders[i], sampled_json_files[i]) for i in range(len(source_json_files))]
        
        wait(futures)  # Wait for all tasks to complete

        for future in futures:
            all_successful_videos.extend(future.result())

if __name__ == "__main__":
    main()
