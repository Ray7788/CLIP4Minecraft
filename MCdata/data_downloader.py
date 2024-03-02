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


def process_json(source_json_file, output_folder, sampled_json_file, failed_json_file):
    """
    Extract video clips from the source JSON file and save them to the output folder.
    Write the list of successfully downloaded videos to the sampled JSON file.

    Args:
    - source_json_file: the path to the video source JSON file
    - output_folder: the path to the extracted video output folder
    - sampled_json_file: the path to the sampled output JSON file
    """
    with open(source_json_file, 'r') as file:
        video_data = json.load(file)
    
    successful_videos = []  # List of videos that were successfully downloaded
    failed_videos = []  # List of videos that failed to download
    sampled_json_file_name = os.path.basename(sampled_json_file)
    sampled_json_file_path = os.path.join(output_folder, sampled_json_file_name)
    failed_json_file_name = os.path.basename(failed_json_file)
    failed_json_file_path = os.path.join(output_folder, failed_json_file_name)

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

            # Write successful videos to corresponding output JSON file
            if os.path.exists(sampled_json_file_path):
                with open(sampled_json_file_path, 'a') as file:
                    file.write(',\n')
                    json.dump(successful_videos[-1], file)
            else:
                with open(sampled_json_file_path, 'w') as file:  # Create the file if it doesn't exist
                    file.write('[\n')
                    json.dump(successful_videos[-1], file)
                    file.write('\n')

        except Exception as e:
            print(f"Error downloading video {video_id}: {e}")
            failed_videos.append(video_id)
            # Write failed videos to corresponding output JSON file immediately
            if os.path.exists(failed_json_file_path):
                with open(failed_json_file_path, 'a') as file:
                    file.write(',\n')
                    json.dump(video_id, file)
            else:
                with open(failed_json_file_path, 'w') as file:  # Create the file if it doesn't exist
                    file.write('[\n')
                    json.dump(video_id, file)
                    file.write('\n')

            continue

    # Write successful videos to corresponding output JSON file
    with open(sampled_json_file_path, 'a') as file:
        file.write(']')
    with open(failed_json_file_path, 'a') as file:
        file.write(']')
    return successful_videos


def main():
    # MCdata/train_1.json
    source_json_files = ['train_4.json', 'test_4.json']  # the path to source JSON file
    sampled_json_files = ['train_4_log.json', 'test_4_log.json']  # actual selected output JSON file (after sampling)
    output_folders = ['train_4_videos', 'test_4_videos']  # the path to the output folder
    failed_json_files = ['train_4_failed.json', 'failed_test_4_failed.json']  # the path to the failed output JSON file
    all_successful_videos = []   # List of videos that were successfully downloaded
    
    print("Downloading video clips---------------------------------")
    with ThreadPoolExecutor(max_workers=len(source_json_files)) as executor:
        futures = [executor.submit(process_json, source_json_files[i], output_folders[i], sampled_json_files[i], failed_json_files[i]) for i in range(len(source_json_files))]
        
        wait(futures)  # Wait for all tasks to complete

        for future in futures:
            all_successful_videos.extend(future.result())
    print("Download complete!!!---------------------------------")


if __name__ == "__main__":
    main()
