import yt_dlp
from yt_dlp.utils import download_range_func
import json
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import subprocess
import sys
# import ffmpeg


def download_video_clip(video_id, start_time, end_time, width, height, output_folder='train_videos'):
    """
    Download a video clip from YouTube and save it to the specified folder.
    """
    os.makedirs(output_folder, exist_ok=True)

    ydl_opts = {
        'format': 'bestvideo[height={}][width={}][filesize<50MB]'.format(height,width),
        'quiet': False,
        'outtmpl': os.path.join(output_folder, '{}.mp4'.format(video_id)),
        'merge_output_format': 'mp4',
        'download_ranges': download_range_func(None, [(start_time, end_time)]),
        'force_keyframes_at_cuts': True
        # 'download_ranges': f'{start_time}-{end_time}',
        # 'postprocessors': [{
        #     'key': 'FFmpegVideoConvertor',
        #     'preferedformat': 'mp4',}],
        # 'external_downloader': 'ffmpeg',
        # 'external_downloader_args': ['-ss', str(start_time), '-to', str(end_time)],
        # 'external-downloader-args': ['-ss', str(start_time), '-to', str(end_time)],
        # 'postprocessor_args': [
        #     '-ss', str(start_time),
        #     '-to', str(end_time),
        # ],
    }

    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        video_url = 'https://www.youtube.com/watch?v={}'.format(video_id)
        info_dict = ydl.extract_info(video_url, download=False)

        # Check if the video has subtitles
        if 'automatic_captions' not in info_dict or not info_dict['automatic_captions']:
            print("Skipping video {} - No automatic captions available.".format(video_id))
            return

        # Create a video clip using the specified time range
        start_time = max(start_time, 0)
        end_time = min(end_time, info_dict['duration'])
        if start_time >= end_time:
            print("Skipping video {} - Invalid time range.".format(video_id))
            return
        ydl.download([video_url])


# def cut_video(video_id, start_time, end_time, output_folder):
#     # 
#     input_file = os.path.join(output_folder, f'{video_id}.mp4')
#     output_file = os.path.join(output_folder, f'{video_id}.mp4')

#     subprocess.run([
#         'ffmpeg',
#         '-i', input_file,
#         '-ss', str(start_time),
#         '-to', str(end_time),
#         '-c', 'copy',
#         output_file,
#     ])


def main():
    source_json_file = 'example.json'  # the path to source JSON file
    sampled_json_file = 'output.json'   # actual output JSON file

    with open(source_json_file, 'r') as file:
        video_data = json.load(file)
    
    successful_videos = []
    for entry in video_data:
        video_id = entry['vid']
        start_time = entry['begin position']
        end_time = entry['end position']
        
        try:
            download_video_clip(video_id, start_time, end_time, width=256, height=160,output_folder='train_videos')
            # If the download is successful, add the video to the successful_videos list
            successful_videos.append({
                'vid': video_id,
                'transcript clip': entry['transcript clip']
            })
        except Exception as e:
            print(f"Error downloading video {video_id}: {e}")
            continue
        # cut_video(video_id, start_time, end_time, output_folder='train_videos')

    # Write the successful videos to the new JSON file
    with open(sampled_json_file, 'w') as file:
        json.dump(successful_videos, file, indent=4)

if __name__ == "__main__":
    main()
