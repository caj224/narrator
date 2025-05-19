# Use:
# cd C:\Users\DonsLaptop\Desktop\narrator
# venv\Scripts\Activate
# presets.bat
#
# Add a folder inside this repo, with the input video.
#
# python narrator_video.py input_videos\chase.mov 
# ^ the __main__ below has chunks of code to (un)comment as you work through the process

# Folder
import sys
folder = sys.argv[1]

portrait = True 

# imports 

from PIL import Image
import numpy as np
import cv2
import time
import os
import shutil
from PIL import Image
import numpy as np
import csv
from openai import OpenAI
import base64
import json
import time
import simpleaudio as sa
import errno
from elevenlabs import generate, play, set_api_key, voices

csv_filename = f'{folder}/video_script.csv'

def extract_frames(video_path, interval=25, max_size=250):
    """
    Extract frames from the video at the specified interval and resize them.
    
    :param video_path: Path to the video file.
    :param interval: Time interval in seconds between frames.
    :param max_size: The maximum size of the longer dimension of the resized image.
    :return: None
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video file")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # No more frames or error

        if frame_count % frame_interval == 0:
            # Resize the frame
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ratio = max_size / max(pil_img.size)
            new_size = tuple([int(x*ratio) for x in pil_img.size])
            resized_img = pil_img.resize(new_size, Image.LANCZOS)
            frame = cv2.cvtColor(np.array(resized_img), cv2.COLOR_RGB2BGR)

            # Save the resized frame
            frame_path = f'{folder}/frame_{str(frame_count).zfill(7)}.jpg'
            cv2.imwrite(frame_path, frame)

        frame_count += 1

    cap.release()

client = OpenAI()

set_api_key(os.environ.get("ELEVENLABS_API_KEY"))

def encode_image(image_path):
    while True:
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except IOError as e:
            if e.errno != errno.EACCES:
                # Not a "file in use" error, re-raise
                raise
            # File is being written to, wait a bit and retry
            time.sleep(0.1)


def save_audio(text,file_path,folder=folder):
    audio = generate(text, voice=os.environ.get("ELEVENLABS_VOICE_ID"))    
    with open(file_path, "wb") as f:
        f.write(audio)

def generate_new_line(base64_image):
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                },
            ],
        },
    ]


def analyze_image(base64_image, script):
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "system",
                "content": """
                You are Sir David Attenborough. Narrate the picture of the animal as if it is a nature documentary.
                Make it snarky and funny. Don't repeat yourself. Make it short. If it does anything remotely interesting, make a big deal about it!
                """,
            },
        ]
        + script
        + generate_new_line(base64_image),
        max_tokens=500,
    )
    response_text = response.choices[0].message.content
    return response_text
    
if __name__ == '__main__':    
    if len(sys.argv) < 2:
        print("Usage: python narrator_video.py <folder_with_video> <filename>")
    else:
        video_path = os.path.join(sys.argv[1],sys.argv[2])
        
        # EXTRACT PICTURES TO GET NARRATION OF
        
        #print("Extracting photos from video")
        #if os.path.exists(folder):
        #    shutil.rmtree(folder)
        #os.makedirs(folder, exist_ok=True)
        #extract_frames(video_path)
        
        # GET SCRIPT OF NARRATION (from gpt)
        
        #print("Getting narration script")
        #gpt_message_history = [] 
        #video_script = [] # elements are [frame_number,narration]
        #
        #frame_fnames = sorted(os.listdir(folder))
        #for frame_fname in frame_fnames:
        #    frame_number = int(frame_fname.split('_')[1].split('.')[0])
        #    print(frame_number, frame_fname)
        #
        #    # getting the base64 encoding
        #    base64_image = encode_image(os.path.join(folder,frame_fname))
        #
        #    # analyze picture to get narration 
        #    print("👀 David is watching...")
        #    narration = analyze_image(base64_image, script=gpt_message_history)
        #    print(narration)
        #    
        #    # add to gpt conversation
        #    gpt_message_history = gpt_message_history + [{"role": "assistant", "content": narration}]    
        #    
        #    # extend video script 
        #    video_script.append([frame_number,narration])
        #
        ## Save video_script as csv
        #with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        #    csvwriter = csv.writer(csvfile)
        #    # Write the headers
        #    csvwriter.writerow(['Frame Number', 'Narration'])
        #    # Write the data
        #    for script_line in video_script:
        #        csvwriter.writerow(script_line)
        
        # GET AUDIO OF NARRATION (from elevenlabs)
        
        print("Getting audio")
        video_script = []
        with open(csv_filename, 'r', encoding='utf-8') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                video_script.append(row)        
        
        for frame_count, narration in video_script:
            print(frame_count, narration)
            file_path = f'{folder}/narration_{str(frame_count).zfill(7)}.wave'
            save_audio(narration,file_path)

        # OVERLAY NARRATION 
        # video_script.csv is key input here!

        from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip, concatenate_videoclips

        # Load the input video
        video = VideoFileClip(video_path)
        fps = video.fps  # frame per second of the video
        
        video_script = []
        with open(csv_filename, 'r', encoding='utf-8') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                video_script.append(row)        

        # Read the video_script and overlay audio files
        audio_clips = []
        for frame_count, narration in video_script:
            frame_count = int(frame_count)  # Convert frame_count to integer
            audio_path = f'{folder}/narration_{str(frame_count).zfill(7)}.wave'
            audio_clip = AudioFileClip(audio_path)
            start_time = frame_count / fps  # Calculate the start time in seconds
            audio_clip = audio_clip.set_start(start_time)
            audio_clips.append(audio_clip)

        # Reduce the volume of the original audio to 30%
        original_audio = video.audio.volumex(0.15)

        # Add the original audio to the list of audio clips
        audio_clips.append(original_audio)
        
        # Calculate the total audio duration
        total_audio_duration = max(audio_clip.end for audio_clip in audio_clips)
        
        # Check if the audio is longer than the video
        if total_audio_duration > video.duration:
            # Create a freeze frame from the last frame of the video
            last_frame = video.to_ImageClip(video.duration- 0.1)
            freeze_duration = total_audio_duration - video.duration
            freeze_frame = last_frame.set_duration(freeze_duration)

            # Extend the video by appending the freeze frame
            extended_video = concatenate_videoclips([video, freeze_frame])
        else:
            extended_video = video

        if portrait:
            extended_video = extended_video.resize((1080, 1920)) # moviepy doesn't take portrait mode into account, here is a hack

        # Set the combined audio to the extended video
        final_audio = CompositeAudioClip(audio_clips).set_duration(extended_video.duration)
        extended_video = extended_video.set_audio(final_audio)        
        
        # Write the result to a file
        extended_video.write_videofile(f"{folder}/output.mp4", codec="libx264", audio_codec="aac")


#
#        play_audio(speech)
#
#        gpt_message_history = gpt_message_history + [{"role": "assistant", "content": analysis}]        
        
