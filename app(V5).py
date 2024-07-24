!pip install --quiet g4f[all] --upgrade
!pip install --quiet TTS


import os
import json
from pprint import pprint
import time
from json import JSONDecodeError



# Function to fetch topic and goal.
def fetch_topic_and_goal(niche, goal_description):
    prompt = f"""
    You are an expert in creating engaging video content. Based on the niche '{niche}', generate a specific and unique topic for a video. The goal of the video should be '{goal_description}'.
    """
    try:
        response = g4f.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert video content creator."},
                {"role": "user", "content": prompt}
            ],
            temperature=1.0,
            max_tokens=200,
            top_p=1,
            stream=False
        )
        output = json.loads(response)
        topic = output.get('topic', 'Default Topic')
        goal = output.get('goal', 'Default Goal')
        return topic, goal
    except (JSONDecodeError, Exception) as e:
        print(f"Error: {e}. Retrying...")
        time.sleep(1)
    raise Exception("Failed to fetch topic and goal.")



def fetch_imagedescription_and_script(prompt)
    max_retries = 25
    for i in range(max_retries)
        try
            response = g4f.ChatCompletion.create(
                model=gpt-3.5-turbo,
                messages=[
                    {role system, content You are an expert short form video script writer for Instagram Reels and Youtube shorts.},
                    {role user, content prompt}
                ],
                temperature=1.3,
                max_tokens=2000,
                top_p=1,
                stream=False
            )

            # Parse the response
            output = json.loads(response)
            pprint(output)
            image_prompts = [k['image_description'] for k in output]
            texts = [k['text'] for k in output]

            return image_prompts, texts
        except (JSONDecodeError, Exception) as e
            print(fError {e}. Retrying...)
            time.sleep(1)  # wait for 1 second before retrying
    raise Exception(Failed to fetch image description and script after {} retries.format(max_retries))

import g4f
import json
from pprint import pprint
import time
from json import JSONDecodeError

# Define your topic and goal
topic = Happiness and Joy
goal = help people find happiness in simple moments and enjoy life's journey

prompt_prefix = You are tasked with creating a script for a {} video that is about 30 seconds.
Your goal is to {}.
Please follow these instructions to create an engaging and impactful video
1. Begin by setting the scene and capturing the viewer's attention with a captivating visual.
2. Each scene cut should occur every 5-10 seconds, ensuring a smooth flow and transition throughout the video.
3. For each scene cut, provide a detailed description of the stock image being shown.
4. Along with each image description, include a corresponding text that complements and enhances the visual. The text should be concise and powerful.
5. Ensure that the sequence of images and text builds excitement and encourages viewers to take action. ONLY ENGLISH
6. Strictly output your response in a JSON list format, adhering to the following sample structure.format(topic, goal)

sample_output=
   [
       { image_description Description of the first image here., text Text accompanying the first scene cut. },
       { image_description Description of the second image here., text Text accompanying the second scene cut. },
     ...
   ]

prompt_postinstruction=By following these instructions, you will create an impactful {} short-form video.
Output.format(topic)

prompt = prompt_prefix + sample_output + prompt_postinstruction

image_prompts, texts = fetch_imagedescription_and_script(prompt)
print(image_prompts , image_prompts)
print(texts , texts)
print(len(texts))

# 2. Create a new folder with a unique name.
import uuid

current_uuid = uuid.uuid4()
active_folder = str(current_uuid)
print(active_folder)

#############################################################################
# Generate high-quality images for those descriptions using Segmind API or Hercai
#############################################################################

# User's choice of image generator
video_source = segmind #  @param [segmind,  hercai] {allow-input true}


import os
import io
import requests
from PIL import Image
import random

# Segmind API
# Use multiple API keys. In case one key's quota ends, the second API will be automatically selected
segmind_apikey = os.environ.get('SG_2d3504ba72dbeacc', 'SG_2d3504ba72dbeacc').split(',')
api_key_index = 0

# Generate images using Segmind API (supports waiting to comply with the rate limit of 5 image requests per minute)
def generate_images_segmind(prompts, fname)
    global api_key_index
    url = httpsapi.segmind.comv1sdxl1.0-txt2img

    # Initialize headers outside the loop
    headers = {'x-api-key' segmind_apikey[api_key_index]}

    # Create a folder for the UUID if it doesn't exist
    if not os.path.exists(fname)
        os.makedirs(fname)

    num_images = len(prompts)
    requests_made = 0
    start_time = time.time()

    for i, prompt in enumerate(prompts)
        # Handle rate limit (5 requests per minute)
        if requests_made = 5 and time.time() - start_time = 60
            time_to_wait = 60 - (time.time() - start_time)
            print(fWaiting for {time_to_wait.2f} seconds to comply with rate limit...)
            time.sleep(time_to_wait)
            requests_made = 0
            start_time = time.time()

        final_prompt = ((perfect quality)), ((cinematic photo1.3)), ((raw candid)), 4k, {}, no occlusion, Fujifilm XT3, highly detailed, bokeh, cinemascope.format(prompt.strip('.'))
        data = {
            prompt final_prompt,
            negative_prompt ((deformed)), ((limbs cut off)), ((quotes)), ((extra fingers)), ((deformed hands)), extra limbs, disfigured, blurry, bad anatomy, absent limbs, blurred, watermark, disproportionate, grainy, signature, cut off, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, worst face, three crus, extra crus, fused crus, worst feet, three feet, fused feet, fused thigh, three thigh, fused thigh, extra thigh, worst thigh, missing fingers, extra fingers, ugly fingers, long fingers, horn, extra eyes, amputation, disconnected limbs,
            style hdr,
            samples 1,
            scheduler UniPC,
            num_inference_steps 30,
            guidance_scale 8,
            strength 1,
            seed random.randint(1, 1000000),
            img_width 1024,
            img_height 1024,
            refiner yes,
            base64 False
        }

        while True  # Loop to retry with a different API key if quota exceeded
            # Make the API call to Segmind
            response = requests.post(url, json=data, headers=headers)
            requests_made += 1

            if response.status_code == 200 and response.headers.get('content-type') == 'imagejpeg'
                image_data = response.content
                image = Image.open(io.BytesIO(image_data))

                image_filename = os.path.join(fname, f{i + 1}.jpg)
                image.save(image_filename)

                print(fImage {i + 1}{num_images} saved as '{image_filename}')
                break  # Exit the retry loop if successful
            else
                print(response.status_code)
                print(response.text)
                if response.status_code == 429  # Quota exceeded
                    api_key_index = (api_key_index + 1) % len(segmind_apikey)
                    headers['x-api-key'] = segmind_apikey[api_key_index]  # Update header
                    print(fSwitching to API key {segmind_apikey[api_key_index]})
                    time.sleep(1)  # Wait a bit
                    continue  # Retry with the new API key
                else
                    print(fError Failed to retrieve or save image {i + 1})
                    break  # Exit the retry loop if there's an error other than quota exceeded

# Generate images using Hercai
def generate_images_hercai(prompts, fname)
    image_model = animefy

    # Create a folder for the UUID if it doesn't exist
    if not os.path.exists(fname)
        os.makedirs(fname)

    num_images = len(prompts)

    currentseed = random.randint(1, 1000000)
    print(seed, currentseed)

    for i, prompt in enumerate(prompts)
        final_prompt = ((perfect quality)), ((cinematic photo1.3)), ((raw candid)), 4k, {}, no occlusion, Fujifilm XT3, highly detailed, bokeh, cinemascope.format(prompt.strip('.'))
        url = fhttpshercai.onrender.com{image_model}text2imageprompt={final_prompt}
        response = requests.get(url)
        if response.status_code == 200
            parsed = response.json()
            if url in parsed and parsed[url]
                image_url = parsed[url]
                image_response = requests.get(image_url)
                if image_response.status_code == 200
                    image_data = image_response.content
                    image = Image.open(io.BytesIO(image_data))
                    image_filename = os.path.join(fname, f{i + 1}.png)
                    image.save(image_filename)
                    print(fImage {i + 1}{num_images} saved as '{image_filename}')
                else
                    print(fError Failed to retrieve image from URL for image {i + 1})
            else
                print(fError No image URL in response for image {i + 1})
        else
            print(response.text)
            print(fError Failed to retrieve or save image {i + 1})

# Call the appropriate function based on the user's choice
if video_source == segmind
    generate_images_segmind(image_prompts, active_folder)
elif video_source == hercai
    generate_images_hercai(image_prompts, active_folder)
else
    print(Invalid image generator choice. Please choose either 'segmind' or 'hercai'.)

# 2.1 Visualize generated images
!pip install --quiet ipyplot

import os
from PIL import Image

def read_images_from_folder(folder_path)
    image_objects = []

    # Get a list of filenames in the folder and sort them based on their numeric values
    filenames = [filename for filename in os.listdir(folder_path) if filename.lower().endswith(.jpg) or filename.lower().endswith(.png)]
    filenames = sorted(filenames, key=lambda x int(x.split('.')[0]))
    print(filenames)

    for filename in filenames
        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path)
        image_objects.append(image)

    return image_objects

images = read_images_from_folder(active_folder)

import ipyplot
from PIL import Image
import numpy as np

# Convert PIL Image objects to NumPy arrays
np_images = [np.array(img) for img in images]
ipyplot.plot_images(np_images, labels=image_prompts, img_width=300)

#############################################################################
# Convert text to speech using Elevenlabs API or CoquieTTS
#############################################################################

# User's choice of TTS provider
tts_provider = coquietts #  @param [elevenlabs,  coquietts] {allow-input true}

import os
import re
import getpass
import requests
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

# Elevenlabs API
# Retrieve API keys from environment variable (if it exists) or use the provided default keys
api_keys = os.environ.get('ec71cc5fb466bbbeaa935e5a7b001d25', '675e7d7c9d7a10caf1bb77f6264cd1c9,sk_7659d722c316eb37935ec20cedf13cea5e80dd5ac899b95f').split(',')
api_key_index = 0

def generate_and_save_audio_elevenlabs(text, foldername, filename, voice_id, model_id=eleven_multilingual_v2, stability=0.4, similarity_boost=0.80)
    global api_key_index  # Access the global index variable

    # Cycle through API keys
    api_key = api_keys[api_key_index]
    url = fhttpsapi.elevenlabs.iov1text-to-speech{voice_id}
    headers = {
        Accept audiompeg,
        Content-Type applicationjson,
        xi-api-key api_key
    }

    data = {
        text text,
        model_id model_id,
        voice_settings {
            stability stability,
            similarity_boost similarity_boost
        }
    }

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 429  # Handle quota exceeded error
        print(Quota exceeded for current API key. Switching to the next key.)
        api_key_index = (api_key_index + 1) % len(api_keys)
        generate_and_save_audio_elevenlabs(text, foldername, filename, voice_id, model_id, stability, similarity_boost)  # Retry with the new key
    elif response.status_code != 200
        print(response.text)
    else
        file_path = f{foldername}{filename}.mp3
        with open(file_path, 'wb') as f
            f.write(response.content)

# CoquieTTS
def generate_script_to_speech_coquietts(text, foldername, filename, index, total)
    # Initialize the synthesizer once
    model_manager = ModelManager()
    tts_model_path, tts_config_path, _ = model_manager.download_model(tts_modelsenljspeechtacotron2-DDC_ph)
    vocoder_model_path, vocoder_config_path, _ = model_manager.download_model(vocoder_modelsenljspeechunivnet)

    synthesizer = Synthesizer(
        tts_checkpoint=tts_model_path,
        tts_config_path=tts_config_path,
        vocoder_checkpoint=vocoder_model_path,
        vocoder_config=vocoder_config_path
    )

    path = os.path.join(foldername, f{filename}.wav)
    text = re.sub(r'[^ws.!]', '', text)

    # Use the synthesizer to generate speech
    wav = synthesizer.tts(text)

    # Save the generated audio
    synthesizer.save_wav(wav, path)

    print(fVoice {index}{total} saved as '{path}')
    return path

# Call the appropriate function based on the user's choice
if tts_provider == elevenlabs
    voice_id = pNInz6obpgDQGcFmaJgB
    for i, text in enumerate(texts)
        output_filename = str(i + 1)
        print(output_filename)
        generate_and_save_audio_elevenlabs(text, active_folder, output_filename, voice_id)
elif tts_provider == coquietts
    total_texts = len(texts)
    for i, text in enumerate(texts, 1)
        output_filename = str(i)
        generate_script_to_speech_coquietts(text, active_folder, output_filename, i, total_texts)
else
    print(Invalid TTS provider choice. Please choose either 'elevenlabs' or 'coquietts'.)


import os
import cv2
import numpy as np
from moviepy.editor import AudioFileClip, ImageClip, concatenate_audioclips, concatenate_videoclips

def create_combined_video_audio(media_folder, output_filename, output_resolution=(1080, 1920), fps=24)
    audio_files = sorted([file for file in os.listdir(media_folder) if file.lower().endswith((.mp3, .wav))])
    audio_files = sorted(audio_files, key=lambda x int(x.split('.')[0]))

    audio_clips = []
    video_clips = []

    for audio_file in audio_files
        audio_clip = AudioFileClip(os.path.join(media_folder, audio_file))
        audio_clips.append(audio_clip)

        # Try to load the corresponding image (either JPG or PNG)
        base_name = audio_file.split('.')[0]
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png']
            temp_path = os.path.join(media_folder, f{base_name}{ext})
            if os.path.exists(temp_path)
                img_path = temp_path
                break

        if img_path is None
            raise FileNotFoundError(fNo image file found for audio file {audio_file})

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize the original image to 1080x1080
        image_resized = cv2.resize(image, (1080, 1080))

        # Blur the image
        blurred_img = cv2.GaussianBlur(image, (0, 0), 30)
        blurred_img = cv2.resize(blurred_img, output_resolution)

        # Overlay the original image on the blurred one
        y_offset = (output_resolution[1] - 1080)  2
        blurred_img[y_offsety_offset+1080, ] = image_resized

        video_clip = ImageClip(np.array(blurred_img), duration=audio_clip.duration)
        video_clips.append(video_clip)

    final_audio = concatenate_audioclips(audio_clips)
    final_video = concatenate_videoclips(video_clips, method=compose)

    # Set the audio directly
    final_video.audio = final_audio
    finalpath = os.path.join(media_folder, output_filename)
    final_video.write_videofile(finalpath, fps=fps, codec='libx264', audio_codec=aac)

output_filename = combined_video.mp4
create_combined_video_audio(active_folder, output_filename)
output_video_file = os.path.join(active_folder, output_filename)

# 4. Install Moviepy to stitch everything
!pip install --quiet git+httpsgithub.comZulkomoviepy.git@bc8d1a831d2d1f61abfdf1779e8df95d523947a5
!pip install --quiet imageio==2.25.1
!apt install -qq imagemagick
!cat etcImageMagick-6policy.xml  sed 'snoneread,writeg' etcImageMagick-6policy.xml
!pip install ffmpeg-python==0.2.0


import ffmpeg

def extract_audio_from_video(outvideo)
    
    Extract audio from a video file and save it as an MP3 file.

    param output_video_file Path to the video file.
    return Path to the generated audio file.
    

    audiofilename = outvideo.replace(.mp4,'.mp3')

    # Create the ffmpeg input stream
    input_stream = ffmpeg.input(outvideo)

    # Extract the audio stream from the input stream
    audio = input_stream.audio

    # Save the audio stream as an MP3 file
    output_stream = ffmpeg.output(audio, audiofilename)

    # Overwrite output file if it already exists
    output_stream = ffmpeg.overwrite_output(output_stream)

    ffmpeg.run(output_stream)

    return audiofilename

audiofilename = extract_audio_from_video(output_video_file)
print(audiofilename)



from IPython.display import Audio
Audio(audiofilename)

!pip install --quiet faster-whisper==0.7.0

from faster_whisper import WhisperModel

model_size = base
model = WhisperModel(model_size)

segments, info = model.transcribe(audiofilename, word_timestamps=True)
segments = list(segments)  # The transcription will actually run here.
for segment in segments
    for word in segment.words
        print([%.2fs - %.2fs] %s % (word.start, word.end, word.word))

wordlevel_info = []

for segment in segments
    for word in segment.words
      wordlevel_info.append({'word'word.word,'start'word.start,'end'word.end})

wordlevel_info

from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip

# Load the video file
video = VideoFileClip(output_video_file)

def generate_text_clip(word, start, end, video)
    txt_clip = (TextClip(word, fontsize=80, color='white', font=Nimbus-Sans-Bold, stroke_width=3, stroke_color='black').with_position('center')
               .with_duration(end - start))

    return txt_clip.with_start(start)

# Generate a list of text clips based on timestamps
clips = [generate_text_clip(item['word'], item['start'], item['end'], video) for item in wordlevel_info]

# Overlay the text clips on the video
final_video = CompositeVideoClip([video] + clips)

finalvideoname = active_folder++final.mp4
# Write the result to a file
final_video.write_videofile(finalvideoname, codec=libx264,audio_codec=aac)



####################################
# WEBUI
###################################
# import gradio as gr

# def save_secrets():
#     return "Secrets saved successfully!"

# def generate_video():
#     return "Video generated successfully!"

# def update_video_settings(video_source):
#     if video_source == "Image Model":
#         return gr.Dropdown.update(visible=True), gr.Checkbox.update(visible=True)
#     else:
#         return gr.Dropdown.update(visible=False), gr.Checkbox.update(visible=True)

# def update_image_input(use_image):
#     return gr.Number.update(visible=use_image)

# with gr.Blocks(theme=gr.themes.Base()) as demo:
#     gr.Markdown("# SocialGPT")
    
#     with gr.Accordion("Secrets", open=False):
#         with gr.Row():
#             with gr.Column(scale=4):
#                 pexel_key = gr.Textbox(label="PEXEL_API_KEY", placeholder="Enter Pexel API Key")
#                 elevenlabs_key = gr.Textbox(label="ELEVENLABS_API_KEY", placeholder="Enter ElevenLabs API Key")
#                 google_key = gr.Textbox(label="GOOGLE_API_KEY", placeholder="Enter Google API Key")
#                 openai_key = gr.Textbox(label="OPENAI_API_KEY", placeholder="Enter OpenAI API Key")
#                 assembly_key = gr.Textbox(label="ASSEMBLY_API_KEY", placeholder="Enter Assembly API Key")
#             with gr.Column(scale=1):
#                 save_button = gr.Button("Save", size="lg")
    
#     with gr.Row():
#         # First column
#         with gr.Column(scale=1, min_width=300):
#             with gr.Box():
#                 gr.Markdown("### Script Settings")
#                 subject = gr.Textbox(label="Video Subject", lines=1, value="Make Money Online")
#                 llm = gr.Dropdown(["gpt4", "gpt35_turbo", "llama2_7b", "llama2_13b", "llama2_70b", "mixtral_8x7b", "google"], label="LLM", value="google")
#                 prompter = gr.Dropdown(["gpt4", "gpt35_turbo", "llama2_7b", "llama2_13b", "llama2_70b", "mixtral_8x7b", "google"], label="Prompter", value="google")
#                 watermark = gr.Textbox(label="Watermark", value="SocialGPT")
#                 watermark_position = gr.Dropdown(["Top", "Bottom", "Left", "Right", "Center"], label="Watermark Position", value="Bottom")
        
#         # Second column
#         with gr.Column(scale=1, min_width=300):
#             with gr.Box():
#                 gr.Markdown("### Audio Settings")
#                 tts = gr.Dropdown(["Elevenlabs", "Edge", "Coqui", "OpenAI"], label="TTS", value="Elevenlabs")
#                 language = gr.Dropdown(["English", "Hindi", "Chinese", "Urdu", "Arabic"], label="Language", value="English")
#                 voice = gr.Dropdown(["Adam", "Antoni", "Bella"], label="Voice", value="Antoni")
#                 bg_music_volume = gr.Slider(minimum=0, maximum=1, value=0.2, step=0.1, label="Background Music Volume")
            
#             with gr.Box():
#                 gr.Markdown("### Video Settings")
#                 video_source = gr.Dropdown(["Pexel", "Image Model"], label="Video Source", value="Pexel")
#                 image_model = gr.Dropdown(["v1", "v2", "v3 (DALL-E)", "lexica", "prodia", "simurg", "animefy", "raava", "shonin"], label="Image Model", visible=False, value="v3 (DALL-E)")
#                 aspect_ratio = gr.Dropdown(["Landscape 1920x1080 (Youtube)", "Portrait 1080x1920", "Square 1080x1080"], label="Aspect Ratio", value="Landscape 1920x1080 (Youtube)")
#                 use_image = gr.Checkbox(label="Use Image", value=False)
#                 image_input = gr.Number(label="Number of Images", minimum=1, maximum=25, value=1, visible=False)
        
#         # Third column
#         with gr.Column(scale=1, min_width=300):
#             with gr.Box():
#                 gr.Markdown("### Subtitle Settings")
#                 enable_subtitles = gr.Checkbox(label="Enable Subtitles", value=True)
#                 subtitle_font = gr.Dropdown(["MicrosoftYaHei"], label="Font", value="MicrosoftYaHei")
#                 subtitle_position = gr.Dropdown(["Top", "Bottom", "Left", "Right", "Center"], label="Position", value="Bottom")
#                 subtitle_color = gr.ColorPicker(label="Font Color", value="#ffffff")
#                 subtitle_size = gr.Slider(minimum=0, maximum=100, value=40, step=1, label="Font Size")
#                 subtitle_outline = gr.Checkbox(label="Outline Color", value=False)
#                 subtitle_outline_width = gr.Slider(minimum=0, maximum=20, value=0, step=1, label="Outline Width")

#     generate_button = gr.Button("Generate Video", variant="primary")
#     output = gr.Textbox(label="Output")

#     save_button.click(fn=save_secrets, outputs=output)
#     generate_button.click(fn=generate_video, outputs=output)
    
#     video_source.change(fn=update_video_settings, inputs=video_source, outputs=[image_model, use_image])
#     use_image.change(fn=update_image_input, inputs=use_image, outputs=image_input)

# demo.launch()
