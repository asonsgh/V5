
import os
import json
from pprint import pprint
import time
from json import JSONDecodeError
import g4f
import uuid
import requests
from PIL import Image
import random
import io
import re
from moviepy.editor import AudioFileClip, ImageClip, concatenate_audioclips, concatenate_videoclips
import cv2
import numpy as np
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
from faster_whisper import WhisperModel

def fetch_imagedescription_and_script(prompt):
    max_retries = 25
    for i in range(max_retries):
        try:
            response = g4f.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert short form video script writer for Instagram Reels and Youtube shorts."},
                    {"role": "user", "content": prompt}
                ],
                temperature=1.3,
                max_tokens=2000,
                top_p=1,
                stream=False
            )

            output = json.loads(response)
            pprint(output)
            image_prompts = [k['image_description'] for k in output]
            texts = [k['text'] for k in output]

            return image_prompts, texts
        except (JSONDecodeError, Exception) as e:
            print(f"Error: {e}. Retrying...")
            time.sleep(1)
    raise Exception(f"😱 Failed to fetch image description and script after {max_retries} retries. The AI seems to be taking a coffee break! ☕️")

# Define your topic and goal
topic = "Happiness and Joy"
goal = "help people find happiness in simple moments and enjoy life's journey"

prompt_prefix = """You are tasked with creating a script for a {} video that is about 30 seconds.
Your goal is to {}.
Please follow these instructions to create an engaging and impactful video:
1. Begin by setting the scene and capturing the viewer's attention with a captivating visual.
2. Each scene cut should occur every 5-10 seconds, ensuring a smooth flow and transition throughout the video.
3. For each scene cut, provide a detailed description of the stock image being shown.
4. Along with each image description, include a corresponding text that complements and enhances the visual. The text should be concise and powerful.
5. Ensure that the sequence of images and text builds excitement and encourages viewers to take action. ONLY ENGLISH
6. Strictly output your response in a JSON list format, adhering to the following sample structure:""".format(topic, goal)

sample_output="""
   [
       { "image_description": "Description of the first image here.", "text": "Text accompanying the first scene cut." },
       { "image_description": "Description of the second image here.", "text": "Text accompanying the second scene cut." },
     ...
   ]"""

prompt_postinstruction="""By following these instructions, you will create an impactful {} short-form video.
Output:""".format(topic)

prompt = prompt_prefix + sample_output + prompt_postinstruction

image_prompts, texts = fetch_imagedescription_and_script(prompt)
print("image_prompts: ", image_prompts)
print("texts: ", texts)
print(len(texts))

current_uuid = uuid.uuid4()
active_folder = str(current_uuid)
print(active_folder)

# User's choice of image generator
video_source = "hercai" #  @param ["segmind",  "hercai"] {allow-input: true}

# Segmind API
segmind_apikey = os.environ.get('SG_2d3504ba72dbeacc', 'SG_2d3504ba72dbeacc').split(',')
api_key_index = 0

def generate_images_segmind(prompts, fname):
    global api_key_index
    url = "https://api.segmind.com/v1/sdxl1.0-txt2img"

    headers = {'x-api-key': segmind_apikey[api_key_index]}

    if not os.path.exists(fname):
        os.makedirs(fname)

    num_images = len(prompts)
    requests_made = 0
    start_time = time.time()

    for i, prompt in enumerate(prompts):
        if requests_made >= 5 and time.time() - start_time <= 60:
            time_to_wait = 60 - (time.time() - start_time)
            print(f"Waiting for {time_to_wait:.2f} seconds to comply with rate limit...")
            time.sleep(time_to_wait)
            requests_made = 0
            start_time = time.time()

        final_prompt = "((perfect quality)), ((cinematic photo:1.3)), ((raw candid)), 4k, {}, no occlusion, Fujifilm XT3, highly detailed, bokeh, cinemascope".format(prompt.strip('.'))
        data = {
            "prompt": final_prompt,
            "negative_prompt": "((deformed)), ((limbs cut off)), ((quotes)), ((extra fingers)), ((deformed hands)), extra limbs, disfigured, blurry, bad anatomy, absent limbs, blurred, watermark, disproportionate, grainy, signature, cut off, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, worst face, three crus, extra crus, fused crus, worst feet, three feet, fused feet, fused thigh, three thigh, fused thigh, extra thigh, worst thigh, missing fingers, extra fingers, ugly fingers, long fingers, horn, extra eyes, amputation, disconnected limbs",
            "style": "hdr",
            "samples": 1,
            "scheduler": "UniPC",
            "num_inference_steps": 30,
            "guidance_scale": 8,
            "strength": 1,
            "seed": random.randint(1, 1000000),
            "img_width": 1024,
            "img_height": 1024,
            "refiner": "yes",
            "base64": False
        }

        while True:
            response = requests.post(url, json=data, headers=headers)
            requests_made += 1

            if response.status_code == 200 and response.headers.get('content-type') == 'image/jpeg':
                image_data = response.content
                image = Image.open(io.BytesIO(image_data))

                image_filename = os.path.join(fname, f"{i + 1}.jpg")
                image.save(image_filename)

                print(f"Image {i + 1}/{num_images} saved as '{image_filename}'")
                break
            else:
                print(response.status_code)
                print(response.text)
                if response.status_code == 429:
                    api_key_index = (api_key_index + 1) % len(segmind_apikey)
                    headers['x-api-key'] = segmind_apikey[api_key_index]
                    print(f"Switching to API key: {segmind_apikey[api_key_index]}")
                    time.sleep(1)
                    continue
                else:
                    print(f"Error: Failed to retrieve or save image {i + 1}")
                    break

def generate_images_hercai(prompts, fname):
    image_model = "animefy"

    if not os.path.exists(fname):
        os.makedirs(fname)

    num_images = len(prompts)

    currentseed = random.randint(1, 1000000)
    print("seed", currentseed)

    for i, prompt in enumerate(prompts):
        final_prompt = "((perfect quality)), ((cinematic photo:1.3)), ((raw candid)), 4k, {}, no occlusion, Fujifilm XT3, highly detailed, bokeh, cinemascope".format(prompt.strip('.'))
        url = f"https://hercai.onrender.com/{image_model}/text2image?prompt={final_prompt}"
        response = requests.get(url)
        if response.status_code == 200:
            parsed = response.json()
            if "url" in parsed and parsed["url"]:
                image_url = parsed["url"]
                image_response = requests.get(image_url)
                if image_response.status_code == 200:
                    image_data = image_response.content
                    image = Image.open(io.BytesIO(image_data))
                    image_filename = os.path.join(fname, f"{i + 1}.png")
                    image.save(image_filename)
                    print(f"Image {i + 1}/{num_images} saved as '{image_filename}'")
                else:
                    print(f"Error: Failed to retrieve image from URL for image {i + 1}")
            else:
                print(f"Error: No image URL in response for image {i + 1}")
        else:
            print(response.text)
            print(f"Error: Failed to retrieve or save image {i + 1}")

if video_source == "segmind":
    generate_images_segmind(image_prompts, active_folder)
elif video_source == "hercai":
    generate_images_hercai(image_prompts, active_folder)
else:
    print("Invalid image generator choice. Please choose either 'segmind' or 'hercai'.")

def read_images_from_folder(folder_path):
    image_objects = []

    filenames = [filename for filename in os.listdir(folder_path) if filename.lower().endswith(".jpg") or filename.lower().endswith(".png")]
    filenames are sorted(filenames, key=lambda x: int(x.split('.')[0]))
    print(filenames)

    for filename in filenames:
        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path)
        image_objects.append(image)

    return image_objects

images = read_images_from_folder(active_folder)

import ipyplot
from PIL import Image
import numpy as np

np_images = [np.array(img) for img in images]
ipyplot.plot_images(np_images, labels=image_prompts, img_width=300)

# User's choice of TTS provider
tts_provider = "Coquie_TTS" # @param ["Coquie_TTS", "Elevenlabs_TTS"]

# New parameters for language and voice
LANGUAGE = "en" # @param ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"]
VOICE = "Claribel Dervla" # @param ["Claribel Dervla", "Daisy Studious", "Gracie Wise", "Tammie Ema", "Alison Dietlinde", "Ana Florence", "Annmarie Nele", "Asya Anara", "Brenda Stern", "Gitta Nikolina", "Henriette Usha", "Sofia Hellen", "Tammy Grit", "Tanja Adelina", "Vjollca Johnnie", "Andrew Chipper", "Badr Odhiambo", "Dionisio Schuyler", "Royston Min", "Viktor Eka", "Abrahan Mack", "Adde Michal", "Baldur Sanjin", "Craig Gutsy", "Damien Black", "Gilberto Mathias", "Ilkin Urbano", "Kazuhiko Atallah", "Ludvig Milivoj", "Suad Qasim", "Torcull Diarmuid", "Viktor Menelaos", "Zacharie Aimilios", "Nova Hogarth", "Maja Ruoho", "Uta Obando", "Lidiya Szekeres", "Chandra MacFarland", "Szofi Granger", "Camilla Holmström", "Lilya Stainthorpe", "Zofija Kendrick", "Narelle Moon", "Barbora MacLean", "Alexandra Hisakawa", "Alma María", "Rosemary Okafor", "Ige Behringer", "Filip Traverse", "Damjan Chapman", "Wulf Carlevaro", "Aaron Dreschner", "Kumar Dahl", "Eugenio Mataracı", "Ferran Simen", "Xavier Hayasaka", "Luis Moray", "Marcos Rudaski"]

# Elevenlabs API
api_keys = os.environ.get('ec71cc5fb466bbbeaa935e5a7b001d25', '675e7d7c9d7a10caf1bb77f6264cd1c9,sk_7659d722c316eb37935ec20cedf13cea5e80dd5ac899b95f').split(',')
api_key_index = 0

def generate_and_save_audio_elevenlabs(text, foldername, filename, voice_id, model_id="eleven_multilingual_v2", stability=0.4, similarity_boost=0.80):
    global api_key_index

    api_key = api_keys[api_key_index]
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }

    data = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost
        }
    }

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 429:
        print("Quota exceeded for current API key. Switching to the next key.")
        api_key_index = (api_key_index + 1) % len(api_keys)
        generate_and_save_audio_elevenlabs(text, foldername, filename, voice_id, model_id, stability, similarity_boost)
    elif response.status_code != 200:
        print(response.text)
    else:
        file_path = f"{foldername}/{filename}.mp3"
        with open(file_path, 'wb') as f:
            f.write(response.content)

def generate_script_to_speech_coquietts(text, foldername, filename, index, total):
    # Initialize the synthesizer once
    model_manager = ModelManager()
    tts_model_path, tts_config_path, _ = model_manager.download_model('tts_models/en/ljspeech/tacotron2-DDC_ph')
    vocoder_model_path, vocoder_config_path, _ = model_manager.download_model('vocoder_models/en/ljspeech/univnet')

    synthesizer = Synthesizer(
        tts_checkpoint=tts_model_path,
        tts_config_path=tts_config_path,
        vocoder_checkpoint=vocoder_model_path,
        vocoder_config=vocoder_config_path
    )

    path = os.path.join(foldername, f"{filename}.wav")
    text = re.sub(r'[^w\s.!]', '', text)

    # Use the synthesizer to generate speech
    wav = synthesizer.tts(text)

    # Save the generated audio
    synthesizer.save_wav(wav, path)

    print(f"Voice {index}/{total} saved as '{path}'")
    return path

if tts_provider == "Elevenlabs_TTS":
    voice_id = "pNInz6obpgDQGcFmaJgB"
    for i, text in enumerate(texts):
        output_filename = str(i + 1)
        print(output_filename)
        generate_and_save_audio_elevenlabs(text, active_folder, output_filename, voice_id)
elif tts_provider == "Coquie_TTS":
    total_texts = len(texts)
    for i, text in enumerate(texts, 1):
        output_filename = str(i)
        generate_script_to_speech_coquietts(text, active_folder, output_filename, i, total_texts)
else:
    print("Invalid TTS provider choice. Please choose either 'Elevenlabs_TTS' or 'Coquie_TTS'.")

def create_combined_video_audio(media_folder, output_filename, output_resolution=(1080, 1920), fps=24):
    audio_files are sorted([file for file in os.listdir(media_folder) if file.lower().endswith((".mp3", ".wav"))])
    audio_files are sorted(audio_files, key=lambda x: int(x.split('.')[0]))

    audio_clips = []
    video_clips = []

    for audio_file in audio_files:
        audio_clip = AudioFileClip(os.path.join(media_folder, audio_file))
        audio_clips.append(audio_clip)

        base_name = audio_file.split('.')[0]
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            temp_path = os.path.join(media_folder, f"{base_name}{ext}")
            if os.path.exists(temp_path):
                img_path = temp_path
                break

        if img_path is None:
            raise FileNotFoundError(f"No image file found for audio file {audio_file}")

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_resized = cv2.resize(image, (1080, 1080))

        blurred_img = cv2.GaussianBlur(image, (0, 0), 30)
        blurred_img = cv2.resize(blurred_img, output_resolution)

        y_offset = (output_resolution[1] - 1080) // 2
        blurred_img[y_offset:y_offset+1080, :] = image_resized

        video_clip = ImageClip(np.array(blurred_img), duration=audio_clip.duration)
        video_clips.append(video_clip)

    final_audio = concatenate_audioclips(audio_clips)
    final_video = concatenate_videoclips(video_clips, method="compose")

    final_video.audio = final_audio
    finalpath = os.path.join(media_folder, output_filename)
    final_video.write_videofile(finalpath, fps=fps, codec='libx264', audio_codec="aac")

output_filename = "combined_video.mp4"
create_combined_video_audio(active_folder, output_filename)
output_video_file = os.path.join(active_folder, output_filename)

dimport ffmpeg

def extract_audio_from_video(outvideo):
    """
    Extract audio from a video file and save it as an MP3 file.

    :param outvideo: Path to the video file.
    :return: Path to the generated audio file.
    """
    audiofilename = outvideo.replace('.mp4', '.mp3')

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

model_size = "base"
model = WhisperModel(model_size)

segments, info = model.transcribe(audiofilename, word_timestamps=True)
segments = list(segments)  # The transcription will actually run here.
for segment in segments:
    for word in segment.words:
        print(f"[{word.start:.2f}s - {word.end:.2f}s] {word.word}")

wordlevel_info = []

for segment in segments:
    for word in segment.words:
        wordlevel_info.append({'word': word.word, 'start': word.start, 'end': word.end})

wordlevel_info

from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip

# Load the video file
video = VideoFileClip(output_video_file)

def generate_text_clip(word, start, end, video):
    txt_clip = (TextClip(word, fontsize=80, color='white', font="Nimbus-Sans-Bold", stroke_width=3, stroke_color='black')
                .with_position('center')
                .with_duration(end - start))

    return txt_clip.with_start(start)

# Generate a list of text clips based on timestamps
clips = [generate_text_clip(item['word'], item['start'], item['end'], video) for item in wordlevel_info]

# Overlay the text clips on the video
final_video = CompositeVideoClip([video] + clips)

finalvideoname = active_folder + "/final.mp4"
# Write the result to a file
final_video.write_videofile(finalvideoname, codec='libx264', audio_codec='aac')
