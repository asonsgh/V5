import os
import g4f
import json
import time
from TTS.api import TTS
from pprint import pprint
from json import JSONDecodeError

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

            # Parse the response
            output = json.loads(response)
            pprint(output)
            image_prompts = [k['image_description'] for k in output]
            texts = [k['text'] for k in output]

            return image_prompts, texts
        except (JSONDecodeError, Exception) as e:
            print(f"Error: {e}. Retrying...")
            time.sleep(1)  # wait for 1 second before retrying
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

# 2. Create a new folder with a unique name.
import uuid

current_uuid = uuid.uuid4()
active_folder = str(current_uuid)
print(active_folder)

#############################################################################
# Generate high-quality images for those descriptions using Segmind API or Hercai
#############################################################################

# User's choice of image generator
video_source = "hercai" #  @param ["segmind",  "hercai"] {allow-input: true}


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
def generate_images_segmind(prompts, fname):
    global api_key_index
    url = "https://api.segmind.com/v1/sdxl1.0-txt2img"

    # Initialize headers outside the loop
    headers = {'x-api-key': segmind_apikey[api_key_index]}

    # Create a folder for the UUID if it doesn't exist
    if not os.path.exists(fname):
        os.makedirs(fname)

    num_images = len(prompts)
    requests_made = 0
    start_time = time.time()

    for i, prompt in enumerate(prompts):
        # Handle rate limit (5 requests per minute)
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

        while True:  # Loop to retry with a different API key if quota exceeded
            # Make the API call to Segmind
            response = requests.post(url, json=data, headers=headers)
            requests_made += 1

            if response.status_code == 200 and response.headers.get('content-type') == 'image/jpeg':
                image_data = response.content
                image = Image.open(io.BytesIO(image_data))

                image_filename = os.path.join(fname, f"{i + 1}.jpg")
                image.save(image_filename)

                print(f"Image {i + 1}/{num_images} saved as '{image_filename}'")
                break  # Exit the retry loop if successful
            else:
                print(response.status_code)
                print(response.text)
                if response.status_code == 429:  # Quota exceeded
                    api_key_index = (api_key_index + 1) % len(segmind_apikey)
                    headers['x-api-key'] = segmind_apikey[api_key_index]  # Update header
                    print(f"Switching to API key: {segmind_apikey[api_key_index]}")
                    time.sleep(1)  # Wait a bit
                    continue  # Retry with the new API key
                else:
                    print(f"Error: Failed to retrieve or save image {i + 1}")
                    break  # Exit the retry loop if there's an error other than quota exceeded
