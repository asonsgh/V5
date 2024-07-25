this is example of generating images with segmand ai api

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
