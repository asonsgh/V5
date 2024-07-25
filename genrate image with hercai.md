this is the example of genrate image with hercai
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


def generate_images_hercai(prompts, fname):
    image_model = "animefy"

    # Create a folder for the UUID if it doesn't exist
    if not os.path.exists(fname):
        os.makedirs(fname)  # Ensure directory is created

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
                    image_filename = os.path.join(fname, f"{i + 1}.png")  # Use .png extension
                    image.save(image_filename)
                    print(f"Image {i + 1}/{num_images} saved as '{image_filename}'")
                else:
                    print(f"Error: Failed to retrieve image from URL for image {i + 1}")
            else:
                print(f"Error: No image URL in response for image {i + 1}")
        else:
            print(response.text)
            print(f"Error: Failed to retrieve or save image {i + 1}")

# Call the appropriate function based on the user's choice
if video_source == "segmind":
    generate_images_segmind(image_prompts, active_folder)
elif video_source == "hercai":
    generate_images_hercai(image_prompts, active_folder)
else:
    print("Invalid image generator choice. Please choose either 'segmind' or 'hercai'.")
