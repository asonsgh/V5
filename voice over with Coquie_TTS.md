Example to create Voice over with Coquie TTS 

import os
import json
from pprint import pprint
import time
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

import g4f
import json
from pprint import pprint
import time
from json import JSONDecodeError

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


LANGUAGE = "en" # @param ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"]
VOICE = "Ana Florence" # @param ["Claribel Dervla", "Daisy Studious", "Gracie Wise", "Tammie Ema", "Alison Dietlinde", "Ana Florence", "Annmarie Nele", "Asya Anara", "Brenda Stern", "Gitta Nikolina", "Henriette Usha", "Sofia Hellen", "Tammy Grit", "Tanja Adelina", "Vjollca Johnnie", "Andrew Chipper", "Badr Odhiambo", "Dionisio Schuyler", "Royston Min", "Viktor Eka", "Abrahan Mack", "Adde Michal", "Baldur Sanjin", "Craig Gutsy", "Damien Black", "Gilberto Mathias", "Ilkin Urbano", "Kazuhiko Atallah", "Ludvig Milivoj", "Suad Qasim", "Torcull Diarmuid", "Viktor Menelaos", "Zacharie Aimilios", "Nova Hogarth", "Maja Ruoho", "Uta Obando", "Lidiya Szekeres", "Chandra MacFarland", "Szofi Granger", "Camilla Holmström", "Lilya Stainthorpe", "Zofija Kendrick", "Narelle Moon", "Barbora MacLean", "Alexandra Hisakawa", "Alma María", "Rosemary Okafor", "Ige Behringer", "Filip Traverse", "Damjan Chapman", "Wulf Carlevaro", "Aaron Dreschner", "Kumar Dahl", "Eugenio Mataracı", "Ferran Simen", "Xavier Hayasaka", "Luis Moray", "Marcos Rudaski"] {allow-input: true}

# Initialize TTS
os.environ["COQUI_TOS_AGREED"] = "1"  # Assuming agreement to TOS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
device = "cuda" if torch.cuda.is_available() else "cpu"
tts.to(device)

def force_duration(duration: float, path: str):
    audio_clip = mp.AudioFileClip(path)
    if audio_clip.duration > duration:
        speed_factor = audio_clip.duration / duration
        new_audio = audio_clip.fx(mp.vfx.speedx, speed_factor, final_duration=duration)
        new_audio.write_audiofile(path, codec="libmp3lame")
    audio_clip.close()

def synthesize(text: str, path: str, to_force_duration: bool, duration: float) -> float:
    tts.tts_to_file(text=text, file_path=path, language=LANGUAGE, speaker=VOICE)
    if to_force_duration:
        force_duration(float(duration), path)
    return get_audio_duration(path)

def get_audio_duration(path: str):
    audio_clip = mp.AudioFileClip(path)
    duration = audio_clip.duration
    audio_clip.close()
    return duration

def main():
    for i, text in enumerate(texts):
        path = f"{i+1}.mp3"
        to_force_duration = False
        duration = 57
        synthesize(text, path, to_force_duration, duration)
if __name__ == "__main__":
    main()
