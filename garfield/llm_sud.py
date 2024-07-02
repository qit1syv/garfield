import base64
import requests
from PIL import Image
import argparse 
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import csv 
from tqdm import tqdm 

prompt = """
Example: The first image is an image of a scene. The second image has the same scene from the first image with two entities highlighted in blue and green. Can the green entity be considered a direct child of the blue entity AND can the green entity be described with a single english word or few words?
Example Answer: Yes, the green entity is a mug and the blue entity is a table with many objects, so the green entity is a child of the blue entity. The green entity can be described succintly in english with 'mug'."
Task: Given the third image as the image of the scene and the fourth image highlighting the blue and green entities. Is the green entity a child of the blue entity AND can the green entity be described with a single english word or few words? If you can't see the blue or green entity answer with no. Structure your response as a yes or no then followed by a reason."""
prompt_examples = [("./prompt_examples/scene_1.png", "./prompt_examples/scene_1_mask.png")] # pair of image and mask
api_key = ""
def generate_response(target_image_path, target_mask_path):
    for scene_image_path, mask_image_path in prompt_examples:
        with open(scene_image_path, "rb") as image_file:
            scene_image = base64.b64encode(image_file.read()).decode('utf-8')
        with open(mask_image_path, "rb") as image_file:
            mask_image = base64.b64encode(image_file.read()).decode('utf-8')
    with open(target_image_path, "rb") as image_file:
            target_scene = base64.b64encode(image_file.read()).decode('utf-8')
    with open(target_mask_path, "rb") as image_file:
            target_mask = base64.b64encode(image_file.read()).decode('utf-8')
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
        }
    payload = {
        "model": "gpt-4-vision-preview",
        "seed": 42,
        "temperature": 0.8, 
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{scene_image}",
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{mask_image}",
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{target_scene}",
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{target_mask}",
                        }
                    }
                ]
            },
        ],
        "max_tokens": 500
    }
    success = False
    num_attempts = 0
    gpt_response = None
    while not success and num_attempts < 3:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        try:
            gpt_response = response.json()['choices'][0]['message']['content']
        except:
            num_attempts += 1
            gpt_response = "error"
            print("Error in generating response, generating again...")
            continue
        print(gpt_response)
        if len(gpt_response) > 0 and "request" not in gpt_response and "error" not in gpt_response:
            success = True
        else:
            gpt_response = None
            num_attempts += 1
    return gpt_response

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()