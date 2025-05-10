# agents/image_identifier.py
import os, base64
from typing import List
import openai

RAW_IMAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images", "raw")

def get_image_paths() -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    return [
        os.path.join(RAW_IMAGE_DIR, f)
        for f in os.listdir(RAW_IMAGE_DIR)
        if os.path.splitext(f)[1].lower() in exts
    ]

def identify_image(image_path: str) -> str:
    """
    Return a **detailed, multi-sentence description** of the main item and any
    notable context in the photo.  The response should mention:
      • what the item is and its apparent brand/model (if recognizable)  
      • colour / material / size cues  
      • visible condition or wear  
      • included accessories, packaging or background elements that matter for resale
    """
    with open(image_path, "rb") as img:
        img_b64 = base64.b64encode(img.read()).decode()
    resp = openai.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Please describe this image in as much detail as possible "
                            "for an online product listing. Include the primary object, "
                            "brand or make (if obvious), colour, notable features, "
                            "visible condition, and any accessories or context."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    },
                ],
            }
        ],
        max_tokens=128,
        temperature=0,
    )
    return resp.choices[0].message.content.strip()
