"""
End-to-end pipeline that
1) identifies items in each image,
2) copies the file with a new name to output folder,
3) creates a Facebook-Marketplace-ready JSON post,
4) appends the info to inventory.csv.

Run:  python main.py
"""

import csv, json, re, shutil, os
from pathlib import Path
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from agents.image_identifier import get_image_paths, identify_image

load_dotenv()

# ── LLMs ────────────────────────────────────────────────────────────────────────
vision_llm = ChatOpenAI(model="gpt-4o-2024-11-20", temperature=0)
cheap_llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.2
)
reasoning_llm = ChatGroq(
    model_name="deepseek-r1-distill-llama-70b",
    temperature=0.2
)

# ── Directory Setup ──────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "images", "output")
# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. IDENTIFIER ──────────────────────────────────────────────────────────────
def _id_fn(image_path: str) -> dict:
    desc = identify_image(image_path)
    return {"image_path": image_path, "identification": desc}

image_identifier = RunnableLambda(_id_fn)

# ── 2. COPIER ─────────────────────────────────────────────────────────────────
rename_prompt = PromptTemplate.from_template(
    "Create a concise snake_case file stem (≤ 5 words, no spaces) for: {identification}"
)

def _copy_fn(d: dict) -> dict:
    suggestion = cheap_llm.invoke(rename_prompt.format(**d)).content.strip()
    
    # Get original extension
    orig_path = Path(d["image_path"])
    extension = orig_path.suffix
    
    # Create new path in output directory
    new_filename = f"{suggestion}{extension}"
    new_path = os.path.join(OUTPUT_DIR, new_filename)
    
    # Copy the file (not rename)
    shutil.copy2(d["image_path"], new_path)
    
    # Update the path in dictionary to the new location
    d["original_path"] = d["image_path"]
    d["image_path"] = new_path
    return d

image_copier = RunnableLambda(_copy_fn)

# ── 3. POST GENERATOR ─────────────────────────────────────────────────────────
post_prompt = PromptTemplate.from_template(
    """Respond ONLY with valid JSON.
    Create a Facebook Marketplace listing for the item below.
    Item description: {identification}
    Condition: "Used - like new".
    Return keys: title, description."""
)

def _post_fn(d: dict) -> dict:
    raw_response = reasoning_llm.invoke(post_prompt.format(**d)).content
    # Remove <think>...</think>
    trimmed = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL).strip()
    # Remove ```json and closing ```
    trimmed = re.sub(r"^```json\s*|\s*```$", "", trimmed).strip()
    d["post"] = trimmed
    return d

post_generator = RunnableLambda(_post_fn)

# ── 4. AGGREGATOR ─────────────────────────────────────────────────────────────
CSV_FILE = Path("inventory.csv")

def _agg_fn(d: dict) -> dict:
    header = ["output_file", "title", "description"]
    post   = json.loads(d["post"])
    row    = [
        d["image_path"],
        post["title"],
        post["description"]
    ]
    new_file = not CSV_FILE.exists()
    with CSV_FILE.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(header)
        w.writerow(row)
    return d

aggregator = RunnableLambda(_agg_fn)

# ── PIPELINE ───────────────────────────────────────────────────────────────────
pipeline = (
    RunnablePassthrough()
    | image_identifier
    | image_copier  # Renamed from image_renamer to image_copier
    | post_generator
    | aggregator
)

def main() -> None:
    results = [pipeline.invoke(p) for p in get_image_paths()]
    print(f"Processed {len(results)} image(s).")
    print(f"Original images preserved in raw folder.")
    print(f"Renamed copies saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()