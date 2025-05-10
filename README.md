# Image Processing Pipeline for Marketplace Listings

This tool automates the process of preparing images for online marketplace listings by:

1. **Identifying items** in images using vision AI
2. **Copying and renaming** files to a structured format
3. **Generating marketplace listings** including title and description
4. **Tracking inventory** in a CSV file

## Setup

### Prerequisites

- Python 3.8+
- Required packages:
  ```
  pip install langchain langchain-groq openai python-dotenv
  ```

### API Keys

Create a `.env` file in the project root with:

```
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
```

### Directory Structure

The tool expects the following directory structure:
```
project_root/
├── main.py
├── .env
├── agents/
│   └── image_identifier.py
└── images/
    ├── raw/
    │   └── [your_images_here]
    └── output/
        └── [processed_images_will_go_here]
```

## Usage

1. Place images in the `images/raw/` directory
2. Run the script:
   ```
   python main.py
   ```
3. Check the results:
   - Renamed image copies in `images/output/`
   - Inventory data in `inventory.csv`

## How It Works

### 1. Image Identification

Uses OpenAI's GPT-4o vision model to analyze each image and generate detailed descriptions of the items, including:
- Item type and brand/model
- Color, material, and size characteristics
- Condition and wear details
- Included accessories or context

### 2. File Management

- Preserves original images in `images/raw/`
- Creates renamed copies in `images/output/` with SEO-friendly filenames
- Names are derived from the AI's identification using a lightweight LLM

### 3. Marketplace Listing Generation

Uses Groq's DeepSeek model to create marketplace-ready listings from the image descriptions:
- Generates appropriate titles
- Creates detailed descriptions
- Sets condition as "Used - like new"

### 4. Inventory Tracking

Maintains a CSV file (`inventory.csv`) with:
- Original image path
- New image path
- Listing title
- Listing description

## Models Used

- **OpenAI GPT-4o**: Vision analysis for detailed item identification
- **Llama 3.1 8B**: Lightweight processing for filename generation
- **DeepSeek-R1-Distill-Llama-70B**: More powerful reasoning for marketplace listing creation

## Customization

- Edit the prompts in `main.py` to adjust the style of descriptions, listings, or filenames
- Modify the CSV format in the `_agg_fn` function to store additional data
- Change LLM models based on your preference for speed vs. quality
