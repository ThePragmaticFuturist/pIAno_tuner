import os
import requests
import PyPDF2 # PyPDF2 is commonly used, but pypdf is the maintained fork
import pypdf # Using the more modern pypdf
import json
import time
import logging
import tempfile
from openai import OpenAI, RateLimitError, APIError
from tqdm import tqdm
from dotenv import load_dotenv
import re

# --- Configuration ---
load_dotenv() # Load environment variables from .env file

SOURCE_FILE = "pdf_sources.txt"  # File listing PDF paths/URLs
OUTPUT_FILE = "training_data.jsonl" # Output JSON Lines file
CHUNK_SIZE = 1500   # Characters per chunk (adjust based on context length and desired granularity)
CHUNK_OVERLAP = 200 # Overlap between chunks to maintain context
MAX_RETRIES = 3     # Max retries for OpenAI API calls
RETRY_DELAY = 5     # Seconds to wait before retrying API calls

# --- OpenAI API Configuration ---
# Make sure OPENAI_API_KEY is set in your .env file or environment variables
OPENAI_MODEL = "gpt-3.5-turbo" # Or "gpt-4-turbo-preview", "gpt-4", etc. Choose based on cost/quality needs
# It's HIGHLY recommended to use a model that supports JSON mode if possible (like gpt-4-turbo-preview, gpt-3.5-turbo-1106+)

# --- Prompt Template for Data Generation ---
# Instruct the LLM to generate instruction/response pairs based ONLY on the text.
# Using JSON mode is preferred if the model supports it.
GENERATION_PROMPT_TEMPLATE = """
Based *only* on the following text content extracted from a textbook, generate one relevant and specific instruction-response pair suitable for training an AI assistant.

The instruction should be a clear question or command derivable from the text.
The response should accurately answer the instruction using *only* information present in the provided text.
Do not add any information not present in the text.
Avoid generic instructions like "Summarize the text". Focus on specific concepts, definitions, explanations, or processes mentioned.

Format the output strictly as a single JSON object containing two keys: "instruction" and "response".

Example Input Text:
"Photosynthesis is the process used by plants, algae and cyanobacteria to convert light energy into chemical energy, through a process that uses water and carbon dioxide. Oxygen is released as a byproduct."

Example Output JSON:
{
  "instruction": "Explain the process of photosynthesis based on the provided text.",
  "response": "Photosynthesis is a process where plants, algae, and cyanobacteria use light energy, water, and carbon dioxide to create chemical energy, releasing oxygen."
}


Now, generate the JSON for the following text:

Text Content:
\"\"\"
{text_chunk}
\"\"\"

Output JSON:
"""

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def read_source_list(filepath):
    """Reads the list of PDF sources from a file."""
    if not os.path.exists(filepath):
        logging.error(f"Source file not found: {filepath}")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        sources = [line.strip() for line in f if line.strip()]
    logging.info(f"Read {len(sources)} sources from {filepath}")
    return sources

def download_pdf(url, target_path):
    """Downloads a PDF from a URL."""
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status() # Raise an exception for bad status codes
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logging.info(f"Successfully downloaded PDF from {url} to {target_path}")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download {url}: {e}")
        return False

def extract_text_from_pdf(pdf_path):
    """Extracts text content from a PDF file using pypdf."""
    text = ""
    try:
        reader = pypdf.PdfReader(pdf_path)
        num_pages = len(reader.pages)
        logging.info(f"Extracting text from {num_pages} pages in {os.path.basename(pdf_path)}...")
        for i, page in enumerate(tqdm(reader.pages, desc=f"Reading {os.path.basename(pdf_path)}", unit="page")):
            try:
                page_text = page.extract_text()
                if page_text:
                    # Basic cleaning: replace multiple newlines/spaces
                    cleaned_text = re.sub(r'\s+', ' ', page_text).strip()
                    text += cleaned_text + "\n" # Add newline between pages
            except Exception as e:
                logging.warning(f"Could not extract text from page {i+1} in {os.path.basename(pdf_path)}: {e}")
        logging.info(f"Finished extracting text from {os.path.basename(pdf_path)}.")
        return text
    except Exception as e:
        logging.error(f"Failed to read PDF {pdf_path}: {e}")
        return None

def chunk_text(text, chunk_size, chunk_overlap):
    """Splits text into overlapping chunks."""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
        if start >= len(text): # Avoid infinite loop on very short overlaps
            break
    # Ensure the last part is captured if smaller than chunk_size
    if start < len(text) and len(text) - start < chunk_size and len(text) - start > 0 :
         if not chunks or text[start:] != chunks[-1][-(len(text)-start):]: # Avoid adding duplicate tail
            chunks.append(text[start:])

    # Filter out very small chunks resulting from overlaps near the end
    chunks = [chunk for chunk in chunks if len(chunk.strip()) > chunk_overlap / 2 or len(chunk.strip()) == len(text)] # Keep if larger than half overlap or if it's the only chunk

    logging.info(f"Split text into {len(chunks)} chunks.")
    return chunks


def generate_training_pairs(text_chunk, client, model, prompt_template):
    """Generates instruction/response pairs using the OpenAI API."""
    prompt = prompt_template.format(text_chunk=text_chunk)
    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an AI assistant tasked with creating training data. Follow the user's instructions precisely and output only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                # For models supporting JSON mode (like gpt-4-turbo-preview, gpt-3.5-turbo-1106+):
                response_format={ "type": "json_object" },
                temperature=0.3, # Lower temperature for more deterministic output based on text
                max_tokens=300  # Adjust as needed for expected response length
            )
            content = response.choices[0].message.content
            # Attempt to parse the JSON content
            try:
                data = json.loads(content)
                if isinstance(data, dict) and "instruction" in data and "response" in data:
                    # Basic validation
                    if data["instruction"] and data["response"] and len(data["instruction"]) > 5 and len(data["response"]) > 5:
                         # Add basic check to prevent using placeholder text
                         if "based on the provided text" not in data["instruction"].lower() and \
                            "based on the text" not in data["response"].lower():
                                return data
                         else:
                            logging.warning(f"Generated pair might contain placeholder text: {data}")
                            return None # Skip this pair
                    else:
                        logging.warning(f"Generated JSON missing fields or fields too short: {content}")
                        return None # Skip malformed/short pairs
                else:
                    logging.warning(f"Generated content is not a valid JSON object with required keys: {content}")
                    return None
            except json.JSONDecodeError:
                logging.warning(f"Failed to decode JSON from LLM response: {content}")
                # Optional: Add more robust parsing here if the model struggles with JSON
                return None # Skip if parsing fails

        except RateLimitError as e:
            logging.warning(f"Rate limit exceeded. Retrying in {RETRY_DELAY} seconds... ({retries+1}/{MAX_RETRIES})")
            retries += 1
            time.sleep(RETRY_DELAY)
        except APIError as e:
            logging.error(f"OpenAI API error: {e}. Retrying in {RETRY_DELAY} seconds... ({retries+1}/{MAX_RETRIES})")
            retries += 1
            time.sleep(RETRY_DELAY)
        except Exception as e:
            logging.error(f"An unexpected error occurred during API call: {e}")
            return None # Don't retry on unexpected errors
    logging.error("Max retries reached for API call.")
    return None


def format_for_finetuning(instruction, response):
    """Formats the instruction/response pair into the desired final structure."""
    # Example: Using Alpaca format within a 'text' field, common for TRL's SFTTrainer
    # Adjust this if your fine-tuning script expects a different format
    formatted_text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{response}"
    return json.dumps({"text": formatted_text})


# --- Main Execution ---

def main():
    if not os.getenv("OPENAI_API_KEY"):
        logging.error("OPENAI_API_KEY environment variable not set. Please create a .env file or set the variable.")
        return

    client = OpenAI() # Initializes with API key from environment variable

    pdf_sources = read_source_list(SOURCE_FILE)
    if not pdf_sources:
        return

    generated_count = 0
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile: # Use 'w' to overwrite, 'a' to append
        for source in tqdm(pdf_sources, desc="Processing PDFs"):
            pdf_path = None
            temp_pdf = None

            if source.startswith("http://") or source.startswith("https://"):
                # It's a URL, download it to a temporary file
                temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                if download_pdf(source, temp_pdf.name):
                    pdf_path = temp_pdf.name
                else:
                    temp_pdf.close()
                    os.unlink(temp_pdf.name) # Clean up temp file on download failure
                    continue # Skip to next source if download fails
            elif os.path.exists(source):
                # It's a local file path
                pdf_path = source
            else:
                logging.warning(f"Source not found or invalid URL: {source}")
                continue

            if pdf_path:
                extracted_text = extract_text_from_pdf(pdf_path)
                if extracted_text:
                    text_chunks = chunk_text(extracted_text, CHUNK_SIZE, CHUNK_OVERLAP)
                    logging.info(f"Processing {len(text_chunks)} chunks from {os.path.basename(source)}...")
                    for chunk in tqdm(text_chunks, desc=f"Generating pairs for {os.path.basename(source)}", leave=False):
                         if len(chunk.strip()) < 50: # Skip very short chunks
                             continue
                         generated_data = generate_training_pairs(chunk, client, OPENAI_MODEL, GENERATION_PROMPT_TEMPLATE)
                         if generated_data:
                            # Format and write to JSONL file
                            finetuning_entry = format_for_finetuning(generated_data["instruction"], generated_data["response"])
                            outfile.write(finetuning_entry + "\n")
                            generated_count += 1
                            # Optional: Add a small delay to avoid hitting rate limits too quickly
                            time.sleep(0.5) # Adjust as needed

                # Clean up temporary file if it was created
                if temp_pdf:
                    try:
                        temp_pdf.close()
                        os.unlink(pdf_path)
                    except Exception as e:
                         logging.warning(f"Could not clean up temp file {pdf_path}: {e}")


    logging.info(f"--- Processing Complete ---")
    logging.info(f"Successfully generated {generated_count} instruction/response pairs.")
    logging.info(f"Training data saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
