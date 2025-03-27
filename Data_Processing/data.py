import kagglehub
import json
import os
import requests
from bs4 import BeautifulSoup
import spacy
import weaviate
import configparser
import logging
from weaviate.classes.config import Configure, Property, DataType

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data_processing.log"),  # Save logs to a file
        logging.StreamHandler()  # Print logs to the console
    ]
)
logger = logging.getLogger(__name__)

# Initialize spaCy for NER
nlp = spacy.load("en_core_web_sm")

def load_weaviate_config(config_file="../config.ini"):
    """Loads Weaviate configuration from a config.ini file."""
    config = configparser.ConfigParser()
    try:
        config.read(config_file)
        logger.info("Successfully loaded Weaviate configuration.")
        return config
    except configparser.Error as e:
        logger.error(f"Error reading config file: {e}")
        return None

# Step 1: Download the dataset
def download_dataset():
    logger.info("Downloading dataset...")
    try:
        path = kagglehub.dataset_download("rmisra/news-category-dataset")
        logger.info(f"Dataset downloaded to: {path}")
        return path
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise

# Step 2: Scrape article content from URL
def scrape_article(url):
    try:
        logger.info(f"Scraping article from URL: {url}")
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        # Check if the target element exists
        main_content = soup.find("div", class_="entry__text")
        if main_content:
            logger.info("Successfully scraped article content.")
            return main_content.get_text(separator="\n").strip()
        else:
            logger.warning(f"Target element not found in {url}.")
            return None
    except Exception as e:
        logger.error(f"Failed to scrape {url}: {str(e)}")
        return None

# Step 3: Extract entities from text
def extract_entities(text):
    logger.info("Extracting entities from text...")
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    logger.info(f"Extracted entities: {entities}")
    return entities

def save_to_weaviate(client, data):
    logger.info("Saving data to Weaviate...")
    
    try:
        # Delete existing collection if it exists
        if client.collections.exists("NewsArticle2"):
            client.collections.delete("NewsArticle2")
            logger.info("Deleted existing NewsArticle2 collection")

        # Create new collection with proper configuration
        news_articles = client.collections.create(
            name="NewsArticle2",
            description="A collection of news articles",
            # Change to this configuration in your Python code
            vectorizer_config=Configure.Vectorizer.text2vec_ollama(
                api_endpoint="http://host.docker.internal:11434",  # For Docker-to-host communication
                model="nomic-embed-text"
            ),
            generative_config=Configure.Generative.ollama(
                api_endpoint="http://host.docker.internal:11434",
                model="llama3"  # Corrected model name
            ),
            properties=[
                Property(
                    name="headline",
                    data_type=DataType.TEXT,
                    description="The headline of the article",
                    indexFilterable=True,  # Enable filtering
                    indexSearchable=True   # Enable full-text search
                ),
                Property(
                    name="short_description",
                    data_type=DataType.TEXT,
                    description="A short description of the article",
                    indexFilterable=True,  # Enable filtering
                    indexSearchable=True   # Enable full-text search
                ),
                Property(
                    name="scraped_content",
                    data_type=DataType.TEXT,
                    description="The full content of the article (scraped)",
                    indexFilterable=True,  # Enable filtering
                    indexSearchable=True   # Enable full-text search
                ),
                Property(
                    name="category",
                    data_type=DataType.TEXT,
                    description="The category of the article",
                    indexFilterable=True,  # Enable filtering
                    indexSearchable=True   # Enable full-text search
                ),
                Property(
                    name="authors",
                    data_type=DataType.TEXT_ARRAY,
                    description="The authors of the article",
                    indexFilterable=True,  # Enable filtering
                    indexSearchable=True   # Enable full-text search    
                ),
                Property(
                    name="date",
                    data_type=DataType.DATE,
                    description="The publication date of the article",
                    indexFilterable=True,  # Enable filtering
                    indexSearchable=True   # Enable full-text search
                ),
                Property(
                    name="link",
                    data_type=DataType.TEXT,
                    description="The URL of the article",
                    indexFilterable=True,  # Enable filtering
                    indexSearchable=True   # Enable full-text search
                ),
                Property(
                    name="entities",
                    data_type=DataType.TEXT_ARRAY,
                    description="Entities extracted from the article",
                    indexFilterable=True,  # Enable filtering
                    indexSearchable=True   # Enable full-text search    
                )
            ]
        )
        logger.info("Created NewsArticle2 collection with Ollama integration")

        # Batch import
        with news_articles.batch.dynamic() as batch:
            for idx, article in enumerate(data):
                try:
                    # Process content
                    if article["link"]:
                        scraped_content = scrape_article(article["link"])
                        article["scraped_content"] = scraped_content or article["short_description"]
                    else:
                        article["scraped_content"] = article["short_description"]

                    # Extract entities
                    article["entities"] = extract_entities(article["scraped_content"])

                    
                    
                    # Format date with time component
                    formatted_date = f"{article['date']}T00:00:00Z"

                    # Add to batch
                    batch.add_object({
                        "headline": article["headline"],
                        "short_description": article["short_description"],
                        "scraped_content": article["scraped_content"],
                        "category": article["category"],
                        "authors": article["authors"].split(", "),  # Convert comma-separated string to list
                        "date": formatted_date,
                        "link": article["link"],
                        "entities": article["entities"]
                    })

                    logger.debug(f"Added article {idx + 1}/{len(data)}: {article['headline'][:50]}...")

                except Exception as e:
                    logger.error(f"Error processing article {idx + 1}: {str(e)}")
                    if batch.number_errors > 10:
                        logger.error("Too many errors, aborting batch import")
                        break

        # Handle failed objects
        if failed := news_articles.batch.failed_objects:
            logger.error(f"Failed to import {len(failed)} objects")
            for i, obj in enumerate(failed[:3]):
                logger.error(f"Error {i + 1}: {obj}")
        else:
            logger.info(f"Successfully imported {len(data)} articles")

    except Exception as e:
        logger.error(f"Critical error in save_to_weaviate: {str(e)}")
        raise    

def process_dataset(path):
    logger.info("Processing dataset...")
    json_file_path = os.path.join(path, "News_Category_Dataset_v3.json")
    data = []

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                article = json.loads(line)
                data.append(article)
                if len(data) >= 100:  # Process first 100 articles for testing
                    break
        logger.info(f"Processed {len(data)} articles.")
    except FileNotFoundError:
        logger.error(f"Error: JSON file not found at {json_file_path}")
    except json.JSONDecodeError:
        logger.error(f"Error: Invalid JSON format in the file.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")

    return data

# Main function
def main():
    try:
        logger.info("Starting data processing pipeline...")

        client = weaviate.connect_to_local()
        logger.info("Weaviate client initialized successfully.")

        # Step 1: Download dataset
        dataset_path = download_dataset()

        # Step 2: Process dataset
        data = process_dataset(dataset_path)

        # Step 3: Save data to Weaviate
        save_to_weaviate(client, data)
        logger.info("Data saved to Weaviate successfully!")
    except Exception as e:
        logger.error(f"Exception occurred: {e}")

if __name__ == "__main__":
    main()