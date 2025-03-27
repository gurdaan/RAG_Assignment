from typing import List, Dict
from datetime import datetime
import spacy
import weaviate
from weaviate.classes import HybridFusion, Filter, Sort
from weaviate.classes.config import Property, DataType

WEAVIATE_COLLECTION = "NewsArticle2"

class NewsRAGSystem:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.search_weights = {
            'semantic': 0.7,
            'keyword': 0.3,
            'category': 1.2,
            'authors': 1.1,
            'entities': 1.3
        }

    def _hybrid_search_implementation(
        self, 
        query: str, 
        limit: int = 5,
        category_filter: str = None,
        author_filter: List[str] = None,
        date_range: tuple[datetime, datetime] = None
    ) -> List[Dict]:
        """Enhanced hybrid search with metadata filters"""
        entities = [ent.text for ent in self.nlp(query).ents]
        expanded_query = f"{query} {' '.join(entities)}"

        # Build filters
        filters = None
        if category_filter:
            filters = Filter.by_property("category").equal(category_filter)
        if author_filter:
            author_filter = Filter.by_property("authors").contains_any(author_filter)
            filters = filters & author_filter if filters else author_filter
        if date_range:
            date_filter = Filter.by_property("date").greater_or_equal(date_range[0].isoformat()) & \
                         Filter.by_property("date").less_or_equal(date_range[1].isoformat())
            filters = filters & date_filter if filters else date_filter

        with weaviate_connection() as client:
            collection = client.collections.get(WEAVIATE_COLLECTION)
            response = collection.query.hybrid(
                query=expanded_query,
                alpha=self.search_weights['semantic'],
                properties=[
                    "headline^2",
                    f"category^{self.search_weights['category']}",
                    f"short_description^{self.search_weights['keyword']}",
                    "scraped_content",
                    f"authors^{self.search_weights['authors']}",
                    f"entities^{self.search_weights['entities']}"
                ],
                filters=filters,
                limit=limit,
                fusion_type=HybridFusion.RELATIVE_SCORE,
                return_metadata=weaviate.classes.query.MetadataQuery(score=True),
                sort=Sort.by_property("date").descending()
            )

            return [{
                'headline': article.properties['headline'],
                'summary': article.properties['short_description'],
                'category': article.properties['category'],
                'authors': article.properties['authors'],
                'date': article.properties['date'],
                'entities': article.properties['entities'],
                'link': article.properties['link'],
                'confidence': article.metadata.score
            } for article in response.objects]

# Helper function for Weaviate connection
@contextmanager
def weaviate_connection():
    client = weaviate.connect_to_local()
    try:
        yield client
    finally:
        client.close()