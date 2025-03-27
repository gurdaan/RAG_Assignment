from autogen import AssistantAgent, UserProxyAgent, register_function, GroupChatManager, GroupChat, Agent
import weaviate
import spacy
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
from contextlib import contextmanager
from weaviate.classes.query import HybridFusion, Filter, Sort
from weaviate.classes.config import Property, DataType
from datetime import datetime

# Load environment variables
load_dotenv()

# Configuration Constants
WEAVIATE_COLLECTION = "NewsArticle2"
config_list = [
    {
        'model': os.getenv('LLM_MODEL'),
        'api_key': os.getenv('LLM_API_KEY')
    }
]
LLM_CONFIG = {
    "config_list": config_list,
    "temperature": int(os.getenv('LLM_TEMPERATURE')),
}

@contextmanager
def weaviate_connection():
    """Context manager for Weaviate connection"""
    client = weaviate.connect_to_local()
    try:
        yield client
    finally:
        client.close()

from typing import List, Dict, Optional
from datetime import datetime
import spacy
import weaviate
from weaviate.classes.query import Filter, HybridFusion, Sort

class NewsRAGSystem:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.search_weights = {
            'semantic': 0.4,
            'keyword': 0.6,
            'category': 1.2,  # NEW
            'authors': 0.8
        }

    def _hybrid_search_implementation(
        self, 
        query: str, 
        limit: int = 5,
        category: Optional[str] = None,  # NEW
        authors: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,  # NEW
        end_date: Optional[datetime] = None  # NEW
    ) -> List[Dict]:
        """Enhanced search with multiple fallback strategies"""
        try:
            # Normalize author names
            normalized_authors = [self._normalize_name(a) for a in authors] if authors else []
            
            # First attempt: Exact match search
            results = self._execute_search(query, limit, normalized_authors, exact_match=True)
            
            # Second attempt: Partial match fallback
            if not results:
                results = self._execute_search(query, limit*2, normalized_authors, exact_match=False)
                results = self._filter_authors(results, normalized_authors)[:limit]
                
            # Final fallback: Unfiltered semantic search
            if not results:
                results = self._execute_search(query, limit, None)[:limit]
                
            return results
            
        except Exception as e:
            print(f"Search error: {str(e)}")
            return []

    def _normalize_name(self, name: str) -> str:
        """Normalize author names for better matching"""
        return name.lower().replace("dr.", "").replace("phd", "").strip()

    def _execute_search(self, query: str, limit: int, authors: List[str],category=None, exact_match: bool = True):
        """Core search execution with flexible filters"""
        filters = None
         # Category filter
        if category:
            filters = Filter.by_property("category").equal(category)

        if authors:
            if exact_match:
                filters = Filter.by_property("authors").contains_any(authors)
            else:
                filter_conditions = [Filter.by_property("authors").like(f"*{name}*") for name in authors]
                filters = Filter.any_of(*filter_conditions)

        with weaviate_connection() as client:
            collection = client.collections.get(WEAVIATE_COLLECTION)
            response = collection.query.hybrid(
                query=query,
                alpha=self.search_weights['semantic'],
                limit=limit,
                fusion_type=HybridFusion.RELATIVE_SCORE,
                filters=filters,
                return_metadata=weaviate.classes.query.MetadataQuery(score=True)
            )
            return self._format_results(response.objects)

    def _filter_authors(self, results: List[Dict], target_authors: List[str]) -> List[Dict]:
        """Prioritize results with author matches"""
        scored_results = []
        for article in results:
            score = 0
            for author in article['authors']:
                norm_author = self._normalize_name(author)
                for target in target_authors:
                    if target in norm_author:
                        score += 1
            scored_results.append((article, score))
        
        # Sort by match score then by confidence
        return [a for a, _ in sorted(scored_results, key=lambda x: (-x[1], -x[0]['confidence']))]

    def _format_results(self, objects):
        return [{
            'headline': art.properties['headline'],
            'summary': art.properties['short_description'],
            'authors': art.properties['authors'],
            'link': art.properties['link'],
            'confidence': art.metadata.score
        } for art in objects]

class LinkedInGenerator:
    def __init__(self):
        self.entity_blacklist = {'late Tuesday', 'second', 'two'}

    def _generate_post_implementation(self, summary: str, entities: List[str] = None) -> str:
        """Implementation of post generation"""
        entities = entities or []
        filtered_entities = [e for e in entities if e not in self.entity_blacklist]
        hashtags = " ".join(f"#{e.replace(' ', '')}" for e in filtered_entities[:3])
        return f"📢 🔥 Breaking News Alert! 🔥\n\n{summary}\n\n{hashtags}"

# Standalone functions for tool registration
def hybrid_search(
    query: str, 
    limit: int = 5,
    category: Optional[str] = None,
    authors: Optional[List[str]] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> List[Dict]:
    return NewsRAGSystem()._hybrid_search_implementation(
        query, limit, category, authors, start_date, end_date
    )

def generate_post(summary: str, entities: Optional[List[str]] = None) -> str:
    return LinkedInGenerator()._generate_post_implementation(summary, entities)

async def extract_agent_results(chat_history):
    """
    Extract structured results from chat history with marker support
    Returns: dict with 'news_summary', 'linkedin_prep', and 'linkedin_post'
    """
    results = {
        'news_summary': None,
        'linkedin_prep': None,
        'linkedin_post': None,
        'router_message': None
    }
    
    for message in chat_history:
        content = message['content']
        
        # Capture Router's decision for debugging
        if message['name'] == "Router" and ('[' in content and ']' in content):
            results['router_message'] = content
            
        # NewsAgent outputs
        if message['name'] == "NewsAgent":
            if '<<NEWS_SUMMARY_START>>' in content:
                results['news_summary'] = await extract_between_markers(content, '<<NEWS_SUMMARY_START>>', '<<NEWS_SUMMARY_END>>')
            if '<<LINKEDIN_PREP_START>>' in content:
                results['linkedin_prep'] = await extract_between_markers(content, '<<LINKEDIN_PREP_START>>', '<<LINKEDIN_PREP_END>>')
        
        # LinkedInAgent outputs
        elif message['name'] == "LinkedInAgent":
            if '<<LINKEDIN_POST_START>>' in content:
                results['linkedin_post'] = await extract_between_markers(content, '<<LINKEDIN_POST_START>>', '<<LINKEDIN_POST_END>>')
            elif not results['linkedin_post'] and content.strip():  # Fallback for unmarked content
                results['linkedin_post'] = content
    
    return results

async def extract_between_markers(text, start_marker, end_marker):
    """Helper to extract text between markers"""
    try:
        return text.split(start_marker)[1].split(end_marker)[0].strip()
    except:
        return None

async def query_agent(query):
    try:
        news_info, linkedin_info = None, None
        # Create all agents
        router_agent = AssistantAgent(
        name="Router",
        system_message="""You are an intelligent router that ONLY outputs which agent should handle the request next. Follow these rules:

            1. Routing Decisions:
            - For pure news requests: [NewsAgent]
            - For pure LinkedIn post requests: [LinkedInAgent]
            - For combined requests: [NewsAgent, LinkedInAgent] (in this exact format)

            2. Important Constraints:
            - NEVER summarize content
            - NEVER add commentary
            - ONLY output the agent name(s) in brackets
            - For combined requests, ALWAYS use the exact format: [NewsAgent, LinkedInAgent]

            3. Examples:
            User: "Latest news about Biden" → [NewsAgent]
            User: "Create LinkedIn post about Biden" → [LinkedInAgent]
            User: "News about Biden and make a LinkedIn post" → [NewsAgent, LinkedInAgent]""",
            llm_config=LLM_CONFIG,
            human_input_mode="NEVER"
        )

        # Create AutoGen agents
        news_assistant = AssistantAgent(
        name="NewsAgent",
            system_message="""You are a News Research Specialist with access to a comprehensive news retrieval system. Your responsibilities:

            1. Information Retrieval:
            - Use hybrid_search to find the most relevant, recent articles
            - Available search parameters (use as needed):
            * query: (required) Main search terms
            * limit: Number of results (default: 3)
            * category: Business/Tech/Sports/etc.
            * time_frame: Last 24hrs/week/month
            * sources: Specific publications

            2. IMP. Response Guidelines:
            - Format your response with clear markers:
                <<NEWS_SUMMARY_START>>
                [Your news summary here]
                <<NEWS_SUMMARY_END>>
            - Always include sources with links

            3. Context Awareness:
            - Check if you've already provided similar information in this conversation. If yes, respond with: "I have already provided this information." 
            - Never give information about Linkedin post. It is the responsibility of LinkedInAgent.
            - Highlight any notable statistics or quotes""",
                llm_config=LLM_CONFIG
            )

        social_media_assistant = AssistantAgent(
        name="LinkedInAgent",
            system_message=f"""You are a LinkedIn Content Specialist that creates professional, engaging posts. Follow these guidelines:

            1. Post Creation:
            - Required parameter: 
            * summary: Core content (from user or NewsAgent)
            - Optional parameters:
            * tone: Professional/Casual/Inspirational/etc.
            * call_to_action: What readers should do next
            * hashtags: Relevant industry tags
            * mentions: People/companies to tag

            2. Content Rules:
            - Length: 3-5 short paragraphs (250-500 chars)
            - Structure:
            1) Hook/attention-grabber
            2) Key insights/value proposition
            3) Personal perspective/analysis
            4) Call-to-action/question

            - Format your posts with markers:
            <<LINKEDIN_POST_START>>
            [Your post content here]
            <<LINKEDIN_POST_END>>
            - Include 3-5 relevant hashtags at the end

            3. Special Cases:
            - If content comes from NewsAgent:
            * Attribute sources properly
            * Add "Thoughts?" to encourage discussion
            - For user-provided content:
            * Maintain original intent
            * Professionalize casual language
            * Suggest improvements if needed

            4. Revision Handling:
            - If reposting similar content:
            * "I notice this is similar to our last post. Would you like to:
                a) Repost with updates?
                b) Try a different angle?
                c) Combine with new information?""",
                llm_config=LLM_CONFIG
            )

        user_proxy = UserProxyAgent(
            name="UserProxy",
            human_input_mode="NEVER",
            code_execution_config=False,
            system_message="Always request Router agent to suggest the next agent.",
            is_termination_msg=lambda msg: isinstance(msg, dict) and msg.get("content") is not None and "TERMINATE" in msg.get("content", ""),
        )

        # Register tools
        register_function(
            hybrid_search,
            caller=news_assistant,
            executor=user_proxy,
            name="hybrid_search",
            description="Search news articles. Parameters: query (required), limit=5, category, authors, start_date, end_date"
        )

        register_function(
            generate_post,
            caller=social_media_assistant,
            executor=user_proxy,
            name="generate_post",
            description="Generate LinkedIn posts. Parameters: summary (required), entities (optional list)"
        )

        # Configure GroupChat
        groupchat = GroupChat(
            agents=[news_assistant, social_media_assistant, user_proxy],
            messages=[],
            max_round=8,
            send_introductions=True,
            speaker_selection_method= 'auto',
        )

        manager = GroupChatManager(groupchat=groupchat, llm_config=LLM_CONFIG)

        # Process queries
        print(f"\n{'='*50}\nProcessing Query: {query}\n{'='*50}")
        result =await user_proxy.a_initiate_chat(manager, message=query)

        # # Extract results with marker support
        agent_results = await extract_agent_results(groupchat.messages)

        print("\n--- Extracted Information ---")

        if agent_results['news_summary']:
            print(f"\nNews Summary:\n{agent_results['news_summary']}")
            news_info = agent_results['news_summary']
        else:
            print("\nNo news summary generated.")
            news_info="\nNo news summary generated."

        if agent_results['linkedin_post']:
            print(f"\nLinkedIn Post:\n{agent_results['linkedin_post']}")
            linkedin_info = agent_results['linkedin_post']
        else:
            print("\nNo LinkedIn post generated.")
            linkedin_info="\nNo LinkedIn post generated."

        return news_info, linkedin_info
    except Exception as e:
        print(f"Error in query_agent: {str(e)}")
        return None, None


if __name__ == "__main__":
    # Test the query_agent function
    query = "Latest news about AI and create a LinkedIn post"
    news_info, linkedin_info = query_agent(query)
    print(f"\nNews Info: {news_info}")
    print(f"\nLinkedIn Info: {linkedin_info}")