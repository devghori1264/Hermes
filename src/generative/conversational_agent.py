import random
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List


@dataclass
class DialogueTurn:
    user_query: str
    agent_response: str
    retrieved_items: List[str] = field(default_factory=list)


_GREETING_SET = frozenset({
    "hi", "hii", "hello", "hey", "greetings",
    "good morning", "good evening", "good afternoon",
    "yo", "sup", "howdy", "namaste"
})

_IDENTITY_TRIGGERS = (
    "who are you", "what is your name", "introduce yourself",
    "what are you", "tell me about yourself", "how are you",
    "what can you do"
)

_GREETING_POOL = (
    "Hey! I'm Hermes, a Universal Intelligence created by Devendra Ghori.",
    "Hello! My name is Hermes. I am a Universal Intelligence engineered by Devendra Ghori.",
    "Hi there! I am Hermes, an advanced Universal Intelligence brought to life by Devendra Ghori.",
    "Greetings! I go by Hermes, a Universal Intelligence created by Devendra Ghori to assist you.",
    "Hello! I'm Hermes, your Universal Intelligence companion, designed by Devendra Ghori.",
    "Hi! Hermes here, a Universal Intelligence crafted by Devendra Ghori. What can I help you explore?",
)

_BEYOND_KNOWLEDGE_POOL = (
    "Sorry about this! You actually managed to ask something that's totally outside my current knowledge base. It's definitely not you, it's just a gap in my training! I'm making a note of this so I can learn it for my next update.",
    "Ah, my apologies! I really wish I had a good answer for you, but I'm drawing a blank. It's a totally fair question, my developers just haven't taught me about it yet. I'll flag this to be included in my next release.",
    "Oops, you caught me off guard! I don't actually have the facts on this one yet. Don't worry, your prompt makes perfect sense; I’m just still in my learning phase for this specific topic. I'll make sure I study up on it for the next update.",
    "Sorry! I hate to leave you hanging, but this one is slightly beyond what I currently know. I'm taking note of it right now so I can gather the right info and learn all about it before my next release.",
    "My bad! You'd think I'd know this, but it completely missed my training radar. You asked a perfectly valid question; I'm just playing catch up on my end. I'll make sure to grab this info for my next update.",
    "Oh, sorry about that! I actually don't have enough solid info to give you a proper answer right now. Please don't think it was a bad question; I just have a few blind spots to fill! I'll flag this so I can learn it for next time.",
    "Ah, I'm really sorry, but you've managed to stump me! It's a great question, but it sits right outside what I've been taught so far. I'll put this on my homework list and make sure I know it for the next update.",
    "Sorry! I'd love to give you an answer, but I'm still learning and haven't quite reached this topic yet. The gap is definitely on my end, not yours! I'll get this sorted out and added to my knowledge base in the next release.",
    "Oh man, I apologize! I really thought I had an answer for you, but I'm coming up empty. It's a completely reasonable question, I just haven't been trained on it yet. I'll flag this for the team so I can learn it for the next update!",
    "So sorry! You've asked something really interesting, but I just don't have the data to back up an answer right now. Definitely not a mistake on your part; I’m just still growing! I'll make sure to learn about this for the next rollout."
)


class ConversationalAgent:
    def __init__(self, model_id: str = "google/flan-t5-large") -> None:
        self.model_id = model_id
        self._tokenizer = None
        self._model = None
        self.history: List[DialogueTurn] = []

    def _ensure_model(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return
        import transformers
        transformers.logging.set_verbosity_error()
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)

    def _is_greeting(self, text: str) -> bool:
        normalized = text.lower().strip().rstrip("?!.")
        if normalized in _GREETING_SET:
            return True
        return any(trigger in normalized for trigger in _IDENTITY_TRIGGERS)

    def _extract_context_facts(self, context: List[str]) -> dict:
        facts = {}
        for entry in context:
            if not entry.startswith("Truth File:"):
                continue
            parts = entry.replace("Truth File: ", "").split(" | ")
            for part in parts:
                if ": " in part:
                    key, value = part.split(": ", 1)
                    facts[key.strip().lower()] = value.strip()
            break
        return facts

    def _detect_domain(self, facts: dict) -> str:
        if facts.get("artist"):
            return "music"
        if facts.get("book"):
            return "books"
        if facts.get("movie"):
            return "movies"
        return "unknown"

    def _entity_names_for_domain(self, facts: dict, domain: str) -> list[str]:
        if domain == "music":
            return [facts.get("artist", "").lower()]
        if domain == "books":
            names = []
            if "book" in facts:
                names.append(facts["book"].lower())
            if "author" in facts:
                names.append(facts["author"].lower())
            return names
        return [facts.get("movie", "").lower()]

    def _query_matches_context(self, user_query: str, facts: dict) -> bool:
        if not facts:
            return False
        domain = self._detect_domain(facts)
        entity_names = self._entity_names_for_domain(facts, domain)
        if not entity_names:
            return False
        
        def _norm(s: str) -> str:
            return unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('utf-8').lower()
            
        query_lower = _norm(user_query)
        query_tokens = set(re.findall(r"[a-z0-9]+", query_lower))
        
        for name in entity_names:
            if not name:
                continue
            name_norm = _norm(name)
            name_tokens = set(re.findall(r"[a-z0-9]+", name_norm))
            if not name_tokens:
                continue
            overlap = name_tokens & query_tokens
            if len(overlap) >= max(1, len(name_tokens) * 0.5):
                return True
        return False

    def _build_music_answer(self, user_query: str, facts: dict) -> str:
        query_lower = user_query.lower()
        name = facts.get("artist", "this artist").title()
        entity_type = facts.get("type", "")
        country = facts.get("country", "")
        genres = facts.get("genres", "")
        born = facts.get("born", "")
        about = facts.get("about", "")

        parts = [f"{name} is a {entity_type.lower()}" if entity_type else f"{name} is an artist"]
        if country:
            parts.append(f"from {country}")
        sentence_one = " ".join(parts) + "."
        sentences = [sentence_one]

        if genres:
            sentences.append(f"Their music spans {genres}.")
        if born:
            sentences.append(f"Born on {born}.")
        if about:
            sentences.append(about + ".")
        return " ".join(sentences)

    def _build_book_answer(self, user_query: str, facts: dict) -> str:
        query_lower = user_query.lower()
        title = facts.get("book", "this book").title()
        author = facts.get("author", "")
        published = facts.get("published", "")
        subjects = facts.get("subjects", "")

        parts = [f"{title} is a book"]
        if author:
            parts.append(f"written by {author}")
        if published:
            parts.append(f"first published in {published}")
        sentence_one = " ".join(parts) + "."
        sentences = [sentence_one]

        if subjects:
            sentences.append(f"The book covers {subjects}.")

        if "who wrote" in query_lower or "author" in query_lower:
            if author:
                return f"The author of {title} is {author}."

        if any(w in query_lower for w in ["when", "publish", "year"]):
            if published:
                return f"{title} was first published in {published}."

        return " ".join(sentences)

    def _build_movie_answer(self, user_query: str, facts: dict) -> str:
        query_lower = user_query.lower()
        movie_name = facts.get("movie", "this movie")
        director = facts.get("director", "")
        cast = facts.get("cast", "")
        overview = facts.get("overview", "")
        release = facts.get("release", "")

        if "director" in query_lower:
            if director:
                return f"The director of {movie_name.title()} is {director}."

        if any(w in query_lower for w in ["cast", "actor", "actress", "star", "who acted"]):
            if cast:
                return f"The cast of {movie_name.title()} includes {cast}."

        if any(w in query_lower for w in ["when", "release", "year", "came out"]):
            if release:
                return f"{movie_name.title()} was released on {release}."

        if any(w in query_lower for w in ["plot", "about", "story", "synopsis"]):
            if overview:
                return f"{movie_name.title()}: {overview}"

        parts = [f"{movie_name.title()} is a film"]
        if director:
            parts.append(f"directed by {director}")
        if release:
            parts.append(f"released on {release}")
        sentence_one = " ".join(parts) + "."
        sentences = [sentence_one]
        if cast:
            sentences.append(f"The film stars {cast}.")
        if overview:
            sentences.append(overview)
        return " ".join(sentences)

    def _build_factual_answer(self, user_query: str, facts: dict) -> str:
        domain = self._detect_domain(facts)
        if domain == "music":
            return self._build_music_answer(user_query, facts)
        if domain == "books":
            return self._build_book_answer(user_query, facts)
        return self._build_movie_answer(user_query, facts)

    def chat(self, user_query: str, retrieved_context: List[str], detected_domain: str = None, true_domain: str = None) -> str:
        if self._is_greeting(user_query):
            response = random.choice(_GREETING_POOL)
            self.history.append(DialogueTurn(
                user_query=user_query,
                agent_response=response,
                retrieved_items=[]
            ))
            return response

        normalized_query = user_query.lower().strip()
        if normalized_query in {"movies", "movie"}:
            return "I can help you explore movies! Search for a film or director in the search bar above, or ask me about any movie."
        if normalized_query in {"music", "songs", "artist", "artists"}:
            return "I can help you explore music! Search for an artist or album in the search bar above, or ask me about your favorite musician."
        if normalized_query in {"books", "book", "author", "authors"}:
            return "I can help you explore books! Search for a novel or author in the search bar above, or ask me about a book you'd like to read."

        facts = self._extract_context_facts(retrieved_context)

        if not self._query_matches_context(user_query, facts):
            if true_domain and detected_domain and true_domain != detected_domain and true_domain in ["music", "movies", "books"]:
                domain_labels = {"music": "Music", "movies": "Movies", "books": "Books"}
                label = domain_labels[true_domain]
                response = f'Hello, you asked the right question in the wrong category. Please try to change or ask again in the "{label}" category.'
            else:
                response = random.choice(_BEYOND_KNOWLEDGE_POOL)
            
            self.history.append(DialogueTurn(
                user_query=user_query,
                agent_response=response,
                retrieved_items=retrieved_context
            ))
            return response

        factual_answer = self._build_factual_answer(user_query, facts)
        if factual_answer:
            self.history.append(DialogueTurn(
                user_query=user_query,
                agent_response=factual_answer,
                retrieved_items=retrieved_context
            ))
            return factual_answer

        response = random.choice(_BEYOND_KNOWLEDGE_POOL)
        self.history.append(DialogueTurn(
            user_query=user_query,
            agent_response=response,
            retrieved_items=retrieved_context
        ))
        return response

    def clear_memory(self) -> None:
        self.history.clear()
