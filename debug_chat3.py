import sys
from pathlib import Path
from src.data.music_catalog_repository import MusicCatalogRepository
from src.services.music_service import MusicService
import unicodedata

def _normalize_for_match(text: str) -> str:
    s = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    return s.lower().replace(" ", "").replace(".", "")

music_repo = MusicCatalogRepository(Path("datasets/music_catalog.csv"))
music_service = MusicService()

query = "suggest me some Beyonce songs."
query_lower = query.lower()
suggestions = music_repo.titles()
matched_title = query
for s in sorted(suggestions, key=len, reverse=True):
    s_norm = _normalize_for_match(s)
    q_norm = _normalize_for_match(query_lower)
    if s_norm and s_norm in q_norm:
        matched_title = s
        break

print(f"Matched title: {matched_title}")
res = music_service.search_and_embed(matched_title)
print(f"Context items: {len(res.truth_context)}")

from src.generative.conversational_agent import ConversationalAgent
agent = ConversationalAgent()
facts = agent._extract_context_facts(res.truth_context)
print(f"Facts extracted: {facts}")
matches = agent._query_matches_context(query, facts)
print(f"Matches context: {matches}")
ans = agent.chat(query, res.truth_context)
print(f"Answer: {ans}")

