import sys
from pathlib import Path
from src.data.books_catalog_repository import BooksCatalogRepository
from src.services.book_service import BookService

books_repo = BooksCatalogRepository(Path("datasets/books_catalog.csv"))
book_service = BookService()

query = "do you know J.K. Rowling?"
query_lower = query.lower()
suggestions = books_repo.titles() + books_repo.authors()
matched_title = query
for s in sorted(suggestions, key=len, reverse=True):
    s_norm = s.lower().replace(" ", "").replace(".", "")
    q_norm = query_lower.replace(" ", "").replace(".", "")
    if s_norm and s_norm in q_norm:
        matched_title = s
        break

print(f"Matched title: {matched_title}")
res = book_service.search_and_embed(matched_title)
print(f"Context items: {len(res.truth_context)}")

from src.generative.conversational_agent import ConversationalAgent
agent = ConversationalAgent()
facts = agent._extract_context_facts(res.truth_context)
print(f"Facts extracted: {facts}")
matches = agent._query_matches_context(query, facts)
print(f"Matches context: {matches}")
ans = agent.chat(query, res.truth_context)
print(f"Answer: {ans}")

