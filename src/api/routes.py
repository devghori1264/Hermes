from __future__ import annotations

import hmac
import json
from pathlib import Path

from flask import Blueprint, Response, jsonify, render_template, request, abort

from src.config import Settings
from src.domain.models import RecommendationContext
from src.services.recommendation_service import RecommendationService
from src.services.sentiment_service import SentimentService
import unicodedata
from src.services.tmdb_service import TmdbConfig, TmdbService
from src.services.conversational_explanation_service import ConversationalExplanationService
from src.services.domain_router import SemanticDomainRouter
from src.services.music_service import MusicService
from src.services.book_service import BookService
import unicodedata
from src.data.music_catalog_repository import MusicCatalogRepository
from src.data.books_catalog_repository import BooksCatalogRepository
from src.serving.dependencies import ServingDependencies
from src.serving.reliability import CircuitBreakerOpen
from src.serving.ab_testing import ABTestingFramework


SECURITY_BLOCKED_METRIC = "security.blocked"
DEPENDENCY_CIRCUIT_OPEN_METRIC = "dependency.circuit_open"
INVALID_REQUEST_MESSAGE = "invalid request"
TMDB_UNAVAILABLE_MESSAGE = "tmdb unavailable"


def _convert_to_list(raw: str) -> list[str]:
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except json.JSONDecodeError:
        pass
    items = raw.split('","')
    if not items:
        return []
    items[0] = items[0].replace('["', '')
    items[-1] = items[-1].replace('"]', '')
    return items


def _client_key() -> str:
    return request.headers.get("X-Forwarded-For", request.remote_addr or "unknown")


def _rate_limit(deps: ServingDependencies) -> Response | None:
    if deps.rate_limiter.allow(_client_key()):
        return None
    deps.metrics.increment("http.rate_limited", tags={"path": request.path})
    return Response("rate limit exceeded", status=429)


def _json_with_status(payload: dict, status_code: int) -> Response:
    response = jsonify(payload)
    response.status_code = status_code
    return response


def create_blueprint(base_path: Path, deps: ServingDependencies, settings: Settings) -> Blueprint:
    bp = Blueprint("app", __name__)
    recommender = RecommendationService(base_path / "main_data.csv", cache_manager=deps.cache_manager, model_registry=deps.model_registry)
    explainer = ConversationalExplanationService()
    sentiment = SentimentService(base_path / "nlp_model.pkl", base_path / "tranform.pkl", settings.request_timeout_seconds)
    tmdb = TmdbService(TmdbConfig(api_key=settings.tmdb_api_key, timeout_seconds=settings.request_timeout_seconds))
    domain_router = SemanticDomainRouter(model_id=settings.text_encoder_model_id)
    music_service = MusicService()
    book_service = BookService()
    music_repo = MusicCatalogRepository(Path(settings.music_catalog_path))
    books_repo = BooksCatalogRepository(Path(settings.books_catalog_path))

    from src.generative.conversational_agent import ConversationalAgent
    conversational_agent = ConversationalAgent()

    @bp.before_request
    def verify_internal_access() -> None:
        expected = settings.internal_gate_token
        if not expected:
            return
        if request.path.startswith('/api/') or request.path in ['/similarity', '/recommend']:
            token = request.headers.get("X-Admin-Access-Token", "")
            provided = token.encode("utf-8")
            expected_b = expected.encode("utf-8")
            if len(provided) != len(expected_b) or not hmac.compare_digest(provided, expected_b):
                abort(403)

    @bp.route("/", methods=["GET"])
    @bp.route("/home", methods=["GET"])
    def home() -> Response | str:
        return jsonify({"status": "Frontend migrated to React. Run npm run dev."})

    @bp.route("/similarity", methods=["POST"])
    def similarity() -> Response | str:
        limited = _rate_limit(deps)
        if limited is not None:
            return limited
        form_decision = deps.security_guard.validate_form(request.form.to_dict(flat=True))
        if not form_decision.allowed:
            deps.metrics.increment(SECURITY_BLOCKED_METRIC, tags={"reason": form_decision.reason, "path": request.path})
            return Response(INVALID_REQUEST_MESSAGE, status=form_decision.status_code)
        movie = form_decision.fields.get("name", "")
        domain = form_decision.fields.get("domain", "movies")
        if not movie:
            return Response("missing name", status=400)
            
        ab_testing = ABTestingFramework()
        group = ab_testing.assign_group(_client_key(), "ranking_v2_test")
        
        ctx = RecommendationContext(
            query_item_title=movie,
            user_id=_client_key(),
            domain=domain,
            experiment_group=group.model_version
        )
        
        recs = recommender.recommend_titles(movie, ctx)
        if not recs:
            return "Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies"
        return "---".join(recs)

    def _normalize_for_match(text: str) -> str:
        s = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
        return s.lower().replace(" ", "").replace(".", "")

    @bp.route("/api/recommend/explanations", methods=["POST"])
    def recommendation_explanations() -> Response:
        limited = _rate_limit(deps)
        if limited is not None:
            return limited

        if request.is_json:
            payload = request.get_json(silent=True) or {}
            raw_title = str(payload.get("title", ""))
            override_domain = payload.get("domain")
        else:
            raw_title = request.form.get("title", "")
            override_domain = request.form.get("domain")

        decision = deps.security_guard.validate_query_text(raw_title)
        if not decision.allowed:
            deps.metrics.increment(SECURITY_BLOCKED_METRIC, tags={"reason": decision.reason, "path": request.path})
            return _json_with_status({"error": INVALID_REQUEST_MESSAGE, "reason": decision.reason}, decision.status_code)

        classification = domain_router.classify(raw_title, override_domain=override_domain)
        detected_domain = classification.domain

        ranked = []
        def _normalize_for_match(text: str) -> str:
            s = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
            return s.lower().replace(" ", "").replace(".", "")

        if detected_domain == "music":
            query_lower = raw_title.lower()
            matched_title = decision.sanitized_value
            try:
                suggestions = music_repo.titles()
                for s in sorted(suggestions, key=len, reverse=True):
                    if s.lower() in query_lower:
                        matched_title = s
                        break
                from src.domain.models import RankedItem
                result = deps.musicbrainz_breaker.call(music_service.search_and_embed, matched_title)
                ranked = [
                    RankedItem(
                        item_id=c.item_id,
                        title=c.title,
                        score=c.score,
                        explanation="musicbrainz_cold_start",
                        metadata=c.metadata
                    ) for c in result.candidates
                ]
            except Exception:
                pass

        elif detected_domain == "books":
            query_lower = raw_title.lower()
            matched_title = decision.sanitized_value
            try:
                suggestions = books_repo.titles() + books_repo.authors()
                for s in sorted(suggestions, key=len, reverse=True):
                    s_norm = _normalize_for_match(s)
                    q_norm = _normalize_for_match(query_lower)
                    if s_norm and s_norm in q_norm:
                        matched_title = s
                        break
                from src.domain.models import RankedItem
                result = deps.openlibrary_breaker.call(book_service.search_and_embed, matched_title)
                ranked = [
                    RankedItem(
                        item_id=c.item_id,
                        title=c.title,
                        score=c.score,
                        explanation="openlibrary_cold_start",
                        metadata=c.metadata
                    ) for c in result.candidates
                ]
            except Exception:
                pass

        else:
            ranked = recommender.recommend_ranked(
                decision.sanitized_value,
                RecommendationContext(query_item_title=decision.sanitized_value, domain="movies"),
            )

        bundle = explainer.build(decision.sanitized_value, ranked)
        return jsonify(
            {
                "query_title": bundle.query_title,
                "summary": bundle.summary,
                "items": [
                    {
                        "title": item.title,
                        "confidence": item.confidence,
                        "rationale": item.rationale,
                        "primary_signals": list(item.primary_signals),
                    }
                    for item in bundle.items
                ],
            }
        )

    @bp.route("/api/chat", methods=["POST"])
    def chat() -> Response:
        limited = _rate_limit(deps)
        if limited is not None:
            return limited

        payload = request.get_json(silent=True) or {}
        query = str(payload.get("query", ""))
        override_domain = payload.get("domain")

        if not query:
            return _json_with_status({"error": "missing query"}, 400)

        decision = deps.security_guard.validate_query_text(query)
        if not decision.allowed:
            return _json_with_status({"error": INVALID_REQUEST_MESSAGE, "reason": decision.reason}, decision.status_code)

        true_classification = domain_router.classify(query, override_domain=None)
        true_domain = true_classification.domain

        classification = domain_router.classify(query, override_domain=override_domain)
        detected_domain = classification.domain
        context_items: list[str] = []

        if detected_domain == "music":
            query_lower = query.lower()
            matched_title = decision.sanitized_value
            try:
                suggestions = music_repo.titles()
                for s in sorted(suggestions, key=len, reverse=True):
                    s_norm = _normalize_for_match(s)
                    q_norm = _normalize_for_match(query_lower)
                    if s_norm and s_norm in q_norm:
                        matched_title = s
                        break
                result = deps.musicbrainz_breaker.call(music_service.search_and_embed, matched_title)
                context_items = result.truth_context
            except CircuitBreakerOpen:
                deps.metrics.increment(DEPENDENCY_CIRCUIT_OPEN_METRIC, tags={"service": "musicbrainz"})
            except Exception:
                pass

        elif detected_domain == "books":
            query_lower = query.lower()
            matched_title = decision.sanitized_value
            try:
                suggestions = books_repo.titles() + books_repo.authors()
                for s in sorted(suggestions, key=len, reverse=True):
                    s_norm = _normalize_for_match(s)
                    q_norm = _normalize_for_match(query_lower)
                    if s_norm and s_norm in q_norm:
                        matched_title = s
                        break
                result = deps.openlibrary_breaker.call(book_service.search_and_embed, matched_title)
                context_items = result.truth_context
            except CircuitBreakerOpen:
                deps.metrics.increment(DEPENDENCY_CIRCUIT_OPEN_METRIC, tags={"service": "openlibrary"})
            except Exception:
                pass

        elif detected_domain == "movies":
            query_lower = query.lower()
            matched_title = decision.sanitized_value
            suggestions = recommender.suggestions()
            for s in sorted(suggestions, key=len, reverse=True):
                if s.lower() in query_lower:
                    matched_title = s
                    break

            try:
                tmdb_results = deps.tmdb_breaker.call(tmdb.search_movie, matched_title)
                if tmdb_results and "results" in tmdb_results and len(tmdb_results["results"]) > 0:
                    movie_id = tmdb_results["results"][0]["id"]
                    details = deps.tmdb_breaker.call(tmdb.movie_details, movie_id)
                    credits = deps.tmdb_breaker.call(tmdb.movie_credits, movie_id)
                    director = next((crew["name"] for crew in credits.get("crew", []) if crew.get("job") == "Director"), "Unknown")
                    cast = ", ".join([c["name"] for c in credits.get("cast", [])[:5]])
                    overview = details.get("overview", "")
                    release_date = details.get("release_date", "Unknown")
                    tmdb_context = f"Truth File: Movie: {details.get('title', matched_title)} | Director: {director} | Cast: {cast} | Release: {release_date} | Overview: {overview}"
                    context_items.append(tmdb_context)
            except Exception:
                pass

            if not context_items:
                try:
                    df = recommender.repo.load_catalog()
                    matched_rows = df[df["movie_title"].str.lower() == matched_title.lower()]
                    if not matched_rows.empty:
                        row = matched_rows.iloc[0]
                        director = str(row.get("director_name", "Unknown"))
                        cast_list = [str(row.get(f"actor_{i}_name", "")) for i in range(1, 4)]
                        cast = ", ".join([c for c in cast_list if c and c != "nan"])
                        local_context = f"Truth File: Movie: {matched_title.title()} | Director: {director} | Cast: {cast}"
                        context_items.append(local_context)
                except Exception:
                    pass

        response_text = conversational_agent.chat(
            query, context_items, detected_domain=detected_domain, true_domain=true_domain
        )

        return jsonify({
            "response": response_text,
            "detected_domain": detected_domain,
            "domain_confidence": classification.confidence,
            "context_items": context_items
        })

    @bp.route("/api/recommend/sub", methods=["POST"])
    def sub_recommendations() -> Response:
        limited = _rate_limit(deps)
        if limited is not None:
            return limited

        payload = request.get_json(silent=True) or {}
        raw_title = str(payload.get("title", ""))
        decision = deps.security_guard.validate_query_text(raw_title)
        if not decision.allowed:
            return _json_with_status({"error": INVALID_REQUEST_MESSAGE}, decision.status_code)

        ranked = recommender.recommend_ranked(
            decision.sanitized_value,
            RecommendationContext(query_item_title=decision.sanitized_value),
        )
        bundle = explainer.build(decision.sanitized_value, ranked)
        sub_items = [
            {
                "title": item.title,
                "confidence": item.confidence,
                "rationale": item.rationale,
                "primary_signals": list(item.primary_signals),
            }
            for item in bundle.items[:4]
        ]
        return jsonify({"source_title": decision.sanitized_value, "items": sub_items})

    @bp.route("/recommend", methods=["POST"])
    def recommend() -> Response | str:
        limited = _rate_limit(deps)
        if limited is not None:
            return limited
        form_decision = deps.security_guard.validate_form(request.form.to_dict(flat=True))
        if not form_decision.allowed:
            deps.metrics.increment(SECURITY_BLOCKED_METRIC, tags={"reason": form_decision.reason, "path": request.path})
            return Response(INVALID_REQUEST_MESSAGE, status=form_decision.status_code)

        fields = form_decision.fields
        title = fields.get("title", "")
        cast_ids = fields.get("cast_ids", "[]")
        cast_names = fields.get("cast_names", "[]")
        cast_chars = fields.get("cast_chars", "[]")
        cast_bdays = fields.get("cast_bdays", "[]")
        cast_bios = fields.get("cast_bios", "[]")
        cast_places = fields.get("cast_places", "[]")
        cast_profiles = fields.get("cast_profiles", "[]")
        imdb_id = fields.get("imdb_id", "")
        poster = fields.get("poster", "")
        genres = fields.get("genres", "")
        overview = fields.get("overview", "")
        vote_average = fields.get("rating", "")
        vote_count = fields.get("vote_count", "")
        release_date = fields.get("release_date", "")
        runtime = fields.get("runtime", "")
        status = fields.get("status", "")
        rec_movies = fields.get("rec_movies", "[]")
        rec_posters = fields.get("rec_posters", "[]")
        domain = fields.get("domain", "movies")

        ab_testing = ABTestingFramework()
        group = ab_testing.assign_group(_client_key(), "ranking_v2_test")

        ctx = RecommendationContext(
            query_item_title=title,
            user_id=_client_key(),
            domain=domain,
            experiment_group=group.model_version
        )
        suggestions = recommender.suggestions()
        rec_movies = _convert_to_list(rec_movies)
        rec_posters = _convert_to_list(rec_posters)
        cast_names = _convert_to_list(cast_names)
        cast_chars = _convert_to_list(cast_chars)
        cast_profiles = _convert_to_list(cast_profiles)
        cast_bdays = _convert_to_list(cast_bdays)
        cast_bios = _convert_to_list(cast_bios)
        cast_places = _convert_to_list(cast_places)
        cast_ids_list = cast_ids.replace("[", "").replace("]", "").split(",")

        for i in range(len(cast_bios)):
            cast_bios[i] = cast_bios[i].replace(r"\n", "\n").replace(r'\"', '"')

        movie_cards = {rec_posters[i]: rec_movies[i] for i in range(min(len(rec_posters), len(rec_movies)))}
        casts = {
            cast_names[i]: [cast_ids_list[i], cast_chars[i], cast_profiles[i]]
            for i in range(min(len(cast_names), len(cast_ids_list), len(cast_chars), len(cast_profiles)))
        }
        cast_details = {
            cast_names[i]: [cast_ids_list[i], cast_profiles[i], cast_bdays[i], cast_places[i], cast_bios[i]]
            for i in range(min(len(cast_names), len(cast_ids_list), len(cast_profiles), len(cast_bdays), len(cast_places), len(cast_bios)))
        }

        try:
            reviews = deps.sentiment_breaker.call(sentiment.imdb_review_sentiment, imdb_id)
            movie_reviews = {r.review: r.label for r in reviews}
        except CircuitBreakerOpen:
            deps.metrics.increment(DEPENDENCY_CIRCUIT_OPEN_METRIC, tags={"service": "sentiment"})
            movie_reviews = {}

        return jsonify({
            "title": title,
            "poster": poster,
            "overview": overview,
            "vote_average": vote_average,
            "vote_count": vote_count,
            "release_date": release_date,
            "runtime": runtime,
            "status": status,
            "genres": genres,
            "movie_cards": movie_cards,
            "reviews": movie_reviews,
            "casts": casts,
            "cast_details": cast_details,
            "suggestions": suggestions,
        })

    @bp.route('/api/tmdb/search', methods=["GET"])
    def tmdb_search() -> Response:
        limited = _rate_limit(deps)
        if limited is not None:
            return limited
        decision = deps.security_guard.validate_query_text(request.args.get('query', ''))
        if not decision.allowed:
            deps.metrics.increment(SECURITY_BLOCKED_METRIC, tags={"reason": decision.reason, "path": request.path})
            return _json_with_status({"error": INVALID_REQUEST_MESSAGE, "reason": decision.reason}, decision.status_code)
        try:
            return jsonify(deps.tmdb_breaker.call(tmdb.search_movie, decision.sanitized_value))
        except CircuitBreakerOpen:
            deps.metrics.increment(DEPENDENCY_CIRCUIT_OPEN_METRIC, tags={"service": "tmdb"})
            return _json_with_status({"error": TMDB_UNAVAILABLE_MESSAGE}, 503)

    @bp.route('/api/tmdb/movie/<int:movie_id>', methods=["GET"])
    def tmdb_movie(movie_id: int) -> Response:
        limited = _rate_limit(deps)
        if limited is not None:
            return limited
        try:
            return jsonify(deps.tmdb_breaker.call(tmdb.movie_details, movie_id))
        except CircuitBreakerOpen:
            deps.metrics.increment(DEPENDENCY_CIRCUIT_OPEN_METRIC, tags={"service": "tmdb"})
            return _json_with_status({"error": TMDB_UNAVAILABLE_MESSAGE}, 503)

    @bp.route('/api/tmdb/movie/<int:movie_id>/credits', methods=["GET"])
    def tmdb_credits(movie_id: int) -> Response:
        limited = _rate_limit(deps)
        if limited is not None:
            return limited
        try:
            return jsonify(deps.tmdb_breaker.call(tmdb.movie_credits, movie_id))
        except CircuitBreakerOpen:
            deps.metrics.increment(DEPENDENCY_CIRCUIT_OPEN_METRIC, tags={"service": "tmdb"})
            return _json_with_status({"error": TMDB_UNAVAILABLE_MESSAGE}, 503)

    @bp.route('/api/tmdb/person/<int:person_id>', methods=["GET"])
    def tmdb_person(person_id: int) -> Response:
        limited = _rate_limit(deps)
        if limited is not None:
            return limited
        try:
            return jsonify(deps.tmdb_breaker.call(tmdb.person_details, person_id))
        except CircuitBreakerOpen:
            deps.metrics.increment(DEPENDENCY_CIRCUIT_OPEN_METRIC, tags={"service": "tmdb"})
            return _json_with_status({"error": TMDB_UNAVAILABLE_MESSAGE}, 503)

    @bp.route('/api/metrics', methods=["GET"])
    def metrics_snapshot() -> Response:
        return jsonify(deps.metrics.snapshot().to_dict())

    return bp
