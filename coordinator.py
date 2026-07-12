import re
import json
from haystack.dataclasses import ChatMessage


def parse_reviews(text):
    """Zieht das JSON-Array aus der Reviewer-Antwort, auch wenn es in ```-Fences steckt."""
    if not text:
        return []
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return []


def avg_score(reviews):
    scores = [r["score"] for r in reviews if isinstance(r.get("score"), (int, float))]
    return sum(scores) / len(scores) if scores else 0


def run_coordinator(pipeline, query, threshold=6.0, max_rounds=3):
    """Sucht + bewertet mehrfach. Sammelt die besten Paper ueber alle Runden hinweg."""
    best_by_id = {}  # doi (oder titel als fallback) -> review-dict, immer der hoechste score
    current_query = query

    for runde in range(1, max_rounds + 1):
        print(f"\n{'='*60}\n RUNDE {runde}: Suche zu: {current_query}\n{'='*60}")

        reviews = []
        review_text = None
        for versuch in range(1, 3):  # 1 Normalversuch + 1 Retry bei leerer/kaputter Antwort
            msg = ChatMessage.from_user(current_query)
            result = pipeline.run(
                data={"searcher": {"messages": [msg]}},
                include_outputs_from={"reviewer"},
            )
            last_message = result["reviewer"]["last_message"]
            review_text = last_message.text
            reviews = parse_reviews(review_text)

            if reviews:
                break

            print(
                f"Reviewer-Antwort leer/ungueltig (Versuch {versuch}/2). "
                f"Meta: {last_message.meta}"
            )

        if not reviews:
            print("Konnte die Bewertungen auch nach Retry nicht als JSON lesen. Breche ab.")
            print(f"--- Rohe Reviewer-Antwort ---\n{review_text}\n-----------------------------")
            break

        # Beste je Paper merken (Dedup ueber Runden)
        for r in reviews:
            key = r.get("doi") or r.get("titel")
            if key and (key not in best_by_id or r.get("score", 0) > best_by_id[key].get("score", 0)):
                best_by_id[key] = r

        schnitt = avg_score(reviews)
        print(f"\nDurchschnitts-Score dieser Runde: {schnitt:.1f} (Schwelle: {threshold})")

        if schnitt >= threshold:
            print("Gut genug - fertig.")
            break

        if runde < max_rounds:
            schwach = [r["begruendung"] for r in reviews if r.get("score", 0) < threshold]
            kritik = " ".join(schwach)[:500]

            print(f"\nDie Ergebnisse waren eher schwach. Kritik: {kritik}")
            eigene_anfrage = input(
                "Moechtest du eine spezifischere Anfrage eingeben? "
                "(Enter = automatisch verfeinern): "
            ).strip()

            if eigene_anfrage:
                current_query = eigene_anfrage
            else:
                # Kritik der schwachen Paper nutzen, um die naechste Suche zu verfeinern
                current_query = (
                    f"{query}. Vorherige Ergebnisse waren zu unspezifisch. "
                    f"Kritik: {kritik}. Suche gezieltere, thematisch passendere Paper."
                )

    # Endergebnis: beste Paper ueber alle Runden, sortiert
    beste = sorted(best_by_id.values(), key=lambda r: r.get("score", 0), reverse=True)
    print(f"\n{'='*60}\n BESTE ERGEBNISSE (ueber alle Runden)\n{'='*60}")
    for r in beste:
        print(f"[{r.get('score')}/10] {r.get('titel')}")
        print(f"        {r.get('begruendung')}\n")
    return beste
