evaluation_templates = {
    "quality": {
        "prompt": (
            "Bitte bewerte die folgende Antwort auf diese Frage:\n"
            "Eingabe: {query}\nAntwort: {answer}"
        ),
        "required_attributes": ["query", "answer"],
        "rating_scale": [1, 2, 3, 4, 5]
    },
    "faithfulness": {
        "prompt": (
            "Bitte bewerte, ob die Antwort ausschließlich auf Fakten aus dem Kontext beruht:\n"
            "Kontext: {retrieved_contexts}\nAntwort: {answer}"
        ),
        "required_attributes": ["retrieved_contexts", "answer"],
        "rating_scale": ["Ja", "Nein"]
    },
    "quality_pairwise": {
        "prompt": (
            "Vergleiche die folgenden zwei Antworten auf die gleiche Frage:\n"
            "Frage: {query}\n\n"
            "A: {answer_a}\nB: {answer_b}"
        ),
        "required_attributes": ["answer_a", "answer_b", "query"],
        "rating_scale": ["A", "B", "Unentschieden"]
    },
    "multi_turn_quality": {
        "prompt": "Bitte bewerte die folgende Konversation:\n{history}",
        "required_attributes": ["history"],
        "rating_scale": [1, 2, 3, 4, 5]
    },
    "multi_turn_quality_pairwise": {
        "prompt": (
            "Vergleiche die folgenden zwei Konversationen:\n\n"
            "Bisheriger Verlauf:\n{history}\n\n"
            "Frage: {query}\n\n"
            "A:\n{answer_a}\n\n"
            "B:\n{answer_b}\n\n"
        ),
        "required_attributes": ["history", "query", "answer_a", "answer_b"],
        "rating_scale": ["A", "B", "Unentschieden"]
    }
}

general_intro_prompt = "Du bist ein unabhängiger Beurteiler der Antwortqualität. Du hast folgende Aufgabe: \n\n"
