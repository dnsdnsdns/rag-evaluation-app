evaluation_templates = {
    "quality": {
        "prompt": (
            "Bitte bewerte die folgende Antwort auf diese Frage:\n"
            "Eingabe: {user_input}\nAntwort: {response}"
        ),
        "required_attributes": ["user_input", "response"],
        "rating_scale": [1, 2, 3, 4, 5]
    },
    "faithfulness": {
        "prompt": (
            "Bitte bewerte, ob die Antwort ausschließlich auf Fakten aus dem Kontext beruht:\n"
            "Kontext: {retrieved_contexts}\nAntwort: {response}"
        ),
        "required_attributes": ["retrieved_contexts", "response"],
        "rating_scale": ["Ja", "Nein"]
    },
    "quality_pairwise": {
        "prompt": (
            "Vergleiche die folgenden zwei Antworten auf die gleiche Frage:\n"
            "A: {response_a}\nB: {response_b}"
        ),
        "required_attributes": ["response_a", "response_b"],
        "rating_scale": ["A", "B", "Unentschieden"]
    },
    "multi_turn_quality": {
        "prompt": "Bitte bewerte die folgende Konversation:\n{dialog}",
        "required_attributes": ["dialog"],
        "rating_scale": [1, 2, 3, 4, 5]
    },
    "multi_turn_quality_pairwise": {
        "prompt": (
            "Vergleiche die folgenden zwei Konversationen:\n\n"
            "A:\n{dialog_a}\n\n"
            "B:\n{dialog_b}"
        ),
        "required_attributes": ["dialog_a", "dialog_b"],
        "rating_scale": ["A", "B", "Unentschieden"]
    }
}

general_intro_prompt = "Du bist ein unabhängiger Beurteiler der Antwortqualität. Du hast folgende Aufgabe: \n\n"
