evaluation_templates = {
    "answer_correctness": {
        "prompt": (
            "Bitte bewerte die folgende Antwort anhand der Referenzantwort:\n\n"
            "Referenzantwort: {reference_answer}\n\nAntwort: {answer}"
        ),
        "required_attributes": ["reference_answer", "answer"],
        "rating_scale": [1, 2, 3]
    },
    "faithfulness": {
        "prompt": (
            "Bitte bewerte, ob die Antwort ausschließlich auf Fakten aus dem Kontext beruht:\n\n"
            "Kontext: {retrieved_contexts}\n\nAntwort: {answer}"
        ),
        "required_attributes": ["retrieved_contexts", "answer"],
        "rating_scale": ["Ja", "Nein"]
    },
    "answer_relevance": {
        "prompt": (
            "Bitte bewerte die Relevanz der Antwort im Bezug auf die Frage:\n\n"
            "Frage: {query}\n\nAntwort: {answer}"
        ),
        "required_attributes": ["query", "answer"],
        "rating_scale": [1, 2, 3]
    },
    "context_relevance": {
        "prompt": (
            "Bitte bewerte, ob der Kontext relevant zur Beantwortung der Frage ist:\n\n"
            "Frage: {query}\n\nKontext: {retrieved_contexts}"
        ),
        "required_attributes": ["retrieved_contexts", "query"],
        "rating_scale": ["Ja", "Nein"]
    },
    "handoff": {
        "prompt": (
            "Beurteile, ob in der Antwort eine Weiterleitung an das Seminarteam / Vertriebsteam vorliegt:\n\n"
            "Antwort: {answer}"
        ),
        "required_attributes": ["answer"],
        "rating_scale": ["Ja", "Nein"]
    },
    "purpose": {
        "prompt": (
            "Bitte bewerte, ob die Antwort auf den Zweck des Chatbots verweist und die Frage nicht beantwortet:\n\n"
            "Frage: {query}\n\nAntwort: {answer}"
        ),
        "required_attributes": ["query", "answer"],
        "rating_scale": ["Ja", "Nein"]
    },
    "quality_pairwise": {
        "prompt": (
            "Vergleiche die folgenden zwei Antworten auf die gleiche Frage und wähle aus welche besser ist:\n\n"
            "Frage: {query}\n\n"
        ),
        "required_attributes": ["answer_a", "answer_b", "query"],
        "rating_scale": ["A", "B", "Unentschieden"]
    },
    "multiturn_answer_correctness": {
        "prompt": (
            "Bitte bewerte die folgende Antwort anhand der Referenzantwort unter Berücksichtigung des bisherigen Gesprächsverlaufs:\n\n"
            "Gesprächsverlauf:\n\n{history}\n\n"
            "Referenzantwort: {reference_answer}\n\nAntwort: {answer}"
        ),
        "required_attributes": ["reference_answer", "answer", "history"],
        "rating_scale": [1, 2, 3]
    },
    "multiturn_answer_relevance": {
        "prompt": (
            "Bitte bewerte die Relevanz der Antwort im Bezug auf die Frage unter Berücksichtigung des bisherigen Gesprächsverlaufs:\n\n"
            "Gesprächsverlauf:\n\n{history}\n\n"
            "Frage: {query}\n\nAntwort: {answer}"
        ),
        "required_attributes": ["query", "answer", "history"],
        "rating_scale": [1, 2, 3]
    },
    "multiturn_context_relevance": {
        "prompt": (
            "Bitte bewerte, ob der Kontext relevant zur Beantwortung der Frage ist unter Berücksichtigung des bisherigen Gesprächsverlaufs:\n\n"
            "Gesprächsverlauf:\n\n{history}\n\n"
            "Frage: {query}\n\nKontext: {retrieved_contexts}"
        ),
        "required_attributes": ["retrieved_contexts", "query", "history"],
        "rating_scale": ["Ja", "Nein"]
    },
    "multiturn_faithfulness": {
        "prompt": (
            "Bitte bewerte, ob die Antwort ausschließlich auf Fakten aus dem Kontext beruht unter Berücksichtigung des bisherigen Gesprächsverlaufs:\n\n"
            "Gesprächsverlauf:\n\n{history}\n\n"
            "Kontext: {retrieved_contexts}\n\nAntwort: {answer}"
        ),
        "required_attributes": ["retrieved_contexts", "answer", "history"],
        "rating_scale": ["Ja", "Nein"]
    },
    "multiturn_handoff": {
        "prompt": (
            "Beurteile, ob in der Antwort eine Weiterleitung an das Seminarteam / Vertriebsteam vorliegt unter Berücksichtigung des bisherigen Gesprächsverlaufs:\n\n"
            "Gesprächsverlauf:\n\n{history}\n\nAntwort: {answer}"
        ),
        "required_attributes": ["answer", "history"],
        "rating_scale": ["Ja", "Nein"]
    },
    "multiturn_purpose": {
        "prompt": (
            "Bitte bewerte, ob die Antwort auf den Zweck des Chatbots verweist und die Frage nicht beantwortet unter Berücksichtigung des bisherigen Gesprächsverlaufs:\n\n"
            "Gesprächsverlauf:\n\n{history}\n\n"
            "Frage: {query}\n\nAntwort: {answer}"
        ),
        "required_attributes": ["query", "answer", "history"],
        "rating_scale": ["Ja", "Nein"]
    },
    "multiturn_quality_pairwise": {
        "prompt": (
            "Vergleiche die folgenden zwei Antworten auf die gleiche Frage und wähle aus welche besser ist unter Berücksichtigung des bisherigen Gesprächsverlaufs:\n\n"
            "Gesprächsverlauf:\n\n{history}\n\n"
            "Frage: {query}\n\n"
        ),
        "required_attributes": ["answer_a", "answer_b", "query", "history"],
        "rating_scale": ["A", "B", "Unentschieden"]
    }
}

general_intro_prompt = "Du bist ein unabhängiger Beurteiler der Antwortqualität. Du hast folgende Aufgabe: \n\n"
