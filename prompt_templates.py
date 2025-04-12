evaluation_templates = {
    "answer_correctness": {
        "prompt": (
            "Bitte bewerte die folgende Antwort anhand der Referenzantwort:<br><br>"
            "Referenzantwort: {reference_answer}<br><br>Antwort: {answer}"
        ),
        "required_attributes": ["reference_answer", "answer"],
        "rating_scale": [1, 2, 3]
    },
    "faithfulness": {
        "prompt": (
            "Bitte bewerte, ob die Antwort ausschließlich auf Fakten aus dem Kontext beruht:<br><br>"
            "Antwort: {answer}"
        ),
        "required_attributes": ["retrieved_contexts", "answer", "retrieved_contexts_full"],
        "rating_scale": ["Ja", "Nein"]
    },
    "answer_relevance": {
        "prompt": (
            "Bitte bewerte die Relevanz der Antwort im Bezug auf die Frage:<br><br>"
            "Frage: {query}<br><br>Antwort: {answer}"
        ),
        "required_attributes": ["query", "answer"],
        "rating_scale": [1, 2, 3]
    },
    "context_relevance": {
        "prompt": (
            "Bitte bewerte, ob der Kontext relevant zur Beantwortung der Frage ist:<br><br>"
            "Frage: {query}"
        ),
        "required_attributes": ["retrieved_contexts", "query", "retrieved_contexts_full", "context"],
        "rating_scale": ["Ja", "Nein"]
    },
    "handoff": {
        "prompt": (
            "Beurteile, ob in der Antwort eine Weiterleitung an das Seminarteam / Vertriebsteam vorliegt:<br><br>"
            "Antwort: {answer}"
        ),
        "required_attributes": ["answer"],
        "rating_scale": ["Ja", "Nein"]
    },
    "purpose": {
        "prompt": (
            "Bitte bewerte, ob die Antwort auf den Zweck des Chatbots verweist und die Frage nicht beantwortet:<br><br>"
            "Frage: {query}<br><br>Antwort: {answer}"
        ),
        "required_attributes": ["query", "answer"],
        "rating_scale": ["Ja", "Nein"]
    },
    "quality_pairwise": {
        "prompt": (
            "Vergleiche die folgenden zwei Antworten auf die gleiche Frage und wähle aus welche besser ist:<br><br>"
            "Frage: {query}<br><br>"
        ),
        "required_attributes": ["answer_a", "answer_b", "query"],
        "rating_scale": ["A", "B", "Unentschieden"]
    },
    "multiturn_answer_correctness": {
        "prompt": (
            "Bitte bewerte die folgende Antwort anhand der Referenzantwort unter Berücksichtigung des bisherigen Gesprächsverlaufs:<br><br>"
            "Gesprächsverlauf:<br><br>{history}<br><br>"
            "Referenzantwort: {reference_answer}<br><br>Antwort: {answer}"
        ),
        "required_attributes": ["reference_answer", "answer", "history"],
        "rating_scale": [1, 2, 3]
    },
    "multiturn_answer_relevance": {
        "prompt": (
            "Bitte bewerte die Relevanz der Antwort im Bezug auf die Frage unter Berücksichtigung des bisherigen Gesprächsverlaufs:<br><br>"
            "Gesprächsverlauf:<br><br>{history}<br><br>"
            "Frage: {query}<br><br>Antwort: {answer}"
        ),
        "required_attributes": ["query", "answer", "history"],
        "rating_scale": [1, 2, 3]
    },
    "multiturn_context_relevance": {
        "prompt": (
            "Bitte bewerte, ob der Kontext relevant zur Beantwortung der Frage ist unter Berücksichtigung des bisherigen Gesprächsverlaufs:<br><br>"
            "Gesprächsverlauf:<br><br>{history}<br><br>"
            "Frage: {query}"
        ),
        "required_attributes": ["retrieved_contexts", "query", "history", "retrieved_contexts_full", "context"],
        "rating_scale": ["Ja", "Nein"]
    },
    "multiturn_faithfulness": {
        "prompt": (
            "Bitte bewerte, ob die Antwort ausschließlich auf Fakten aus dem Kontext beruht unter Berücksichtigung des bisherigen Gesprächsverlaufs:<br><br>"
            "Gesprächsverlauf:<br><br>{history}<br><br>"
            "Antwort: {answer}"
        ),
        "required_attributes": ["retrieved_contexts", "answer", "history", "retrieved_contexts_full"],
        "rating_scale": ["Ja", "Nein"]
    },
    "multiturn_handoff": {
        "prompt": (
            "Beurteile, ob in der Antwort eine Weiterleitung an das Seminarteam / Vertriebsteam vorliegt unter Berücksichtigung des bisherigen Gesprächsverlaufs:<br><br>"
            "Gesprächsverlauf:<br><br>{history}<br><br>Antwort: {answer}"
        ),
        "required_attributes": ["answer", "history"],
        "rating_scale": ["Ja", "Nein"]
    },
    "multiturn_purpose": {
        "prompt": (
            "Bitte bewerte, ob die Antwort auf den Zweck des Chatbots verweist und die Frage nicht beantwortet unter Berücksichtigung des bisherigen Gesprächsverlaufs:<br><br>"
            "Gesprächsverlauf:<br><br>{history}<br><br>"
            "Frage: {query}<br><br>Antwort: {answer}"
        ),
        "required_attributes": ["query", "answer", "history"],
        "rating_scale": ["Ja", "Nein"]
    },
    "multiturn_quality_pairwise": {
        "prompt": (
            "Vergleiche die folgenden zwei Antworten auf die gleiche Frage und wähle aus welche besser ist unter Berücksichtigung des bisherigen Gesprächsverlaufs:<br><br>"
            "Gesprächsverlauf:<br><br>{history}<br><br>"
            "Frage: {query}<br><br>"
        ),
        "required_attributes": ["answer_a", "answer_b", "query", "history"],
        "rating_scale": ["A", "B", "Unentschieden"]
    }
}

# Also update the general intro prompt
general_intro_prompt = "Du bist ein unabhängiger Beurteiler der Antwortqualität. Du hast folgende Aufgabe:<br><br>"

