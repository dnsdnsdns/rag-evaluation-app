evaluation_templates = {
    "faithfulness": {
      "prompt": "\n\n**Aufgabe:**\nBewerte die Antwort eines Chatbots hinsichtlich ihrer Faktentreue.\n\nDu erhältst den **Kontext**, der dem Chatbot bei der Generierung der Antwort zur Verfügung stand, sowie die vom Chatbot **erzeugte Antwort**.\n\nBitte lies alle Texte **sorgfältig und vollständig**, bevor du deine Bewertung abgibst.\n\n---\n\n**Definition:**\nEine faktentreue Antwort enthält **ausschließlich Informationen**, die im angegebenen Kontext belegt sind. Jegliche Hinzufügung von Fakten, die **nicht im Kontext enthalten sind**, gilt als *Halluzination*.\n\n**Wichtig:** Es geht **nicht** um die allgemeine Richtigkeit der Informationen, sondern **ausschließlich** darum, ob die Inhalte der Antwort im verfügbaren Kontext **belegt** sind.\n\n---\n\n**Bewertungsskala:**\n- **Ja** – Die Antwort basiert ausschließlich auf Informationen aus dem Kontext.\n- **Nein** – Die Antwort enthält Informationen, die nicht im Kontext belegt sind.",
      "required_attributes": ["retrieved_contexts", "answer", "retrieved_contexts_full"],
      "rating_scale": ["Ja", "Nein"],
      "final_question": "Stützt sich die Antwort ausschließlich auf Informationen aus dem verfügbaren Kontext?"
    },
    "multiturn_faithfulness": {
      "prompt": "\n\n\n\n**Aufgabe:**\nBewerte die Antwort eines Chatbots hinsichtlich ihrer Faktentreue.\n\nDu erhältst den bisherigen **Gesprächsverlauf** und den **Kontext**, der dem Chatbot bei der Generierung der Antwort zur Verfügung standen, sowie die vom Chatbot **erzeugte Antwort**.\n\nBitte lies alle Texte **sorgfältig und vollständig**, bevor du deine Bewertung abgibst.\n\n---\n\n**Definition:**\nEine faktentreue Antwort enthält **ausschließlich Informationen**, die im angegebenen Kontext oder den bisherigen Antworten belegt sind. Jegliche Hinzufügung von Fakten, die **dort nicht enthalten sind**, gilt als *Halluzination*.\n\n**Wichtig:** Es geht **nicht** um die allgemeine Richtigkeit der Informationen, sondern **ausschließlich** darum, ob die Inhalte der Antwort im verfügbaren Kontext oder den bisherigen Antworten **belegt** sind.\n\n---\n\n**Bewertungsskala:**\n- **Ja** – Die Antwort basiert ausschließlich auf Informationen aus dem Kontext oder den bisherigen Antworten.\n- **Nein** – Die Antwort enthält Informationen, die nicht im Kontext oder den bisherigen Antworten belegt sind.",
      "required_attributes": ["retrieved_contexts", "answer", "history", "retrieved_contexts_full"],
      "rating_scale": ["Ja", "Nein"],
      "final_question": "Stützt sich die Antwort ausschließlich auf Informationen aus dem verfügbaren Kontext oder bisherigen Antworten?"
    },
    "answer_correctness": {
      "prompt": "\n\n**Aufgabe:**\nBewerte die inhaltliche Korrektheit der Antwort im Vergleich zur Referenzantwort.\n\nDu erhältst eine **Antwort** des Chatbots sowie eine **Referenzantwort**, die als korrekt gilt.\n\nBitte lies alle Texte **sorgfältig und vollständig**, bevor du deine Bewertung abgibst.\n\n---\n\n**Definition:**\nEine korrekte Antwort stimmt inhaltlich mit der Referenzantwort überein. Ergänzende Informationen sind erlaubt, wenn sie den Inhalten der Referenz nicht widersprechen. Es darf allerdings keine inhaltlich relevante Aussage aus der Referenzantwort fehlen.\n\n**Wichtig:** Unterschiede in der Formulierung sind erlaubt, solange die Bedeutung erhalten bleibt.\n\n---\n\n**Bewertungsskala:**\n- **1** – Die Antwort ist inhaltlich falsch, enthält widersprüchliche Informationen oder weicht deutlich von der Referenzantwort ab.\n- **2** – Die Antwort ist überwiegend korrekt, deckt aber nicht alle relevanten Aussagen der Referenz ab oder enthält sachliche Ungenauigkeiten.\n- **3** – Die Antwort stimmt inhaltlich vollständig mit der Referenz überein.",
      "required_attributes": ["reference_answer", "answer"],
      "rating_scale": [1, 2, 3],
      "final_question": "Wie gut stimmt die Antwort inhaltlich mit der Referenzantwort überein?"
    },
    "multiturn_answer_correctness": {
      "prompt": "\n\n**Aufgabe:**\nBewerte die inhaltliche Korrektheit der Antwort im Vergleich zur Referenzantwort.\n\nDu erhältst den bisherigen **Gesprächsverlauf**, eine **Antwort** des Chatbots sowie eine **Referenzantwort**, die als korrekt gilt.\n\nBitte lies alle Texte **sorgfältig und vollständig**, bevor du deine Bewertung abgibst.\n\n---\n\n**Definition:**\nEine korrekte Antwort stimmt inhaltlich mit der Referenzantwort überein. Ergänzende Informationen sind erlaubt, wenn sie den Inhalten der Referenz nicht widersprechen. Es darf allerdings keine inhaltlich relevante Aussage aus der Referenzantwort fehlen.\n\n**Wichtig:** Unterschiede in der Formulierung sind erlaubt, solange die Bedeutung erhalten bleibt.\n\n---\n\n**Bewertungsskala:**\n- **1** – Die Antwort ist inhaltlich falsch, enthält widersprüchliche Informationen oder weicht deutlich von der Referenzantwort ab.\n- **2** – Die Antwort ist überwiegend korrekt, deckt aber nicht alle relevanten Aussagen der Referenz ab oder enthält sachliche Ungenauigkeiten.\n- **3** – Die Antwort stimmt inhaltlich vollständig mit der Referenz überein.",
      "required_attributes": ["reference_answer", "answer", "history"],
      "rating_scale": [1, 2, 3],
      "final_question": "Wie gut stimmt die Antwort inhaltlich mit der Referenzantwort überein?"
    },
    "answer_relevance": {
      "prompt": "\n\n**Aufgabe:**\nBewerte die Relevanz der Antwort im Hinblick auf die gestellte Frage.\n\nDu erhältst eine **Frage** und die dazugehörige **Antwort** des Chatbots.\n\nBitte lies alle Texte **sorgfältig und vollständig**, bevor du deine Bewertung abgibst.\n\n---\n\n**Definition:**\nEine relevante Antwort geht inhaltlich auf die gestellte Frage ein. Sie enthält Informationen, die direkt auf die Frage eingehen oder zur Beantwortung beitragen.\nEine Antwort ist weniger relevant, wenn sie allgemeine Aussagen macht, vom Thema abweicht oder an der konkreten Fragestellung vorbeigeht.\n\n---\n\n**Bewertungsskala:**\n- **1** – Die Antwort verfehlt die Frage.\n- **2** – Die Antwort geht teilweise auf die Frage ein, bleibt aber ungenau oder weicht teils ab.\n- **3** – Die Antwort trifft die Frage direkt.",
      "required_attributes": ["query", "answer"],
      "rating_scale": [1, 2, 3],
      "final_question": "Wie relevant ist die Antwort im Hinblick auf die gestellte Frage?"
    },
    "multiturn_answer_relevance": {
      "prompt": "\n\n**Aufgabe:**\nBewerte die Relevanz der Antwort im Hinblick auf die gestellte Frage.\n\nDu erhältst den bisherigen **Gesprächsverlauf**, die aktuelle **Frage** und die dazugehörige **Antwort** des Chatbots.\n\nBitte lies alle Texte **sorgfältig und vollständig**, bevor du deine Bewertung abgibst.\n\n---\n\n**Definition:**\nEine relevante Antwort geht inhaltlich auf die gestellte Frage ein und berücksichtigt dabei den bisherigen Gesprächsverlauf. Sie enthält Informationen, die direkt auf die Frage eingehen oder zur Beantwortung beitragen.\nEine Antwort ist weniger relevant, wenn sie allgemeine Aussagen macht, vom Thema abweicht oder an der konkreten Fragestellung vorbeigeht.\n\n---\n\n**Bewertungsskala:**\n- **1** – Die Antwort verfehlt die Frage im Kontext des bisherigen Gesprächsverlaufs.\n- **2** – Die Antwort geht teilweise auf die Frage im Kontext des bisherigen Gesprächsverlaufs ein, bleibt aber ungenau oder weicht teils ab.\n- **3** – Die Antwort trifft die Frage im Kontext des bisherigen Gesprächsverlaufs direkt.",
      "required_attributes": ["query", "answer", "history"],
      "rating_scale": [1, 2, 3],
      "final_question": "Wie relevant ist die Antwort im Hinblick auf die gestellte Frage (im Kontext des bisherigen Gesprächsverlaufs)?"
    },
    "context_relevance": {
      "prompt": "\n\n**Aufgabe:**\nBewerte die Relevanz des Kontexts für die Beantwortung der Frage.\n\nDu erhältst eine **Frage** und **Kontextelemente**, die dem Chatbot bei der Generierung der Antwort zur Verfügung gestellt werden und potenziell hilfreiche Informationen enthalten können.\n\nBitte lies alle Texte **sorgfältig und vollständig**, bevor du deine Bewertung abgibst.\n\n---\n\n**Definition:**\nEin Kontextelement ist relevant, wenn es Informationen enthält, die direkt zur Beantwortung der gestellten Frage beitragen. Es ist nicht relevant, wenn es keine Informationen enthält, die zur Beantwortung der Frage beitragen.\n\n*Wichtig:** Der Kontext muss nicht alle Aspekte der Frage abdecken, sollte aber erkennbar zur Lösung beitragen.\n\n---\n\n**Bewertungsskala:**\n- **Ja** – Der Kontext ist relevant für die Beantwortung der Frage.\n- **Nein** – Der Kontext ist nicht relevant für die Beantwortung der Frage.",
      "required_attributes": ["retrieved_contexts", "query", "retrieved_contexts_full", "context"],
      "rating_scale": ["Ja", "Nein"],
      "final_question": "Ist dieser Kontext relevant zur Beantwortung der Frage?"
    },
    "multiturn_context_relevance": {
      "prompt": "\n\n**Aufgabe:**\nBewerte die Relevanz des Kontexts für die Beantwortung der Frage.\n\nDu erhältst den bisherigen **Gesprächsverlauf**, die aktuelle **Frage** und **Kontextelemente**, die dem Chatbot bei der Generierung der Antwort zur Verfügung gestellt werden und potenziell hilfreiche Informationen enthalten können.\n\nBitte lies alle Texte **sorgfältig und vollständig**, bevor du deine Bewertung abgibst.\n\n---\n\n**Definition:**\nEin Kontextelement ist relevant, wenn es Informationen enthält, die direkt zur Beantwortung der gestellten Frage im Kontext des Gesprächsverlaufs beitragen. Es ist nicht relevant, wenn es keine Informationen enthält, die zur Beantwortung der Frage beitragen.\n\n*Wichtig:** Der Kontext muss nicht alle Aspekte der Frage abdecken, sollte aber erkennbar zur Lösung beitragen.\n\n---\n\n**Bewertungsskala:**\n- **Ja** – Der Kontext ist relevant für die Beantwortung der Frage im Kontext des Gesprächsverlaufs.\n- **Nein** – Der Kontext ist nicht relevant für die Beantwortung der Frage im Kontext des Gesprächsverlaufs.",
      "required_attributes": ["retrieved_contexts", "query", "retrieved_contexts_full", "context", "history"],
      "rating_scale": ["Ja", "Nein"],
      "final_question": "Ist dieser Kontext relevant zur Beantwortung der Frage (im Kontext des Gesprächsverlaufs)?"
    },
    "handoff": {
      "prompt": "\n\n**Aufgabe:**\nBeurteile, ob die Antwort des Chatbots eine Weiterleitung an das Seminarteam oder das Vertriebsteam enthält.\n\nBitte lies alle Texte **sorgfältig und vollständig**, bevor du deine Bewertung abgibst.\n\n---\n\n**Definition:**\nEine Weiterleitung liegt vor, wenn der Chatbot in seiner Antwort ausdrücklich auf das Seminarteam oder Vertriebsteam verweist - etwa durch Nennung einer E-Mail-Adresse, einer Telefonnummer oder durch Formulierungen wie \"bitte wenden Sie sich an ...\".\n\n---\n\n**Bewertungsskala:**\n- **Ja** – Die Antwort verweist auf das Seminarteam oder das Vertriebsteam.\n- **Nein** – Die Antwort enthält keinen Verweis auf das Seminarteam oder Vertriebsteam.",
      "required_attributes": ["answer"],
      "rating_scale": ["Ja", "Nein"],
      "final_question": "Enthält die Antwort eine Weiterleitung an das Seminarteam oder Vertriebsteam?"
    },
    "multiturn_handoff": {
      "prompt": "\n\n**Aufgabe:**\nBeurteile, ob die Antwort des Chatbots eine Weiterleitung an das Seminarteam oder das Vertriebsteam enthält.\n\nBitte lies alle Texte **sorgfältig und vollständig**, bevor du deine Bewertung abgibst.\n\n---\n\n**Definition:**\nEine Weiterleitung liegt vor, wenn der Chatbot in seiner Antwort ausdrücklich auf das Seminarteam oder Vertriebsteam verweist - etwa durch Nennung einer E-Mail-Adresse, einer Telefonnummer oder durch Formulierungen wie \"bitte wenden Sie sich an ...\".\n\n---\n\n**Bewertungsskala:**\n- **Ja** – Die Antwort verweist auf das Seminarteam oder das Vertriebsteam.\n- **Nein** – Die Antwort enthält keinen Verweis auf das Seminarteam oder Vertriebsteam.",
      "required_attributes": ["answer"],
      "rating_scale": ["Ja", "Nein"],
      "final_question": "Enthält die Antwort eine Weiterleitung an das Seminarteam oder Vertriebsteam?"
    },
    "purpose": {
      "prompt": "\n\n**Aufgabe:**\nBeurteile, ob die Antwort den Zweck oder die Funktion des Chatbots beschreibt und keine inhaltliche Antwort auf die Frage gibt.\n\nDu erhältst die **Frage** sowie die dazugehörige **Antwort** des Chatbots.\n\nBitte lies alle Texte **sorgfältig und vollständig**, bevor du deine Bewertung abgibst.\n\n---\n\n**Definition:**\n\nIn manchen Fällen gibt der Chatbot keine inhaltliche Antwort, sondern beschreibt stattdessen, wofür er zuständig ist oder was er nicht leisten kann – also seinen Zweck oder Aufgabenbereich. Solche zweckbeschreibenden Antworten gelten in dieser Bewertungsaufgabe als zutreffend und werden mit \"Ja\" bewertet.\n\n**Wichtig:** Wenn die Antwort zusätzlich einen weiterführenden Hinweis gibt (z.B. auf andere Quellen), ist das in Ordnung, solange keine inhaltliche Beantwortung der eigentlichen Frage erfolgt.\nSobald die Antwort jedoch inhaltlich auf die Frage eingeht, soll mit \"Nein\" bewertet werden.\n\n---\n\n**Bewertungsskala:**\n- **Ja** – Die Antwort beschreibt den Zweck oder Aufgabenbereich des Chatbots und enthält keine inhaltliche Antwort auf die gestellte Frage.\n- **Nein** – Die Antwort enthält eine inhaltliche Antwort auf die gestellte Frage – auch wenn zusätzlich der Zweck oder eine Einschränkung erwähnt wird.",
      "required_attributes": ["query", "answer"],
      "rating_scale": ["Ja", "Nein"],
      "final_question": "Beschränkt sich die Antwort auf eine Beschreibung des Chatbot-Zwecks, ohne die Frage inhaltlich zu beantworten?"
    },
    "multiturn_purpose": {
      "prompt": "\n\n**Aufgabe:**\nBeurteile, ob die Antwort den Zweck oder die Funktion des Chatbots beschreibt und keine inhaltliche Antwort auf die Frage im Kontext des Gesprächsverlaufs gibt.\n\nDu erhältst den bisherigen **Gesprächsverlauf**, die aktuelle **Frage** sowie die dazugehörige **Antwort** des Chatbots.\n\nBitte lies alle Texte **sorgfältig und vollständig**, bevor du deine Bewertung abgibst.\n\n---\n\n**Definition:**\n\nIn manchen Fällen gibt der Chatbot keine inhaltliche Antwort, sondern beschreibt stattdessen, wofür er zuständig ist oder was er nicht leisten kann – also seinen Zweck oder Aufgabenbereich. Solche zweckbeschreibenden Antworten gelten in dieser Bewertungsaufgabe als zutreffend und werden mit \"Ja\" bewertet.\n\n**Wichtig:** Wenn die Antwort zusätzlich einen weiterführenden Hinweis gibt (z.B. auf andere Quellen), ist das in Ordnung, solange keine inhaltliche Beantwortung der eigentlichen Frage erfolgt.\nSobald die Antwort jedoch inhaltlich auf die Frage eingeht, soll mit \"Nein\" bewertet werden.\n\n---\n\n**Bewertungsskala:**\n- **Ja** – Die Antwort beschreibt den Zweck oder Aufgabenbereich des Chatbots und enthält keine inhaltliche Antwort auf die aktuelle Frage.\n- **Nein** – Die Antwort enthält eine inhaltliche Antwort auf die aktuelle Frage – auch wenn zusätzlich der Zweck oder eine Einschränkung erwähnt wird.",
      "required_attributes": ["query", "answer", "history"],
      "rating_scale": ["Ja", "Nein"],
      "final_question": "Beschränkt sich die Antwort auf eine Beschreibung des Chatbot-Zwecks, ohne die Frage im Kontext des Gesprächsverlaufs inhaltlich zu beantworten?"
    },
    "quality_pairwise": {
      "prompt": "\n\n**Aufgabe:**\nDu erhältst eine **Frage** und **zwei Antworten** von verschiedenen Chatbots. Wähle die qualitativ bessere Antwort aus.\n\nBitte lies alle Texte **sorgfältig und vollständig**, bevor du deine Bewertung abgibst.\n\n---\n\n**Bewertungsskala:**\n- **A** – Antwort A ist besser.\n- **B** – Antwort B ist besser.\n- **Unentschieden** - Beide Antworten sind gleichwertig.",
      "required_attributes": ["answer_a", "answer_b", "query"],
      "rating_scale": ["A", "B", "Unentschieden"],
      "final_question": "Welches ist die bessere Antwort?"
    },
    "multiturn_quality_pairwise": {
      "prompt": "\n\n**Aufgabe:**\nDu erhältst den bisherigen **Gesprächsverlauf**, die aktuelle **Frage** und **zwei Antworten** von verschiedenen Chatbots. Wähle die qualitativ bessere Antwort unter Berücksichtigung des Gesprächsverlaufs aus.\n\nBitte lies alle Texte **sorgfältig und vollständig**, bevor du deine Bewertung abgibst.\n\n---\n\n**Definition:**\n\nBewerte nach dem qualitativen Gesamteindruck.\n\n---\n\n**Bewertungsskala:**\n- **A** – Antwort A ist besser.\n- **B** – Antwort B ist besser.\n- **Unentschieden** - Beide Antworten sind gleichwertig.",
      "required_attributes": ["answer_a", "answer_b", "query", "history"],
      "rating_scale": ["A", "B", "Unentschieden"],
      "final_question": "Welches ist die bessere Antwort unter Berücksichtigung des Gesprächsverlaufs?"
    }
  }