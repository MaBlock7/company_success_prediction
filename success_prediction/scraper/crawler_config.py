"""
Keywords were machine-translated for French and Italian
"""

unwanted_words = [
    "news", "neuigkeiten", "nouvelles", "notizie",
    "terms-and-conditions", "datenschutzbestimmungen", "conditions-generales", "termini-e-condizioni",
    "terms-of-use", "nutzungsbedingungen", "conditions-d'utilisation", "termini-di-utilizzo",
    "imprint", "impressum", "mentions-legales", "informazioni-legali",
    "blog", "blog", "blog", "blog",
    "privacy", "datenschutz", "confidentialite", "privacy",
    "disclosure", "offenlegung", "divulgation", "divulgazione",
    "legal", "rechtliches", "mentions-legales", "informazioni-legali",
    "shop", "laden", "boutique", "negozio",
    "store", "einkauf", "magasin", "store",
    "career", "karriere", "carriere", "carriera",
    "jobs", "offene stellen", "emplois", "lavori",

    # File extensions
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".csv", ".ppt", ".pptx",
    ".zip", ".rar", ".7z", ".tar", ".gz",

    # Admin and non-content paths
    "login", "register", "signup", "signin", "logout",
    "admin", "dashboard", "account", "user"
]

product_keywords = [
    "product", "produkte", "produit", "prodotto",
    "service", "leistung", "service", "servizio",
    "solution", "loesung", "solution", "soluzione",
    "offering", "angebot", "offre", "offerta",
    "platform", "plattform", "plateforme", "piattaforma",
    "features", "funktionen", "fonctionnalites", "funzionalita",
    "tool", "tool", "outil", "strumento", "strumenti",
    "application", "anwendung", "application", "applicazione",
    "technology", "technologie", "technologie", "tecnologia",
    "catalog", "katalog", "catalogue", "catalogo",
    "portfolio", "portfolio", "portfolio", "portfolio",
    "what-we-offer", "was-wir-anbieten", "ce-que-nous-offrons", "cosa-offriamo",
    "what-we-do", "was-wir-tun", "ce-que-nous-faisons", "cosa-facciamo",
]

team_keywords = [
    "about", "ueber-uns", "a-propos", "chi-siamo",
    "team", "mitarbeiter", "equipe", "staff",
    "founder", "gruender", "fondateur", "fondator",
    "leadership", "leitung", "dirigeants", "dirigenti",
    "board", "vorstand", "administration", "dirigenti",
    "people", "menschen", "personnes", "persone",
    "staff", "mitarbeiter", "personnel", "staff",
    "who-we-are", "wer-wir-sind", "qui-nous-sommes", "chi-siamo",
    "company", "unternehmen", "entreprise", "azienda",
    "history", "geschichte", "histoire", "storia",
    "management", "leitung", "administration", "dirigenti",
    "executives", "vorstand", "dirigeants", "dirigenti",
    "bio", "biographie", "a-propos", "chi-siamo",
]

contact_keywords = [
    "contact", "kontakt", "contactez", "contatto"
]

value_keywords = [
    "values", "werte", "valeurs", "valori",
    "goal", "ziel", "but", "obiettivo",
    "mission", "zweck", "ce-que-nous-croyons", "missione",
    "vision", "glaubenssaetze", "ce-que-nous-croyons", "visione",
    "strategy", "strategie", "principes", "strategia",
    "purpose", "zweck", "ce-que-nous-croyons", "scopo",
    "what-we-believe", "glaubenssaetze", "ce-que-nous-croyons", "cosa-crediamo",
    "culture", "kultur", "culture", "cultura",
    "principles", "prinzipien", "principes", "principi",
    "commitment", "engagement", "engagement", "impegno",
    "beliefs", "ueberzeugungen", "convictions", "credenze",
]

esg_keywords = [
    "sustainability", "nachhaltigkeit", "durabilite", "sostenibilita",
    "sustainable", "nachhaltig", "durable", "sostenibile",
    "esg", "esg", "esg", "esg",
    "environment", "umwelt", "environnement", "ambiente",
    "climate", "klima", "climat", "clima",
    "carbon", "co2", "carbone", "carbonio",
    "net-zero", "netto-null", "zerocarbone", "emissionizero",
    "decarbonization", "dekarbonisierung", "decarbonation", "decarbonizzazione",
    "csr", "verantwortung", "responsabilite", "responsabilita",
    "responsibility", "verantwortung", "responsabilite", "responsabilita",
    "responsible", "verantwortung", "responsabilite", "responsabilita",
    "social", "wirkung", "impact", "filantropia",
    "impact", "wirkung", "impact", "filantropia",
    "governance", "verantwortung", "gouvernance", "responsabilita",
    "diversity", "vielfalt", "diversite", "diversita",
    "equality", "gleichstellung", "egalite", "equita",
    "inclusion", "inklusion", "inclusion", "inclusione",
    "community", "gemeinschaft", "communaute", "comunita",
    "ethics", "ethik", "ethique", "etica",
    "human-rights", "menschenrechte", "droitshumains", "dirittiumani",
    "initiative", "initiative", "initiative", "iniziativa",
    "wellbeing", "wohlbefinden", "bienetre", "benessere",
    "volunteer", "freiwilligenarbeit", "engagement", "volontariato",
    "engagement", "stakeholder", "partiesprenantes", "coinvolgimento",
    "contributions", "zuwendungen", "engagement", "coinvolgimento",
    "donations", "spenden", "engagement", "coinvolgimento",
]

WANTED_KEYWORDS = list({k for k in product_keywords + team_keywords + contact_keywords + value_keywords + esg_keywords})
UNWANTED_KEYWORDS = list({k for k in unwanted_words})
