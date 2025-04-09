unwanted_words = [
    'news',
    'terms-and-conditions',
    'terms-of-use',
    'imprint',
    'blog',
    'privacy',
    'disclosure',
    'legal',
    'shop',
    'store',
    'career',
    'jobs',

    'neuigkeiten',
    'impressum',
    'datenschutz',
    'datenschutzbestimmungen',
    'karriere',

    'nouvelles',               # news
    'conditions-generales',    # terms and conditions
    'mentions-legales',        # legal notice
    'blog',
    'confidentialite',         # privacy
    'politique-de-confidentialite',
    'divulgation',             # disclosure
    'boutique',                # shop
    'magasin',                 # store
    'carriere',

    'notizie',                 # news
    'termini-e-condizioni',
    'termini-di-utilizzo',
    'informazioni-legali',     # legal notice / imprint
    'blog',
    'privacy',
    'politica-sulla-privacy',
    'divulgazione',
    'negozio',
    'store',
    'carriera',
]

product_keywords = [
    "product", "service", "solution", "offerings", "platform", "features", "tools",
    "application", "technology", "catalog", "portfolio", "what-we-offer", "what-we-do",

    "produkte", "leistung", "loesung", "angebot", "plattform", "funktionen",
    "anwendung", "technologie", "katalog", "portfolio", "was-wir-anbieten", "was-wir-tun",

    "produit", "service", "solution", "offre", "plateforme", "fonctionnalites", "outils",
    "application", "technologie", "catalogue", "portfolio", "ce-que-nous-offrons", "ce-que-nous-faisons",

    "prodotto", "servizio", "soluzione", "offerta", "piattaforma", "funzionalita", "strumenti",
    "applicazione", "tecnologia", "catalogo", "portfolio", "cosa-offriamo", "cosa-facciamo"
]

team_keywords = [
    "about", "team", "founder", "people", "staff", "who-we-are", "company", "history",

    "ueber-uns", "uber-uns", "gruender", "menschen", "mitarbeiter", "wer-wir-sind", "unternehmen", "geschichte",

    "a-propos", "equipe", "fondateur", "personnes", "personnel", "qui-nous-sommes", "entreprise", "histoire",

    "chi-siamo", "fondator", "persone", "staff", "azienda", "storia"
]

contact_keywords = [
    "contact", "kontakt", "contactez", "contatto"
]

value_keywords = [
    "values", "goal", "mission", "vision", "strategy", "purpose", "what-we-believe", "culture",
    "principles", "commitment", "beliefs",

    "werte", "ziel", "zweck", "glaubenssaetze", "strategie", "kultur",
    "prinzipien", "engagement", "ueberzeugungen",

    "valeurs", "but", "ce-que-nous-croyons", "culture",
    "principes", "engagement", "convictions",

    "valori", "obiettivo", "missione", "visione", "strategia", "scopo", "cosa-crediamo", "cultura",
    "principi", "impegno", "credenze"
]

WANTED_KEYWORDS = list({k for k in product_keywords + team_keywords + contact_keywords + value_keywords})
UNWANTED_KEYWORDS = list({k for k in unwanted_words})
