import re
import unicodedata
import unicodedata
import numpy as np
import pandas as pd
from pathlib import Path

NUMBER = re.compile(r"\\d+(?:\\.\\d+)?\\s*k?", re.I)
def normalize_price(raw):
    text = unicodedata.normalize("NFKC", raw).casefold()
    text = re.sub(r"[–—−]", "-", text)
    # Remove valid thousands separators without merging arbitrary adjacent amounts.
    text = re.sub(r"(?<!\d)\d{1,3}(?:,\d{3})+(?!\d)", lambda m: m.group().replace(",", ""), text)
    text = re.sub(r"(?<!\d)\d{1,3}(?:[ \u00a0]\d{3})+(?!\d)", lambda m: re.sub(r"\s", "", m.group()), text)
    return text

def parse_price(raw):
    text = normalize_price(raw)
    has_digits = bool(re.search(r"\d", text))
    result = {"has_digits": has_digits, "included": False, "amount": float("nan"),
              "currency": "CAD (assumed)", "method": "", "reason": "", "notes": "",
              "range_lower": float("nan"), "range_upper": float("nan")}
    if not has_digits:
        result["reason"] = "Blank answer" if not text.strip() else "No numeric digits; no amount inferred"
        return result

    # A numeric rating is not a monetary answer, even if it contains a range.
    if re.search(r"\bscale\b|\bout of\s*10\b|\b[1-9]\s*/\s*10\b", text):
        result["reason"] = "Rating scale rather than a ticket amount"
        return result
    if re.search(r"€|\beur\b|\beuros?\b|£|\bgbp\b|\busd\b|us\$|\bus dollars?\b|\brmb\b|\bcny\b|¥", text):
        result["currency"] = "Explicit non-CAD currency"
        result["reason"] = "Foreign currency; excluded from the CAD summary without conversion"
        return result
    if re.search(r"\bcad\b|cad(?=\s*\d)|(?<=\d)cad\b|\bcanadian\b", text):
        result["currency"] = "CAD (explicit)"

    matches = list(NUMBER.finditer(text))
    values = [float(m.group().strip().rstrip("k").strip()) * (1000 if m.group().strip().endswith("k") else 1)
              for m in matches]
    notes = []
    if any(m.group().strip().endswith("k") for m in matches):
        notes.append("k expanded to thousands")
    if matches and re.search(r"-\s*\$?\s*$", text[:matches[0].start()]):
        result["reason"] = "Negative amount"
        return result

    if len(values) == 1:
        result["amount"] = values[0]
        result["method"] = "Single amount"
    elif len(values) == 2:
        separator = text[matches[0].end():matches[1].start()]
        separator = re.sub(r"\bcad\b|\bcanadian\b|\bdollars?\b|\bbucks?\b|\$", "", separator).strip()
        if separator not in {"-", "to"}:
            result["reason"] = "Multiple amounts without one unambiguous range"
            return result
        result["range_lower"], result["range_upper"] = sorted(values)
        result["amount"] = sum(values) / 2
        result["method"] = "Range midpoint"
        if values[0] > values[1]:
            notes.append("Reversed range endpoints; midpoint retained")
    else:
        result["reason"] = "Multiple amounts/scenarios; no single price chosen"
        return result

    if re.search(r"\bat most\b|\bmaximum\b|\bmax\b|\bup to\b|\bno more than\b|<=|=<|<", text):
        notes.append("Upper limit wording; stated number retained")
    if re.search(r"\d\s*\+|\bat least\b|\bminimum\b|>=|=>|>", text):
        notes.append("Lower limit wording; stated number retained")
    if re.search(r"\baround\b|\babout\b|\bperhaps\b|\bmaybe\b", text):
        notes.append("Approximate or conditional amount")
    result["included"] = True
    result["notes"] = "; ".join(notes)
    return result


EASE = ("How easy do you think it would be for someone with no prior experience "
        "to start playing this sport recreationally? Rate it on a scale from 0 to 10, "
        "where 0 means extremely difficult and 10 means extremely easy.")
LIKERT = [
    "Success in this sport depends heavily on strategy and planning.",
    "Players must be comfortable with a high risk of injury.",
    "Individual brilliance can outweigh teamwork in this sport.",
]
PRICE = "How much would you be willing to pay for a decent seat at a professional competition of this sport?"
VENUE = "Where might you play this sport?"
SEASON = "During which seasons would you be most likely to play this sport recreationally in Toronto?"
MUSIC = "What types of music do you most associate with this sport?"
OPTIONS = {
    "venue": (VENUE, ["Indoor facility", "Outdoor public space", "School or university facility",
                     "Community recreation facility", "Private sports facility",
                     "At home or nearby", "Others"]),
    "season": (SEASON, ["Spring", "Summer", "Fall", "Winter"]),
    "music": (MUSIC, ["Pop", "Rock", "Hip-hop or rap", "Electronic or dance",
                     "Country", "Latin", "Classical or jazz", "Folk or traditional",
                     "Other", "No particular type"]),
}
LIKERT_MAP = {"strongly disagree": 1, "disagree": 2, "neutral": 3,
              "agree": 4, "strongly agree": 5}
NUMERIC = ["ease", "strategy", "injury", "individual", "log_price"]


def _text(series):
    return series.fillna("").astype(str).str.strip()


def extract_numeric(df):
    required = [EASE, *LIKERT, PRICE]
    if any(c not in df.columns for c in required):
        raise ValueError("Missing numeric survey columns")
    out = pd.DataFrame(index=df.index)
    ease = pd.to_numeric(_text(df[EASE]), errors="coerce")
    out["ease"] = ease.where(ease.between(0, 10))
    for name, col in zip(NUMERIC[1:4], LIKERT):
        out[name] = _text(df[col]).str.casefold().map(LIKERT_MAP)
    parsed = _text(df[PRICE]).map(parse_price)
    price = pd.Series([p["amount"] if p["included"] else np.nan for p in parsed],
                      index=df.index, dtype=float)
    out["log_price"] = np.log1p(price)
    return out.astype(float)


def extract_choices(df):
    out = pd.DataFrame(index=df.index)
    for prefix, (column, options) in OPTIONS.items():
        if column not in df.columns:
            raise ValueError(f"Missing survey column: {column}")
        lookup = {opt.casefold(): opt for opt in options}
        values = []
        missing = []
        for raw in _text(df[column]):
            tokens = [t.strip().casefold() for t in raw.split(",") if t.strip()]
            # As in venue_analysis.py, reject a whole answer if unknown choices appear.
            invalid = bool(tokens) and any(t not in lookup for t in tokens)
            selected = set(tokens) if not invalid else set()
            values.append(selected)
            missing.append(float(not tokens or invalid))
        for option in options:
            out[f"{prefix}__{option}"] = [float(option.casefold() in s) for s in values]
        out[f"{prefix}__missing"] = missing
    return out.astype(float)




COLOR_COLOR_RULES = [('Red', 'red|reds|reddish|maroon|burgundy', '#c74440', 'Includes dark red, maroon and burgundy.'), ('Orange', 'orange|oranges|orangish|orangeish', '#e58c32', 'Includes light/dark orange through the base color word.'), ('Yellow', 'yellow|yellows|yellowish', '#e1c843', 'Includes yellowish; gold remains separate.'), ('Green', 'green|greens|greenish|绿色|绿', '#478c61', 'Includes observed Chinese green wording and light/dark green.'), ('Blue', 'blue|blues|bluish|blueish|navy|navy blue|navy blues', '#447faf', 'Includes navy and navy blues.'), ('Purple', 'purple|purples|purplish|violet|lavender', '#8c68aa', 'Includes common purple shades.'), ('Pink', 'pink|pinks|pinkish', '#d790ae', 'Common palette entry; may have zero observed answers.'), ('Brown', 'brown|browns|brownish', '#96724e', 'Includes light/dark brown; wood alone is not guessed to be brown.'), ('Black', 'black|blacks|blackish', '#333333', 'Specific color word.'), ('White', 'white|whites|whitish', '#fafafa', 'Cream remains separate.'), ('Gray', 'gray|grey|grays|greys|grayish|greyish', '#999999', 'American and British spellings are merged.'), ('Gold', 'gold|golden', '#bc9b31', 'Gold and golden are merged; not grouped into yellow.'), ('Silver', 'silver|silvery', '#c0c7ce', 'Common palette entry; may have zero observed answers.'), ('Beige', 'beige', '#cfba98', 'Kept separate from brown and cream.'), ('Cream', 'cream', '#f0e4bd', 'Yellowish-cream counts as Yellow and Cream.'), ('Tan', 'tan', '#bc946b', 'Kept separate from brown and beige.')]

def COLOR_normalize(text):
    text = unicodedata.normalize('NFKC', text).casefold()
    return ' '.join(re.sub('[^\\w\\s]', ' ', text).split())

COLOR_EXCLUSION_RULES = [('\\ball\\b', 'Contains all'), ('\\bany\\b', 'Contains any'), ('\\bunknown\\b|\\bunsure\\b', 'Unknown or unsure'), ('\\bnone\\b|\\bnothing\\b', 'None or nothing'), ('\\brainbow\\b|\\bvarious\\b|\\bevery\\b|\\beverything\\b', 'Unspecified collection of colors'), ('\\bno idea\\b|\\bno particular\\b|\\bno specific\\b|\\bno colou?rs?\\b', 'No specific color association'), ('\\bdon t know\\b|\\bdo not know\\b|\\bnot sure\\b', 'Does not know'), ('\\bn a\\b|^na$|^name$', 'Missing/error expression')]

def COLOR_make_matcher():
    lookup = {}
    for name, aliases, _, _ in COLOR_COLOR_RULES:
        for alias in aliases.split('|'):
            key = COLOR_normalize(alias)
            assert key not in lookup
            lookup[key] = name
    ordered = sorted(lookup, key=lambda term: (-len(term), term))
    pieces = [re.escape(term) if re.search('[\\u4e00-\\u9fff]', term) else '(?<!\\w)' + re.escape(term) + '(?!\\w)' for term in ordered]
    return (re.compile('|'.join(pieces)), lookup)

FOOD_RULES = [('Soft drinks', 'Drink', 'soda|sodas|soft drink|soft drinks|softdrink|pop|pops|cola|coke|coca cola|coke cola|diet coke|pepsi|sprite|fanta|root beer|fizzy drinks', 'Broad category: generic soft drinks and named brands are merged; original text is retained.'), ('Beer', 'Drink', 'beer|beers|bier', 'Includes bier; excludes root beer. Ambiguous bear is left for review.'), ('Hot dog', 'Food', 'hot dog|hot dogs|hotdog|hotdogs', 'Spacing and plural variants; hyphens normalized.'), ('Popcorn', 'Food', 'popcorn|popcorns|pop corn|popocorn', 'Includes observed popocorn typo; pop corn is matched before pop.'), ('Pizza', 'Food', 'pizza|pizzas', 'Singular and plural.'), ('Fries', 'Food', 'fries|french fries', 'Does not infer that bare chips means fries.'), ('Chips', 'Food', 'chips|potato chips|crisps|cheetos|hot cheetos', 'Includes bare chips, potato chips and crisps. Bare chips is not inferred to mean fries; fish and chips is outside this selected list.'), ('Burger', 'Food', 'burger|burgers|hamburger|hamburgers|slider|sliders|bergers', 'Groups sliders with burgers; includes observed bergers typo.'), ('Nachos', 'Food', 'nacho|nachos|natchos', 'Includes observed natchos typo.'), ('Chicken', 'Food', 'chicken wings|chickenwings|wings|buffalo wings|fried chicken|fry chicken|chicken tenders|tenders|chicken', 'Includes wings, fried chicken, tenders and unspecified chicken; each answer counts once even if it names several parts.'), ('Peanuts', 'Food', 'peanuts|peanut|fried peanuts', 'Kept separate from unspecified nuts.'), ('Energy / electrolyte drinks', 'Drink', 'energy drink|energy drinks|red bull|redbull|monster drink|sports drinks|sports drink|gatorade|gaderade|powerade|electrolyte drinks|电解质饮料|东鹏补水水', 'User-selected analysis group combining energy, sports and electrolyte drinks; these are not nutritionally equivalent.'), ('Water', 'Drink', 'water|watwr', 'Includes observed watwr typo; word boundaries prevent substring matches.'), ('Sandwich', 'Food', 'sandwich|sandwiches|meatball subs|uncrustables', 'Groups named sandwich products with sandwiches.'), ('Poutine', 'Food', 'poutine', 'Kept separate from fries.'), ('Pretzel', 'Food', 'pretzel|pretzels|soft pretzels', 'Singular and plural.'), ('Sunflower seeds', 'Food', 'sunflower seeds', 'Specific seed snack.'), ('Barbecue', 'Food', 'bbq|barbecue', 'Generic barbecue wording; specific skewers remain separate.'), ('Gum', 'Food', 'gum|gums|chewing gum|bubblegum', 'Chewing gum is retained; tobacco and nicotine products are not foods.'), ('Tea', 'Drink', 'tea|black tea|iced tea|bubble tea|milk tea|teh tarik', 'Broad tea category, including milk/bubble tea; aliases remain visible.'), ('Ice cream / frozen treats', 'Food', 'ice cream|icecream|popsicle|freezies|frosty', 'Broad frozen-treat category.')]

def FOOD_normalize(text):
    text = unicodedata.normalize('NFKC', text).casefold()
    return ' '.join(re.sub('[^\\w\\s]', ' ', text).split())

FOOD_MISSING_ANSWERS = {FOOD_normalize(value) for value in ['', 'none', 'nothing', 'no', 'unsure', 'n/a', 'na', 'none in particular', 'nothing in particular', 'no food or drink', 'no food or drink in particular', 'N/A, since I know nothing about it at all', "I haven't ever watched it before.", 'None, the sport is not popular', 'Perhaps some refreshments for the athletes but I do not associate any food or drink with the viewing experience.']}

def FOOD_make_matcher():
    alias_to_name = {}
    for name, kind, aliases, note in FOOD_RULES:
        for alias in aliases.split('|'):
            key = FOOD_normalize(alias)
            assert key not in alias_to_name, key
            alias_to_name[key] = name
    alias_to_name['fish and chips'] = None
    ordered = sorted(alias_to_name, key=lambda x: (-len(x), x))
    pieces = [re.escape(a) if re.search('[\\u4e00-\\u9fff]', a) else '(?<!\\w)' + re.escape(a) + '(?!\\w)' for a in ordered]
    return (re.compile('|'.join(pieces)), alias_to_name)

FEELING_FEELINGS = {'Excited': 'excited|exciting|excitement|excitements|excitments|exictement|thrilled|thrilling|thrill|exhilarated|exhilarating|hyped|ecstatic|euphoric|euphoria|giddy|兴奋', 'Bored': 'bored|boring|boredom|monotonous', 'Interested / engaged': 'interested|interesting|intrigued|intriguing|engaged|engagement|immersed|invested|captivated|captivating|fascinating|fascinated|engrossed|enthralled|absorbed|involved|curious|glued to the game', 'Uninterested': 'uninterested|unattentive|inattentive|not my type', 'Neutral / indifferent': 'neutral|indifferent|unfazed|detached|ordinary', 'Happy / joyful': 'happy|happiness|joy|joyful|joyed|cheerful', 'Entertained': 'entertained|entertaining|amused|amusing|fun|enjoy|enjoyed|enjoyable|satisfying|satisfied|love it', 'Energetic / intense': 'energetic|energized|energised|energetized|invigorated|electric|high energy|full of energy|energy|intense|frenetic|adrenaline|adrenaline rush|dopamine release', 'Calm / relaxed': 'calm|relaxed|relaxing|relaxation|peaceful|comfortable|comfort|chill|meditative|patient|patience', 'Tense / anxious': 'tense|tension|anxious|anxiety|nervous|nervousness|stressed|stressful|stress|stressed out|jittery|on edge|on the edge of my seat|restless', 'Anticipatory / suspenseful': 'anticipation|anticipatory|anticipating|suspense|suspenseful|in suspense', 'Focused / analytical': 'focused|focussed|foucsed|concentrated|analytical|paying attention|locked into', 'Confused': 'confused|confusing|confusion|confusions|confuse', 'Passionate': 'passionate|passion|passinate|enthusiasm|enthusiastic|enthused', 'Angry / frustrated': 'angry|anger|annoyed|frustrated|frustration|contempt', 'Sad / disappointed': 'sad|sadness|sorrow|disappointed|disappointment|heartbreak|heartbroken|regretfulness|pity', 'Motivated / inspired': 'motivated|motivating|motivation|inspired|inspiring|moved', 'Impressed / amazed': 'impressed|amazed|amazing|awe|admiring|admiration|appreciation|surprised', 'Tired / sleepy': 'tired|sleepy|exhausted|fatigued', 'Scared / overwhelmed': 'scared|fearful|afraid|overwhelmed', 'Connected': 'connected|social connection|sense of belonging|united|a part of something bigger', 'Proud': 'proud|pride', 'Nostalgic': 'nostalgic|nostalgia', 'Competitive': 'competitive', 'Emotional': 'emotional'}

def FEELING_normalize(text):
    text = unicodedata.normalize('NFKC', text).casefold().replace('’', "'")
    text = text.replace('\n', '; ')
    text = re.sub('[^\\w\\s.,;:!?]', ' ', text)
    return ' '.join(text.split())

def FEELING_plain(text):
    return ' '.join(re.sub('[^\\w\\s]', ' ', FEELING_normalize(text)).split())

FEELING_MISSING = {'', 'none', 'nothing', 'no sense', 'unknown', 'unsure', 'not sure', 'idk', 'n a', 'na', 'no idea', 'don t know', 'i don t know', 'i do not know', 'i don t feel much in particular'}

FEELING_UNRELATED = {FEELING_plain('I feel as though it is tense about checkpoints security all or nothing.'): 'Off-topic statement about checkpoint security'}

FEELING_NO_WATCH = re.compile('\\b(?:don t|dont|do not)(?:\\s+\\w+){0,2}\\s+watch\\b|\\bnever(?:\\s+\\w+){0,3}\\s+watch(?:ed)?\\b')

FEELING_NEGATION = re.compile('\\b(?:not|never|no|don t|dont|doesn t|didn t|isn t|aren t|wasn t|weren t|can t|cannot)\\b')

def FEELING_make_matcher():
    lookup = {}
    for category, variants in FEELING_FEELINGS.items():
        for variant in variants.split('|'):
            assert variant not in lookup
            lookup[variant] = category
    terms = sorted(lookup, key=lambda term: (-len(term), term))
    patterns = [re.escape(t) if re.search('[\\u4e00-\\u9fff]', t) else '(?<!\\w)' + re.escape(t) + '(?!\\w)' for t in terms]
    return (re.compile('|'.join(patterns)), lookup)

def FEELING_extract_answer(raw, matcher, lookup):
    text = FEELING_normalize(raw)
    simplified = FEELING_plain(raw)
    if not simplified:
        return ({}, 'Blank answer', [])
    if simplified in FEELING_MISSING:
        return ({}, 'No specific feeling / does not know', [])
    if simplified in FEELING_UNRELATED:
        return ({}, FEELING_UNRELATED[simplified], [])
    if FEELING_NO_WATCH.search(simplified):
        return ({}, 'Explicitly does not watch / has never watched the sport', [])
    categories, suppressed = ({}, [])
    for match in matcher.finditer(text):
        phrase = match.group()
        category = lookup[phrase]
        clause = re.split('[.,;:!?]|\\b(?:but|however|although|yet|because|so|then)\\b', text[:match.start()])[-1]
        prefix = ' '.join(clause.split()[-5:])
        if re.search('\\bused to\\b', clause) and 'no longer' in text[match.end():]:
            suppressed.append(phrase + ' (past feeling explicitly no longer present)')
            continue
        if FEELING_NEGATION.search(prefix) and (not re.search('\\bnot only\\b', prefix)):
            if category == 'Interested / engaged':
                categories.setdefault('Uninterested', []).append('negated ' + phrase)
            else:
                suppressed.append(phrase + ' (negated)')
            continue
        categories.setdefault(category, []).append(phrase)
    if simplified in {'good', 'neutral good', 'good motivated excited'}:
        categories.setdefault('Happy / joyful', []).append('good (direct feeling)')
    if re.search('\\bnever really had an interest\\b', simplified):
        categories.setdefault('Uninterested', []).append('never really had an interest')
    reason = '' if categories else 'No recognized feeling expression; unrelated, vague or unmapped response'
    return (categories, reason, suppressed)

SONG_SONG_RULES = {'Waka Waka': ['waka waka', 'waka wake', 'wake waka', 'this time for africa', "it's time for africa"], 'Take Me Out to the Ball Game': ['take me out to the ball game', 'take me out to the ballgame', 'take me out to the ball gam'], 'We Will Rock You': ['we will rock you'], "Wavin' Flag": ['wavin flag', 'wavin flags', 'waving flag', 'waving flags'], 'Sirius': ['sirius'], 'Space Jam': ['space jam'], 'We Are One': ['we are one'], 'We Are the Champions': ['we are the champions'], 'Roundball Rock': ['roundball rock'], 'See You Again': ['see you again'], 'Sweet Caroline': ['sweet caroline'], 'Centerfield': ['centerfield'], 'Sicko Mode': ['sicko mode'], 'Forever': ['forever'], 'Seven Nation Army': ['seven nation army'], 'Remember the Name': ['remember the name'], 'Hall of Fame': ['hall of fame', 'hall of the fame'], 'Dai Dai': ['dai dai', 'daidai'], 'Love Me Again': ['love me again'], "God's Plan": ["god's plan"], 'Wonderwall': ['wonderwall']}

def SONG_normalize(raw):
    """Normalize case, spacing and punctuation without guessing song titles."""
    text = unicodedata.normalize('NFKC', raw).casefold()
    text = re.sub('https?://\\S+', ' ', text)
    return ' '.join(re.sub('[^\\w\\s]', ' ', text).split())

SONG_NO_SPECIFIC_SONG = {SONG_normalize(text) for text in ['none', 'n/a', 'na', 'nil', 'no', 'nothing', 'unsure', 'unknown', 'idk', 'no idea', 'i have no idea', 'really no idea', 'no song', 'no choice', '-', '#NAME?', 'emm', 'no particular song', 'no perticular song', 'no song in particular', 'no particular song comes to mind', 'none in particular', 'nothing really', 'nothing specific', 'nothing in particular, something with a good beat?', "don't have", 'cant think of', "i don't know", 'i dont know', "i don't no", "sorry i don't know", "i don't know any songs", 'none i do not know of any', 'n/a since i know nothing about it at all', 'none i basically know nothing about the sport', 'i do not associate a song with this sport']}

def SONG_make_matcher():
    lookup = {}
    for title, variants in SONG_SONG_RULES.items():
        for variant in variants:
            alias = SONG_normalize(variant)
            if alias in lookup and lookup[alias] != title:
                raise ValueError(f'Ambiguous song alias: {alias}')
            lookup[alias] = title
    ordered = sorted(lookup, key=lambda term: (-len(term), term))
    pattern = '|'.join(('(?<!\\w)' + re.escape(term) + '(?!\\w)' for term in ordered))
    return (re.compile(pattern), lookup)

def SONG_parse_answer(raw, matcher, lookup):
    """Return candidate titles and their matched phrases, without a cutoff."""
    matches = {}
    for match in matcher.finditer(SONG_normalize(raw)):
        matches.setdefault(lookup[match.group()], set()).add(match.group())
    return matches


FREE_TEXT_COLUMNS = {
    "color": "What colour do you associate with this sport?",
    "food": "What foods and drinks do you associate with watching this sport in the place or culture you know best?",
    "feeling": "How do you feel when you watch this sport?",
    "song": "Which song do you associate with this sport?",
}
DICTIONARY_PATHS = {
    "color": "distribution_results/color/color_dictionary.csv",
    "food": "distribution_results/food_and_drinks/food_dictionary.csv",
    "feeling": "distribution_results/feeling/feeling_dictionary.csv",
    "song": "distribution_results/song/song_dictionary.csv",
}
NAME_COLUMNS = {"color":"color", "food":"food_name", "feeling":"feeling", "song":"song"}

def load_dictionary_names(project_dir):
    names = {}
    for kind, rel in DICTIONARY_PATHS.items():
        path = Path(project_dir) / rel
        if not path.exists():
            raise FileNotFoundError(f"Required dictionary missing: {path}")
        dictionary = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
        if kind == "song":
            dictionary = dictionary.loc[dictionary["selected"].astype(str).str.casefold().eq("true")]
        names[kind] = dictionary[NAME_COLUMNS[kind]].tolist()
    return names

def extract_free_text(df, names_by_kind):
    out = pd.DataFrame(index=df.index)
    matchers = {
        "color": COLOR_make_matcher(), "food": FOOD_make_matcher(),
        "feeling": FEELING_make_matcher(), "song": SONG_make_matcher(),
    }
    for kind, names in names_by_kind.items():
        col = FREE_TEXT_COLUMNS[kind]
        if col not in df.columns:
            raise ValueError(f"Missing survey column: {col}")
        matcher, lookup = matchers[kind]
        selections, missing = [], []
        for raw in _text(df[col]):
            if kind == "color":
                norm = COLOR_normalize(raw)
                hits = {lookup[m.group()] for m in matcher.finditer(norm)}
                if any(re.search(pattern, norm) for pattern, _ in COLOR_EXCLUSION_RULES):
                    hits = set()
                is_missing = not norm
            elif kind == "food":
                norm = FOOD_normalize(raw)
                is_missing = norm in FOOD_MISSING_ANSWERS
                hits = set() if is_missing else {lookup[m.group()] for m in matcher.finditer(norm)}
                hits.discard(None)
            elif kind == "feeling":
                parsed, reason, _ = FEELING_extract_answer(raw, matcher, lookup)
                hits = set(parsed)
                is_missing = bool(reason)
            else:
                norm = SONG_normalize(raw)
                parsed = SONG_parse_answer(raw, matcher, lookup)
                hits = set(parsed) & set(names)
                is_missing = not hits
            selections.append(hits)
            missing.append(float(is_missing))
        for name in names:
            out[f"{kind}__{name}"] = [float(name in selected) for selected in selections]
        out[f"{kind}__missing"] = missing
    return out.astype(float)

class SurveyPreprocessor:
    def __init__(self, project_dir=None):
        self.project_dir = Path(project_dir) if project_dir else Path(__file__).resolve().parent
        self.text_names_ = load_dictionary_names(self.project_dir)

    def fit(self, train_df):
        numeric = extract_numeric(train_df)
        self.medians_ = numeric.median().fillna(0.0)
        filled = numeric.fillna(self.medians_)
        self.means_ = filled.mean()
        self.stds_ = filled.std(ddof=0).replace(0, 1.0).fillna(1.0)
        self.columns_ = self.transform(train_df).columns.tolist()
        return self

    def transform(self, df):
        if not hasattr(self, "medians_"):
            raise RuntimeError("Call fit(train_df) before transform(df)")
        numeric = extract_numeric(df)
        flags = numeric.isna().astype(float)
        flags.columns = [f"{c}__missing" for c in NUMERIC]
        scaled = (numeric.fillna(self.medians_) - self.means_) / self.stds_
        result = pd.concat([scaled, flags, extract_choices(df), extract_free_text(df, self.text_names_)], axis=1)
        if not np.isfinite(result.to_numpy(dtype=float)).all():
            raise ValueError("Non-finite values in encoded features")
        if hasattr(self, "columns_") and result.columns.tolist() != self.columns_:
            raise ValueError("Feature columns/order mismatch")
        return result

    def fit_transform(self, df):
        return self.fit(df).transform(df)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    folder = root / "data"
    if not (folder / "train.csv").exists():
        folder = root / "data_splits"
    train = pd.read_csv(folder / "train.csv", dtype=str, keep_default_na=False)
    val = pd.read_csv(folder / "validation.csv", dtype=str, keep_default_na=False)
    p = SurveyPreprocessor()
    a, b = p.fit_transform(train), p.transform(val)
    print("Train:", a.shape, "Validation:", b.shape)
    print("Feature order consistent:", a.columns.tolist() == b.columns.tolist())
    print("No missing/nonfinite values:", np.isfinite(a.to_numpy()).all() and np.isfinite(b.to_numpy()).all())
    assert a.shape[1] == 112, f"Expected 112 features, got {a.shape[1]}"