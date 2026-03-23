import json
import random

JSON_PATH = "first_aid.json"
MATCH_THRESHOLD = 0.28

def _is_word_char(ch):
    return ch == "_" or ch.isalnum()

def normalize_text(text):
    t = text.lower().strip()
    buf = []
    for c in t:
        buf.append(c if _is_word_char(c) else " ")
    t = "".join(buf)
    return " ".join(t.split())


def word_set(text):
    n = normalize_text(text)
    if not n:
        return set()
    return {w for w in n.split() if w}


STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "as",
        "is",
        "was",
        "are",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "am",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "its",
        "our",
        "their",
        "how",
        "when",
        "where",
        "why",
        "there",
        "here",
        "any",
        "all",
        "each",
        "every",
        "some",
        "such",
        "no",
        "not",
        "only",
        "just",
        "also",
        "now",
        "get",
        "got",
        "about",
        "into",
        "from",
        "with",
        "by",
        "without",
        "then",
        "than",
        "too",
        "very",
    }
)


def stemish_token(w):
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss") and w[-2] != "s":
        return w[:-1]
    return w

def content_stem_set(words):
    out = set()
    for w in words:
        if w in STOPWORDS:
            continue
        out.add(stemish_token(w))
    return out

def fuzzy_word_set(words):
    out = set()
    for w in words:
        out.add(w)
        st = stemish_token(w)
        if st != w:
            out.add(st)
    return out

def levenshtein_distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        cur = [i + 1]
        for j, cb in enumerate(b):
            ins = previous[j + 1] + 1
            delete = cur[j] + 1
            sub = previous[j] + (ca != cb)
            cur.append(min(ins, delete, sub))
        previous = cur
    return previous[-1]

def levenshtein_similarity(a, b):
    if not a and not b:
        return 1.0
    d = levenshtein_distance(a, b)
    denom = max(len(a), len(b), 1)
    return 1.0 - (d / denom)

def jaccard_similarity(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0

def token_match_score(u, p):
    if u == p:
        return 1.0
    su, sp = stemish_token(u), stemish_token(p)
    if su and su == sp:
        return 1.0
    return levenshtein_similarity(u, p)

def content_token_alignment(u_c, p_c):
    if not u_c or not p_c:
        return 0.0
    bests = [max((token_match_score(u, p) for p in p_c), default=0.0) for u in u_c]
    prod = 1.0
    for b in bests:
        prod *= max(b, 1e-9)
    geom = prod ** (1.0 / len(bests))
    if len(p_c) < len(u_c):
        strong = [b for b in bests if b >= 0.42]
        if strong:
            return max(geom, sum(strong) / len(strong))
    return geom

class IntentKB:
    def __init__(self, data):
        if "intents" not in data or not isinstance(data["intents"], list):
            raise ValueError("JSON must contain a non-empty 'intents' array.")
        self.intents = data["intents"]
        self.patterns = []
        self._inverted = {}

        pid = 0
        for intent in self.intents:
            tag = intent.get("tag", "")
            responses = intent.get("responses", [])
            if not isinstance(responses, list):
                responses = [str(responses)]
            raw_patterns = intent.get("patterns", [])
            if isinstance(raw_patterns, str):
                raw_patterns = [raw_patterns]
            for p in raw_patterns:
                ps = str(p)
                norm = normalize_text(ps)
                words = word_set(ps)
                rec = {
                    "id": pid,
                    "tag": tag,
                    "raw": ps,
                    "norm": norm,
                    "words": words,
                    "content_stem": content_stem_set(words),
                    "responses": responses,
                }
                self.patterns.append(rec)
                for w in words:
                    self._inverted.setdefault(w, set()).add(pid)
                    st = stemish_token(w)
                    if st != w:
                        self._inverted.setdefault(st, set()).add(pid)
                pid += 1

        if not self.patterns:
            raise ValueError("No patterns found in intents.")

    def candidate_ids(self, user_words):
        cands = set()
        expanded = set()
        for w in user_words:
            expanded.add(w)
            expanded.add(stemish_token(w))
        for w in expanded:
            cids = self._inverted.get(w)
            if cids:
                cands |= cids
        if not cands:
            cands = set(range(len(self.patterns)))
        return cands

    def match_score(self, user_norm, user_words, rec):
        lev = levenshtein_similarity(user_norm, rec["norm"])
        jac_all = jaccard_similarity(fuzzy_word_set(user_words), fuzzy_word_set(rec["words"]))
        u_c = content_stem_set(user_words)
        p_c = rec["content_stem"]
        if u_c or p_c:
            jac_content = jaccard_similarity(u_c, p_c)
        else:
            jac_content = jac_all
        if u_c and p_c:
            align = content_token_alignment(u_c, p_c)
            content_combo = 0.58 * jac_content + 0.42 * align
        else:
            content_combo = jac_content
        return 0.02 * lev + 0.08 * jac_all + 0.90 * content_combo

    def best_match(self, user_input, threshold):
        user_norm = normalize_text(user_input)
        user_words = word_set(user_input)

        if not user_norm:
            return None, [], 0.0

        best_rec = None
        best_score = -1.0

        for idx in self.candidate_ids(user_words):
            rec = self.patterns[idx]
            s = self.match_score(user_norm, user_words, rec)
            if best_rec is None:
                best_rec = rec
                best_score = s
                continue
            if s > best_score + 1e-9:
                best_score = s
                best_rec = rec
            elif abs(s - best_score) < 1e-9:
                cur_key = (len(rec["norm"]), -rec["id"])
                old_key = (len(best_rec["norm"]), -best_rec["id"])
                if cur_key > old_key:
                    best_rec = rec

        if best_rec is None or best_score < threshold:
            return None, [], best_score
        return best_rec["tag"], list(best_rec["responses"]), best_score


def default_unknown_responses():
    return [
        "I'm not sure how to answer that.",
        "Could you rephrase that?",
        "I don't have a good match for that yet.",
    ]


def chat_loop(kb, threshold):
    unknown = default_unknown_responses()
    print(
        "Type 'quit' or 'exit' to stop.\n"
    )
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Goodbye!")
            break
        if user_input.lower() in {"quit", "exit"}:
            print("Bot: Goodbye!")
            break
        tag, responses, _score = kb.best_match(user_input, threshold)
        if tag is None or not responses:
            print(f"Bot: {random.choice(unknown)}")
        else:
            print(f"Bot: {random.choice(responses)}")


def main():
    path = JSON_PATH
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except OSError as e:
        raise SystemExit(1)
    except json.JSONDecodeError as e:
        raise SystemExit(1)

    try:
        kb = IntentKB(data)
    except ValueError as e:
        raise SystemExit(1)

    t = MATCH_THRESHOLD
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    chat_loop(kb, t)


if __name__ == "__main__":
    main()
