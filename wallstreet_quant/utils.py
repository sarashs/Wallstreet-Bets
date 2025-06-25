import re
import unicodedata
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import networkx as nx
import faiss
from sentence_transformers import SentenceTransformer

class CompanyDeduper:
    """Cluster near‑duplicate company names.

    Only one public method: :py:meth:`dedupe`.
    No module‑scope helper functions are exposed, preventing namespace clashes
    with other canonicalisers you may already have.
    """

    # ── class‑level constants (private) ──────────────────────────
    _SUFFIXES = {
        # English
        "inc", "incorporated", "corp", "corporation", "co", "company", "companies",
        "ltd", "limited", "plc", "llc", "llp", "lp",
        # EU
        "gmbh", "kg", "ag", "kgaa", "se",
        # Romance
        "sa", "sas", "sarl", "spa", "srl", "sl",
        # NL / Nordics
        "bv", "nv", "ab", "oy",
        # APAC
        "pte", "pty", "bhd", "sdn bhd", "kk",
        # misc
        "as", "doo",
    }
    _SUFFIX_RE = re.compile(r"\b(?:{})(?:[\s\.]|$)".format("|".join(_SUFFIXES)), re.I)
    _ACRONYM_RE = re.compile(r"^[A-Z]{2,5}$")
    _TICKER_RE  = re.compile(r"^[A-Z]{1,5}$")

    # ── ctor ────────────────────────────────────────────────────
    def __init__(self,
                 ticker_map: Dict[str, str],
                 *,
                 embed_model_name: str = "BAAI/bge-small-en-v1.5",
                 k_neighbors: int = 10,
                 cosine_th: float = 0.80,
                 embed_model: Optional[SentenceTransformer] = None) -> None:
        self._ticker_map = ticker_map
        self._k = k_neighbors
        self._cos_th = cosine_th
        self._embed_model = embed_model or SentenceTransformer(embed_model_name)

    # ── public API ──────────────────────────────────────────────
    def dedupe(self, raw_names: List[str]) -> Tuple[List[List[str]], Dict[str, int], List[str]]:
        """Return *(clusters, name▶cid map, representatives)*."""
        # 1. Expand tickers
        step1 = [self._expand_ticker(tok) for tok in raw_names]
        # 2. Expand acronyms
        step2 = [self._expand_acronym(tok, step1) if self._is_acronym(tok) else tok for tok in step1]
        # 3. Bucket by canonical key
        buckets: defaultdict[str, List[str]] = defaultdict(list)
        for nm in step2:
            buckets[self._canonicalise(nm)].append(nm)
        reps = [names[0] for names in buckets.values()]
        emb = self._embed(reps)
        G = self._build_graph(emb)
        return self._to_clusters(G, reps, buckets, raw_names)

    # ── private helpers (all names prefixed) ────────────────────
    @staticmethod
    def _normalise_unicode(text: str) -> str:
        return unicodedata.normalize("NFKD", text)

    def _canonicalise(self, name: str) -> str:
        s = self._normalise_unicode(name)
        s = self._SUFFIX_RE.sub(" ", s.lower())
        s = re.sub(r"[^\w ]", " ", s)
        return " ".join(sorted(s.split()))

    def _is_acronym(self, token: str) -> bool:
        return bool(self._ACRONYM_RE.fullmatch(token))

    def _expand_ticker(self, token: str) -> str:
        return self._ticker_map[token] if self._TICKER_RE.fullmatch(token) and token in self._ticker_map else token

    def _expand_acronym(self, acr: str, universe: List[str]) -> str:
        target = acr.upper()
        for cand in universe:
            if ''.join(w[0] for w in cand.split()).upper() == target:
                return cand
        return acr

    def _embed(self, strings: List[str]) -> np.ndarray:
        X = self._embed_model.encode(strings, convert_to_numpy=True)
        return X / np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-9)

    def _build_graph(self, X: np.ndarray) -> nx.Graph:
        dim = X.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(X)
        D, I = index.search(X, self._k)
        G = nx.Graph()
        G.add_nodes_from(range(len(X)))
        for i, (scores, idxs) in enumerate(zip(D, I)):
            for score, j in zip(scores, idxs):
                if j > i and score >= self._cos_th:
                    G.add_edge(i, j)
        return G

    def _to_clusters(self,
                     G: nx.Graph,
                     reps: List[str],
                     buckets: Dict[str, List[str]],
                     raw_names: List[str]) -> Tuple[List[List[str]], Dict[str, int], List[str]]:
        clusters: List[List[str]] = []
        name2cid: Dict[str, int] = {}
        representatives: List[str] = []
        
        for cid, comp in enumerate(nx.connected_components(G)):
            members: List[str] = []
            for rep_idx in comp:
                key = self._canonicalise(reps[rep_idx])
                members.extend(buckets[key])
            clusters.append(members)
            representatives.append(reps[next(iter(comp))])
            for m in members:
                name2cid[m] = cid
        
        # Ensure all raw names are mapped, even if they didn't make it through processing
        for raw_name in raw_names:
            if raw_name not in name2cid:
                # Find which processed version this maps to
                expanded = self._expand_ticker(raw_name)
                if self._is_acronym(expanded):
                    step1 = [self._expand_ticker(tok) for tok in raw_names]
                    expanded = self._expand_acronym(expanded, step1)
                
                # Find the cluster this belongs to
                found = False
                for cid, cluster in enumerate(clusters):
                    if raw_name in cluster or expanded in cluster:
                        name2cid[raw_name] = cid
                        found = True
                        break
                
                if not found:
                    # This shouldn't happen, but add as singleton cluster if it does
                    name2cid[raw_name] = len(clusters)
                    clusters.append([raw_name])
                    representatives.append(raw_name)
        
        return clusters, name2cid, representatives