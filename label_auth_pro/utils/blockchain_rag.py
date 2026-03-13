"""
Blockchain Product Provenance + LLM-Embedding RAG
- Simulates immutable SHA-256 blockchain for product lifecycle tracking
- TF-IDF sentence embeddings (simulates Hugging Face sentence-transformers)
- Cosine similarity retrieval for multimodal product blockchain footprints
- Automated authenticity report generation
"""

import numpy as np
import json, hashlib, os
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ─── Blockchain ───────────────────────────────────────────────────────────────

class Block:
    def __init__(self, idx, data, prev_hash="0"*64):
        self.idx       = idx
        self.timestamp = datetime.now().isoformat()
        self.data      = data
        self.prev_hash = prev_hash
        self.nonce     = 0
        self.hash      = self._hash()

    def _hash(self):
        payload = json.dumps({
            "idx": self.idx, "ts": self.timestamp,
            "data": self.data, "prev": self.prev_hash, "nonce": self.nonce
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_dict(self):
        return {"idx":self.idx,"timestamp":self.timestamp,
                "data":self.data,"prev_hash":self.prev_hash[:12]+"…",
                "hash":self.hash[:16]+"…"}


class ProductBlockchain:
    """Append-only blockchain for product provenance tracking."""

    def __init__(self):
        self.chain   = [Block(0, {"type":"genesis","msg":"LabelAuth Chain v2"})]
        self.records = {}   # product_id → [block.to_dict(), …]

    def add(self, product_id: str, event: dict) -> str:
        blk = Block(len(self.chain),
                    {"product_id": product_id, "event": event},
                    self.chain[-1].hash)
        self.chain.append(blk)
        self.records.setdefault(product_id, []).append(blk.to_dict())
        return blk.hash

    def footprint(self, product_id: str) -> list:
        return self.records.get(product_id.upper(), [])

    def verify(self) -> bool:
        for i in range(1, len(self.chain)):
            if self.chain[i].prev_hash != self.chain[i-1].hash:
                return False
        return True


SUPPLY_STAGES = [
    ("MANUFACTURED",  "Factory QC PASS — hologram & QR encoded"),
    ("QUALITY_CHECK", "Lab verified: spectral authenticity OK"),
    ("PACKAGED",      "Tamper-proof seal applied, RFID tagged"),
    ("DISPATCHED",    "Shipped via logistics partner, GPS active"),
    ("CUSTOMS",       "Customs cleared — border scan AUTHENTIC"),
    ("WAREHOUSE",     "Received at regional distribution center"),
    ("RETAIL",        "Scan-in at retail entry — AUTHENTIC"),
]

LOCATIONS = ["Shanghai","Shenzhen","Rotterdam","Hamburg","Chicago",
             "Mumbai","Dubai","London","Toronto","Singapore"]


def build_blockchain(n_products=30, rng_seed=42) -> ProductBlockchain:
    rng   = np.random.default_rng(rng_seed)
    chain = ProductBlockchain()

    for pid_i in range(1, n_products+1):
        pid = f"PROD-{pid_i:04d}"
        for stage, desc in SUPPLY_STAGES:
            event = {
                "stage":       stage,
                "description": desc,
                "location":    rng.choice(LOCATIONS),
                "temperature": f"{rng.integers(14,32)}°C",
                "humidity":    f"{rng.integers(30,80)}%",
                "scan_result": "AUTHENTIC",
                "operator_id": f"OP-{rng.integers(100,999)}",
                "lot":         f"LOT-{rng.integers(1000,9999)}",
            }
            chain.add(pid, event)

    # Inject tampered/counterfeit products
    for pid_i, verdict, reason in [
        (4, "TAMPERED",    "Adhesive failure detected at warehouse scan"),
        (9, "COUNTERFEIT", "QR fingerprint mismatch — clone detected"),
        (14,"TAMPERED",    "Scratch damage reported by retail staff"),
        (20,"COUNTERFEIT", "Serial number duplication found"),
    ]:
        chain.add(f"PROD-{pid_i:04d}", {
            "stage": "ANOMALY_DETECTED", "description": reason,
            "scan_result": verdict, "alert_level": "HIGH",
            "detected_by": "LabelAuth-ViT v2.0",
        })

    return chain


# ─── LLM Embedding RAG ────────────────────────────────────────────────────────

DOMAIN_KNOWLEDGE = [
    # Authenticity science
    "Genuine labels exhibit uniform QR code finder patterns with correct Reed-Solomon ECC.",
    "Tampered labels show adhesive failure: irregular reflectance, edge lifting, ink smear.",
    "Counterfeit labels have QR distortions in finder pattern cells and color channel drift >12%.",
    "Degraded labels exhibit yellowing (R-channel +15%), reduced contrast, and surface micro-cracks.",
    "Hologram verification: authentic labels have consistent iridescent diagonal stripe spacing.",
    "Microtext density below 0.04 per pixel indicates label degradation or low-res counterfeiting.",
    "Surface roughness variance >0.06 combined with linear gradient anomaly confirms scratching.",
    "High-frequency FFT energy ratio >0.75 signals QR finder pattern damage or counterfeiting.",
    "Color coherence standard deviation >0.18 across RGB channels indicates counterfeit printing.",
    "Vision Transformer attention rollout highlights tampered patches via anomalous attention weights.",
    # Model knowledge
    "ViT patch embeddings encode 8×8 spatial blocks; CLS token aggregates global label representation.",
    "Hybrid CNN-ViT: CNN stem captures micro-textures, ViT head captures label-region relationships.",
    "Optuna hyperparameter tuning optimizes GBM learning rate, depth, and subsample ratio.",
    "Ensemble voting (GBM+RF+ET+MLP+SVC) reduces variance and improves generalization by ~8%.",
    "k-fold cross-validation with inferential t-test confirms significance vs random baseline.",
    # Blockchain knowledge
    "Blockchain provenance records create tamper-proof audit trails for product verification.",
    "SHA-256 chained blocks prevent retroactive modification of supply chain scan history.",
    "Product footprint includes: MANUFACTURED → QC → PACKAGED → SHIPPED → CUSTOMS → RETAIL.",
    "Anomaly injection: TAMPERED and COUNTERFEIT verdicts are permanently recorded on chain.",
    "LLM embeddings enable semantic retrieval of similar historical tampering cases.",
    # Supply chain context
    "Supply chain consistency: gap >30 days between stages is a provenance risk indicator.",
    "Temperature excursion during shipping degrades label adhesive, causing peel artifacts.",
    "QR code scan reliability drops below 85% for labels with >20% surface damage.",
    "Retail staff report adhesive failures most frequently in humid climates (>70% RH).",
    "Duplicate serial detection requires cross-referencing blockchain records at customs stage.",
]


class EmbeddingRAG:
    """
    LLM-embedding-based RAG using TF-IDF sentence vectors.
    Simulates HuggingFace sentence-transformers without GPU requirement.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=512, ngram_range=(1,2),
                                          sublinear_tf=True)
        self.documents  = list(DOMAIN_KNOWLEDGE)
        self.metadata   = [{"type":"domain","pid":None}] * len(DOMAIN_KNOWLEDGE)
        self.embeddings = None
        self._fitted    = False

    def build(self, blockchain: ProductBlockchain):
        """Ingest blockchain records into knowledge base."""
        for pid, records in blockchain.records.items():
            for rec in records:
                ev = rec["data"]["event"]
                text = (f"Product {pid} stage {ev.get('stage','')}. "
                        f"{ev.get('description','')}. "
                        f"Location: {ev.get('location','')}. "
                        f"Result: {ev.get('scan_result','UNKNOWN')}. "
                        f"Lot: {ev.get('lot','')}.")
                self.documents.append(text)
                self.metadata.append({"type":"blockchain","pid":pid,
                                      "stage":ev.get("stage",""),
                                      "result":ev.get("scan_result","")})

        self.embeddings = self.vectorizer.fit_transform(self.documents).toarray().astype(np.float32)
        self._fitted    = True
        print(f"  RAG knowledge base: {len(self.documents)} docs, "
              f"{self.embeddings.shape[1]}-dim embeddings")

    def retrieve(self, query: str, top_k=5) -> list:
        if not self._fitted:
            return []
        q    = self.vectorizer.transform([query]).toarray().astype(np.float32)
        sims = cosine_similarity(q, self.embeddings)[0]
        top  = np.argsort(sims)[::-1][:top_k]
        return [{"score": float(sims[i]),
                 "text":  self.documents[i][:250],
                 "meta":  self.metadata[i]}
                for i in top if sims[i] > 0.01]

    def generate_report(self, pid: str, pred: dict,
                         blockchain: ProductBlockchain) -> str:
        """Generate a structured authenticity report with RAG context."""
        footprint   = blockchain.footprint(pid)
        query       = (f"product {pid} classified {pred['label']} "
                       f"confidence {pred['confidence']:.0%} "
                       f"probabilities {pred['probabilities']}")
        retrieved   = self.retrieve(query, top_k=4)
        stages      = [r["data"]["event"]["stage"] for r in footprint] if footprint else ["N/A"]
        last_result = (footprint[-1]["data"]["event"].get("scan_result","N/A")
                       if footprint else "N/A")

        lines = [
            "╔══════════════════════════════════════════════════════════╗",
            f"║  LABEL AUTHENTICITY REPORT  ·  {pid:<24}  ║",
            "╠══════════════════════════════════════════════════════════╣",
            f"║  Classification : {pred['label']:<14}  "
            f"  Confidence : {pred['confidence']:.1%}     ║",
            f"║  Blockchain scan: {last_result:<14}  "
            f"Chain blocks : {len(footprint):<6}  ║",
            "╠══════════════════════════════════════════════════════════╣",
            "║  CLASS PROBABILITIES                                     ║",
        ]
        for cls, prob in pred["probabilities"].items():
            bar = "█"*int(prob*22) + "░"*(22-int(prob*22))
            lines.append(f"║  {cls:<12} [{bar}] {prob:.1%}  ║")
        lines += [
            "╠══════════════════════════════════════════════════════════╣",
            f"║  SUPPLY CHAIN ({len(footprint)} events)                          ║",
        ]
        for st in stages[-6:]:
            lines.append(f"║   ✓ {st:<52}  ║")
        lines += [
            "╠══════════════════════════════════════════════════════════╣",
            f"║  RAG CONTEXT ({len(retrieved)} retrieved documents)               ║",
        ]
        for r in retrieved[:3]:
            txt = r["text"][:54]
            lines.append(f"║   [{r['score']:.2f}] {txt:<54}  ║")
        lines.append("╚══════════════════════════════════════════════════════════╝")
        return "\n".join(lines)

    def save(self, path):
        import joblib
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump({"vec": self.vectorizer, "docs": self.documents,
                     "emb": self.embeddings, "meta": self.metadata}, path)

    def load(self, path):
        import joblib
        d = joblib.load(path)
        self.vectorizer = d["vec"]; self.documents = d["docs"]
        self.embeddings = d["emb"]; self.metadata   = d["meta"]
        self._fitted    = True
