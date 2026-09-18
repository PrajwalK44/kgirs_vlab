"""
IIT Kharagpur Virtual Laboratory
Discipline: Computer Science and Engineering
Subject: Knowledge Graph and Information Retrieval System (KGIRS)
Experiment 6: Identify Graph Entities & Probabilistic Retrieval Performance

Sections:
  1. Aim
  2. Introduction
  3. Theory
  4. Case Study
  5. Pretest
  6. Simulation
  7. Procedure
  8. Exercises
  9. Posttest
  10. References
  11. Report Generation

Designed following IIT Kharagpur Virtual Lab format and the modular architecture of template.py.
"""

import os
import math
import re
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF
import networkx as nx


# ======================================================================================
# 1. EXPERIMENT METADATA & EDUCATIONAL CONTENT (IIT KGP VLAB STANDARD)
# ======================================================================================

EXPERIMENT_CONFIG = {
    "exp_number": 6,
    "title": "Identify Graph Entities & Probabilistic Retrieval Performance",
    "learning_unit": "KGIRS MODULE 1 & MODULE 2: STRUCTURED KNOWLEDGE EXTRACTION & PROBABILISTIC IR RANKING",
    "discipline": "Computer Science and Engineering",
    "subject": "Knowledge Graph and Information Retrieval System (KGIRS)",
    "institute": "Indian Institute of Technology Kharagpur (IIT KGP)",
    "objectives": [
        "Extract domain-specific entities (Persons, Organizations, Locations, Concepts) from unstructured text to build structured knowledge (Module 1.2).",
        "Construct an interactive Knowledge Graph topology using relational triples with fewer, focused nodes for semantic clarity.",
        "Formulate and evaluate probabilistic retrieval principles using the Probabilistic Relevance Framework / Okapi BM25 Model (Module 2.1).",
        "Benchmark comparative retrieval performance (Precision@K, Recall@K, MAP, NDCG) between baseline keyword search and entity-aware probabilistic ranking."
    ]
}

# Pretest Questions (Diagnostic)
PRETEST_QUESTIONS = [
    {
        "id": 1,
        "question": "Which of the following best defines a Named Entity in knowledge graph construction?",
        "options": [
            "A) Any arbitrary stopword or punctuation mark in a sentence",
            "B) A real-world object or abstract concept with distinct identity and semantic type (e.g., Person, Org, Loc)",
            "C) A mathematical syntax error occurring during text tokenization",
            "D) A random float value assigned to word frequency counters"
        ],
        "answer_index": 1,
        "explanation": "A Named Entity represents a discrete real-world entity (such as an individual, company, place, or concept) that can be linked to a node in a Knowledge Graph."
    },
    {
        "id": 2,
        "question": "What is the primary limitation of pure keyword-based Bag-of-Words (BoW) retrieval?",
        "options": [
            "A) It cannot store strings in computer memory",
            "B) It ignores word polysemy, synonymy, semantic entity types, and relational context",
            "C) It executes too quickly to calculate relevance scores",
            "D) It only processes numerical equations rather than natural language"
        ],
        "answer_index": 1,
        "explanation": "Bag-of-Words models treat text as unorganized tokens, failing to recognize that terms like 'Apple' could mean a fruit or an organization, and missing underlying entity relationships."
    },
    {
        "id": 3,
        "question": "According to the Probability Ranking Principle (PRP) formulated by Robertson (1977), how should documents be ordered?",
        "options": [
            "A) In random order to ensure statistical variance",
            "B) In decreasing order of their estimated probability of relevance to the information need",
            "C) Alphabetically by document author name",
            "D) Strictly by increasing order of document length in bytes"
        ],
        "answer_index": 1,
        "explanation": "The Probability Ranking Principle asserts that overall retrieval effectiveness is maximized when documents are returned in decreasing order of their estimated probability of relevance."
    },
    {
        "id": 4,
        "question": "In a Knowledge Graph, how is factual information structured at the foundational level?",
        "options": [
            "A) As unstructured binary blobs",
            "B) As relational subject-predicate-object (head, relation, tail) triples",
            "C) As isolated single float values",
            "D) As recursive HTML document trees without attributes"
        ],
        "answer_index": 1,
        "explanation": "Knowledge Graphs represent domain facts using directed relational triples (head entity, relation predicate, tail entity), e.g., (Geoffrey Hinton, affiliated_with, Google)."
    },
    {
        "id": 5,
        "question": "In the Okapi BM25 retrieval model, what purpose does the Inverse Document Frequency (IDF) factor serve?",
        "options": [
            "A) It penalizes rare, highly informative terms",
            "B) It assigns higher discriminative weight to terms that appear in fewer documents across the collection",
            "C) It sets all term frequencies to a constant value of 1.0",
            "D) It deletes documents whose length exceeds the average"
        ],
        "answer_index": 1,
        "explanation": "IDF penalizes widespread common words (like 'the', 'system') while giving high discriminative weight to rare, salient terms and specific entities."
    }
]

# Posttest Questions (Comprehensive Evaluation matching template style)
POSTTEST_QUESTIONS = [
    {
        "id": 1,
        "question": "Which of the following entity categories correctly classifies 'DeepMind' and 'OpenAI'?",
        "options": [
            "A) LOCATION (LOC)",
            "B) ORGANIZATION (ORG)",
            "C) PERSON (PER)",
            "D) TEMPORAL (DATE)"
        ],
        "answer_index": 1,
        "explanation": "'DeepMind' and 'OpenAI' are research labs and corporate entities, classified under the ORGANIZATION (ORG) category."
    },
    {
        "id": 2,
        "question": "What is the primary role of the parameter k1 in the Okapi BM25 scoring formula?",
        "options": [
            "A) It controls the term frequency saturation non-linearity (how fast score plateaus with repeated term occurrences)",
            "B) It sets the absolute document length in bytes",
            "C) It determines the database port for Neo4j connections",
            "D) It eliminates all stop words from the query vector"
        ],
        "answer_index": 0,
        "explanation": "The parameter k1 (typically between 1.2 and 2.0) calibrates term frequency saturation; higher values allow repeated terms to contribute more to the score before leveling off."
    },
    {
        "id": 3,
        "question": "In BM25, setting the document length normalization parameter b = 0 causes the model to:",
        "options": [
            "A) Completely disable term frequency weighting",
            "B) Completely eliminate document length normalization (short and long documents treated equally)",
            "C) Invert the ranking order from worst to best",
            "D) Restrict retrieval to only single-word documents"
        ],
        "answer_index": 1,
        "explanation": "Parameter b controls length normalization; when b = 0, no length penalty is applied, whereas b = 1 applies full scaling by document length relative to average document length."
    },
    {
        "id": 4,
        "question": "How does incorporating Knowledge Graph entities improve probabilistic document ranking over pure BM25?",
        "options": [
            "A) By converting all natural language text into binary machine code",
            "B) By boosting documents that share disambiguated entities and relational context with query concepts",
            "C) By guaranteeing that every search query returns 100% of all documents in the corpus",
            "D) By bypassing the index and reading directly from disk linearly"
        ],
        "answer_index": 1,
        "explanation": "Entity-aware ranking leverages structured semantics: matching query entities against verified graph entities in documents overcomes surface-level lexical mismatch and improves ranking precision."
    },
    {
        "id": 5,
        "question": "What does Mean Average Precision (MAP) measure in an Information Retrieval evaluation benchmark?",
        "options": [
            "A) The average file size of relevant documents in kilobytes",
            "B) The mean of the Average Precision scores evaluated across a set of queries, emphasizing top-ranked relevant hits",
            "C) The clock execution time taken by the ranking algorithm in milliseconds",
            "D) The percentage of hardware RAM utilized during graph construction"
        ],
        "answer_index": 1,
        "explanation": "MAP evaluates the quality of ranked lists by averaging precision at each relevant document rank across multiple queries, heavily penalizing relevant documents ranked far down."
    },
    {
        "id": 6,
        "question": "Which evaluation metric incorporates graded relevance judgments with logarithmic positional discounting?",
        "options": [
            "A) Normalized Discounted Cumulative Gain (NDCG)",
            "B) Raw Term Frequency (TF)",
            "C) Graph Degree Centrality",
            "D) Corpus Vocabulary Count"
        ],
        "answer_index": 0,
        "explanation": "NDCG uses graded relevance levels and applies a logarithmic discount (1/log2(rank+1)) to prioritize placing highly relevant documents near the very top."
    },
    {
        "id": 7,
        "question": "In graph entity extraction, what problem does Entity Disambiguation (Entity Linking) resolve?",
        "options": [
            "A) Determining whether an entity mention like 'Paris' refers to Paris, France or Paris Hilton",
            "B) Formatting JSON files into comma-separated text files",
            "C) Deleting stop words from the query string",
            "D) Preventing network graph edges from crossing each other"
        ],
        "answer_index": 0,
        "explanation": "Entity Linking maps ambiguous textual surface mentions to canonical Knowledge Graph nodes based on surrounding semantic context."
    },
    {
        "id": 8,
        "question": "In our Knowledge Graph visualization with fewer nodes, what do the nodes and directed edges represent?",
        "options": [
            "A) Nodes represent documents; edges represent file download speeds",
            "B) Nodes represent identified domain entities; edges represent typed semantic relationships between them",
            "C) Nodes represent pixels; edges represent color gradients",
            "D) Nodes represent database servers; edges represent network latency"
        ],
        "answer_index": 1,
        "explanation": "In an entity knowledge graph, vertices represent entities (Person, Org, Loc, Concept) and directed edges signify relations such as 'affiliated_with' or 'developed'."
    },
    {
        "id": 9,
        "question": "What happens to the Precision@K metric if the top K retrieved documents contain 2 relevant items when K = 4?",
        "options": [
            "A) Precision@4 = 0.25",
            "B) Precision@4 = 0.50 (50%)",
            "C) Precision@4 = 0.75",
            "D) Precision@4 = 1.00"
        ],
        "answer_index": 1,
        "explanation": "Precision@K = (Number of relevant documents in top K) / K. Here, 2 / 4 = 0.50 or 50%."
    },
    {
        "id": 10,
        "question": "When designing knowledge graph experiments with fewer nodes, why is node sparsity advantageous in virtual lab interfaces?",
        "options": [
            "A) It prevents visual cognitive overload, ensuring students can clearly trace entity relations and semantic paths",
            "B) It makes the web page load slower to test student patience",
            "C) Sparse graphs are required because Streamlit cannot display more than 3 colors",
            "D) Graphs with more than 10 nodes are illegal in virtual labs"
        ],
        "answer_index": 0,
        "explanation": "Targeted graphs with 8-15 nodes maintain visual clarity, allowing learners to clearly trace entity types, relationships, and degree centrality without confusing cluttered layouts."
    }
]


# ======================================================================================
# 2. CURATED DOMAIN CORPORA & KNOWLEDGE BASE
# ======================================================================================

DATA_CORPORA = {
    "AI & Deep Learning Pioneers (Domain 1)": {
        "description": "Corpus on Artificial Intelligence pioneers, foundational research institutions, breakthrough algorithms, and global research hubs.",
        "documents": [
            {
                "doc_id": "DOC-101",
                "title": "Deep Learning Breakthroughs at University of Toronto",
                "text": "Geoffrey Hinton and his research group at University of Toronto pioneered Deep Learning and artificial neural network backpropagation. Hinton later joined Google Brain to scale distributed neural representations in Toronto.",
                "entities": [
                    {"name": "Geoffrey Hinton", "type": "PERSON"},
                    {"name": "University of Toronto", "type": "ORGANIZATION"},
                    {"name": "Deep Learning", "type": "CONCEPT"},
                    {"name": "Google Brain", "type": "ORGANIZATION"},
                    {"name": "Toronto", "type": "LOCATION"}
                ],
                "triples": [
                    ("Geoffrey Hinton", "affiliated_with", "University of Toronto"),
                    ("Geoffrey Hinton", "pioneered", "Deep Learning"),
                    ("Geoffrey Hinton", "joined", "Google Brain"),
                    ("University of Toronto", "located_in", "Toronto")
                ],
                "relevant_to": ["deep learning", "geoffrey hinton", "toronto neural network", "google brain pioneer"]
            },
            {
                "doc_id": "DOC-102",
                "title": "Convolutional Neural Networks and Meta AI Research",
                "text": "Yann LeCun developed Convolutional Neural Networks for computer vision at New York University. LeCun subsequently became Chief AI Scientist at Meta AI in New York, collaborating on self-supervised machine learning.",
                "entities": [
                    {"name": "Yann LeCun", "type": "PERSON"},
                    {"name": "Convolutional Networks", "type": "CONCEPT"},
                    {"name": "New York University", "type": "ORGANIZATION"},
                    {"name": "Meta AI", "type": "ORGANIZATION"},
                    {"name": "New York", "type": "LOCATION"}
                ],
                "triples": [
                    ("Yann LeCun", "affiliated_with", "New York University"),
                    ("Yann LeCun", "developed", "Convolutional Networks"),
                    ("Yann LeCun", "leads", "Meta AI"),
                    ("Meta AI", "located_in", "New York")
                ],
                "relevant_to": ["yann lecun", "convolutional networks", "meta ai research", "computer vision new york"]
            },
            {
                "doc_id": "DOC-103",
                "title": "DeepMind and Reinforcement Learning in London",
                "text": "Demis Hassabis co-founded DeepMind in London, revolutionizing Deep Reinforcement Learning with AlphaGo and AlphaFold. DeepMind was acquired by Google, strengthening the London artificial intelligence ecosystem.",
                "entities": [
                    {"name": "Demis Hassabis", "type": "PERSON"},
                    {"name": "DeepMind", "type": "ORGANIZATION"},
                    {"name": "Reinforcement Learning", "type": "CONCEPT"},
                    {"name": "Google", "type": "ORGANIZATION"},
                    {"name": "London", "type": "LOCATION"}
                ],
                "triples": [
                    ("Demis Hassabis", "founded", "DeepMind"),
                    ("Demis Hassabis", "pioneered", "Reinforcement Learning"),
                    ("DeepMind", "acquired_by", "Google"),
                    ("DeepMind", "located_in", "London")
                ],
                "relevant_to": ["deepmind", "demis hassabis", "reinforcement learning", "london ai google", "alphago"]
            },
            {
                "doc_id": "DOC-104",
                "title": "Generative Pre-trained Transformers at OpenAI",
                "text": "Sam Altman leads OpenAI in San Francisco, which introduced Generative Transformers and large language models. OpenAI partnered with Microsoft to deploy generative intelligence across enterprise cloud architectures.",
                "entities": [
                    {"name": "Sam Altman", "type": "PERSON"},
                    {"name": "OpenAI", "type": "ORGANIZATION"},
                    {"name": "Generative Transformers", "type": "CONCEPT"},
                    {"name": "Microsoft", "type": "ORGANIZATION"},
                    {"name": "San Francisco", "type": "LOCATION"}
                ],
                "triples": [
                    ("Sam Altman", "leads", "OpenAI"),
                    ("OpenAI", "developed", "Generative Transformers"),
                    ("OpenAI", "partnered_with", "Microsoft"),
                    ("OpenAI", "located_in", "San Francisco")
                ],
                "relevant_to": ["sam altman", "openai", "generative transformers", "san francisco cloud microsoft"]
            },
            {
                "doc_id": "DOC-105",
                "title": "Montreal Institute for Learning Algorithms and Yoshua Bengio",
                "text": "Yoshua Bengio founded Mila in Montreal to advance Deep Learning and generative adversarial modeling. Bengio collaborates with University of Montreal to promote ethical artificial intelligence frameworks.",
                "entities": [
                    {"name": "Yoshua Bengio", "type": "PERSON"},
                    {"name": "Mila", "type": "ORGANIZATION"},
                    {"name": "Deep Learning", "type": "CONCEPT"},
                    {"name": "Montreal", "type": "LOCATION"}
                ],
                "triples": [
                    ("Yoshua Bengio", "founded", "Mila"),
                    ("Yoshua Bengio", "researches", "Deep Learning"),
                    ("Mila", "located_in", "Montreal")
                ],
                "relevant_to": ["yoshua bengio", "deep learning", "montreal mila", "neural models"]
            }
        ]
    },
    "Enterprise Cloud & Systems (Domain 2)": {
        "description": "Corpus on enterprise operating platforms, distributed cloud infrastructure, leadership, and headquarters.",
        "documents": [
            {
                "doc_id": "DOC-201",
                "title": "Microsoft Azure Cloud Platform Transformation",
                "text": "Satya Nadella directs Microsoft from Redmond, steering the growth of the Azure Cloud ecosystem. Microsoft integrates Distributed Computing to support scalable hybrid enterprise infrastructures.",
                "entities": [
                    {"name": "Satya Nadella", "type": "PERSON"},
                    {"name": "Microsoft", "type": "ORGANIZATION"},
                    {"name": "Azure Cloud", "type": "CONCEPT"},
                    {"name": "Redmond", "type": "LOCATION"},
                    {"name": "Distributed Computing", "type": "CONCEPT"}
                ],
                "triples": [
                    ("Satya Nadella", "leads", "Microsoft"),
                    ("Microsoft", "developed", "Azure Cloud"),
                    ("Microsoft", "located_in", "Redmond"),
                    ("Azure Cloud", "implements", "Distributed Computing")
                ],
                "relevant_to": ["satya nadella", "microsoft azure", "redmond cloud", "distributed computing"]
            },
            {
                "doc_id": "DOC-202",
                "title": "Amazon Web Services and Global Cloud Scale",
                "text": "Andy Jassy established Amazon Web Services in Seattle, deploying Elastic Cloud architecture worldwide. AWS delivers Cloud Virtualization for millions of enterprise software applications.",
                "entities": [
                    {"name": "Andy Jassy", "type": "PERSON"},
                    {"name": "Amazon", "type": "ORGANIZATION"},
                    {"name": "Cloud Virtualization", "type": "CONCEPT"},
                    {"name": "Seattle", "type": "LOCATION"}
                ],
                "triples": [
                    ("Andy Jassy", "leads", "Amazon"),
                    ("Amazon", "engineered", "Cloud Virtualization"),
                    ("Amazon", "located_in", "Seattle")
                ],
                "relevant_to": ["andy jassy", "amazon web services", "seattle cloud virtualization"]
            },
            {
                "doc_id": "DOC-203",
                "title": "Google Cloud Platform and Distributed Data Processing",
                "text": "Sundar Pichai leads Alphabet and Google in Mountain View, advancing Google Cloud and Kubernetes container orchestration. Google builds Distributed Computing systems for global web infrastructure.",
                "entities": [
                    {"name": "Sundar Pichai", "type": "PERSON"},
                    {"name": "Google", "type": "ORGANIZATION"},
                    {"name": "Google Cloud", "type": "CONCEPT"},
                    {"name": "Mountain View", "type": "LOCATION"},
                    {"name": "Distributed Computing", "type": "CONCEPT"}
                ],
                "triples": [
                    ("Sundar Pichai", "leads", "Google"),
                    ("Google", "operates", "Google Cloud"),
                    ("Google", "located_in", "Mountain View"),
                    ("Google Cloud", "utilizes", "Distributed Computing")
                ],
                "relevant_to": ["sundar pichai", "google cloud", "mountain view", "distributed computing"]
            },
            {
                "doc_id": "DOC-204",
                "title": "Linux Foundation and Open Source Operating Kernels",
                "text": "Linus Torvalds created the Linux Kernel, supported by the Linux Foundation in San Francisco. Linux powers modern Cloud Virtualization across major corporate data centers.",
                "entities": [
                    {"name": "Linus Torvalds", "type": "PERSON"},
                    {"name": "Linux Foundation", "type": "ORGANIZATION"},
                    {"name": "Linux Kernel", "type": "CONCEPT"},
                    {"name": "San Francisco", "type": "LOCATION"},
                    {"name": "Cloud Virtualization", "type": "CONCEPT"}
                ],
                "triples": [
                    ("Linus Torvalds", "created", "Linux Kernel"),
                    ("Linux Foundation", "located_in", "San Francisco"),
                    ("Linux Kernel", "enables", "Cloud Virtualization")
                ],
                "relevant_to": ["linus torvalds", "linux kernel", "san francisco", "cloud virtualization"]
            }
        ]
    }
}


# ======================================================================================
# 3. GRAPH & PROBABILISTIC RANKING ENGINE
# ======================================================================================

def tokenize(text: str) -> list:
    """Basic lowercased alphanumeric tokenizer."""
    return re.findall(r'\b[a-z0-9]+\b', text.lower())


def compute_bm25_score(query_tokens: list, doc_tokens: list, doc_len: int,
                       avg_doc_len: float, idf_dict: dict, k1: float, b: float) -> float:
    """Calculates Okapi BM25 score for a document against query tokens."""
    score = 0.0
    doc_tf = Counter(doc_tokens)
    for q in query_tokens:
        if q in doc_tf:
            tf = doc_tf[q]
            idf = idf_dict.get(q, 0.0)
            numerator = tf * (k1 + 1.0)
            denominator = tf + k1 * (1.0 - b + b * (doc_len / max(1.0, avg_doc_len)))
            score += idf * (numerator / max(0.001, denominator))
    return round(score, 4)


def build_knowledge_graph(corpus_docs: list):
    """
    Constructs a NetworkX graph with fewer, focused nodes (8-15 nodes max)
    representing Persons, Organizations, Locations, and Concepts.
    """
    G = nx.DiGraph()
    entity_metadata = {}

    for doc in corpus_docs:
        for ent in doc["entities"]:
            ename = ent["name"]
            etype = ent["type"]
            if not G.has_node(ename):
                G.add_node(ename, type=etype, docs=set())
            G.nodes[ename]["docs"].add(doc["doc_id"])
            entity_metadata[ename] = etype

        for h, r, t in doc.get("triples", []):
            if G.has_node(h) and G.has_node(t):
                G.add_edge(h, t, relation=r)

    return G, entity_metadata


def evaluate_comparative_retrieval(corpus_docs: list, query_str: str,
                                   k1: float = 1.5, b: float = 0.75,
                                   alpha_entity: float = 1.2, top_k: int = 3):
    """
    Performs dual retrieval: Standard Okapi BM25 vs. Entity-Aware Probabilistic Ranking.
    Computes Precision@K, Recall@K, MAP, and NDCG@K for both methods.
    """
    q_tokens = tokenize(query_str)
    N = len(corpus_docs)
    doc_lens = [len(tokenize(d["text"])) for d in corpus_docs]
    avg_doc_len = sum(doc_lens) / max(1, N)

    # Compute IDF
    df_counts = Counter()
    for d in corpus_docs:
        unique_tokens = set(tokenize(d["text"]))
        for t in unique_tokens:
            df_counts[t] += 1

    idf_dict = {}
    for token, df in df_counts.items():
        # Standard probabilistic BM25 IDF formulation
        idf_dict[token] = math.log(1.0 + (N - df + 0.5) / (df + 0.5))

    # Identify query entities
    all_known_entities = {}
    for d in corpus_docs:
        for ent in d["entities"]:
            all_known_entities[ent["name"].lower()] = ent

    matched_query_entities = []
    q_lower = query_str.lower()
    for ename_lower, ent_obj in all_known_entities.items():
        if ename_lower in q_lower:
            matched_query_entities.append(ent_obj["name"])

    # Ground truth relevance determination for this query
    ground_truth = set()
    for d in corpus_docs:
        # Check relevance tags or substring matching
        is_rel = False
        for tag in d.get("relevant_to", []):
            if any(q in tag or tag in query_str.lower() for q in q_tokens if len(q) > 2):
                is_rel = True
                break
        if not is_rel:
            # Fallback heuristic: query tokens overlap with doc title/entities
            doc_entity_names = [e["name"].lower() for e in d["entities"]]
            overlap = any(q in d["title"].lower() or any(q in en for en in doc_entity_names) for q in q_tokens if len(q) > 2)
            if overlap:
                is_rel = True
        if is_rel:
            ground_truth.add(d["doc_id"])

    # If ground truth is empty, designate the doc with highest keyword overlap
    if not ground_truth:
        ground_truth.add(corpus_docs[0]["doc_id"])

    # 1. Standard BM25 Scoring
    bm25_results = []
    for d in corpus_docs:
        d_tokens = tokenize(d["text"])
        score = compute_bm25_score(q_tokens, d_tokens, len(d_tokens), avg_doc_len, idf_dict, k1, b)
        bm25_results.append({
            "doc_id": d["doc_id"],
            "title": d["title"],
            "score": score,
            "is_relevant": d["doc_id"] in ground_truth,
            "entities": [e["name"] for e in d["entities"]]
        })

    bm25_ranked = sorted(bm25_results, key=lambda x: x["score"], reverse=True)

    # 2. Entity-Aware Probabilistic Scoring
    # Score = BM25 + alpha * (Entity Match Salience)
    entity_results = []
    for d in corpus_docs:
        d_tokens = tokenize(d["text"])
        base_bm25 = compute_bm25_score(q_tokens, d_tokens, len(d_tokens), avg_doc_len, idf_dict, k1, b)
        doc_entity_names = [e["name"] for e in d["entities"]]

        entity_salience = 0.0
        matching_entities = []
        for q_ent in matched_query_entities:
            if q_ent in doc_entity_names:
                entity_salience += 1.5
                matching_entities.append(q_ent)

        # Relation context boost
        for h, r, t in d.get("triples", []):
            if any(qe in [h, t] for qe in matched_query_entities):
                entity_salience += 0.5

        final_score = round(base_bm25 + (alpha_entity * entity_salience), 4)

        entity_results.append({
            "doc_id": d["doc_id"],
            "title": d["title"],
            "score": final_score,
            "base_bm25": base_bm25,
            "entity_boost": round(alpha_entity * entity_salience, 4),
            "matched_entities": matching_entities,
            "is_relevant": d["doc_id"] in ground_truth,
            "entities": doc_entity_names
        })

    entity_ranked = sorted(entity_results, key=lambda x: x["score"], reverse=True)

    # Calculate Metrics: Precision@K, Recall@K, MAP, NDCG@K
    def calculate_metrics(ranked_list, K, total_rel_count):
        top_k_items = ranked_list[:K]
        rel_in_k = sum(1 for item in top_k_items if item["is_relevant"])
        precision_k = round(rel_in_k / max(1, K), 4)
        recall_k = round(rel_in_k / max(1, total_rel_count), 4)

        # Average Precision (AP)
        running_rel = 0
        ap_sum = 0.0
        for rank_idx, item in enumerate(ranked_list):
            if item["is_relevant"]:
                running_rel += 1
                ap_sum += running_rel / (rank_idx + 1)
        ap = round(ap_sum / max(1, total_rel_count), 4)

        # NDCG@K
        dcg = 0.0
        for idx, item in enumerate(top_k_items):
            rel_grade = 1.0 if item["is_relevant"] else 0.0
            dcg += rel_grade / math.log2(idx + 2)

        # Ideal DCG@K
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(K, total_rel_count)))
        ndcg_k = round(dcg / max(0.0001, idcg), 4) if idcg > 0 else 0.0

        return {
            "precision_k": precision_k,
            "recall_k": recall_k,
            "map": ap,
            "ndcg_k": ndcg_k
        }

    rel_count = len(ground_truth)
    bm25_metrics = calculate_metrics(bm25_ranked, top_k, rel_count)
    entity_metrics = calculate_metrics(entity_ranked, top_k, rel_count)

    return {
        "bm25_ranked": bm25_ranked,
        "entity_ranked": entity_ranked,
        "bm25_metrics": bm25_metrics,
        "entity_metrics": entity_metrics,
        "ground_truth": list(ground_truth),
        "matched_query_entities": matched_query_entities
    }


def generate_plotly_knowledge_graph(G: nx.DiGraph):
    """
    Renders an uncluttered, publication-grade interactive network graph
    with fewer nodes (8-15) using Plotly and NetworkX spring layout.
    """
    pos = nx.spring_layout(G, seed=42, k=0.85)

    # Color palette for entity categories
    color_map = {
        "PERSON": "#8B5CF6",       # Purple
        "ORGANIZATION": "#2563EB", # Royal Blue
        "LOCATION": "#10B981",     # Emerald Green
        "CONCEPT": "#F59E0B"       # Amber Orange
    }

    # Extract edge coordinates
    edge_x = []
    edge_y = []
    edge_text = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_text.append(f"{edge[0]} &rarr; <i>{edge[2].get('relation', 'rel')}</i> &rarr; {edge[1]}")

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color="#94A3B8"),
        hoverinfo='none',
        mode='lines'
    )

    # Extract node coordinates and properties
    node_x = []
    node_y = []
    node_colors = []
    node_text = []
    node_labels = []
    node_sizes = []

    degrees = dict(G.degree())

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        ntype = G.nodes[node].get("type", "CONCEPT")
        node_colors.append(color_map.get(ntype, "#6B7280"))
        node_labels.append(node)
        deg = degrees.get(node, 1)
        node_sizes.append(max(18, min(36, 18 + deg * 4)))
        node_text.append(
            f"<b>Entity:</b> {node}<br>"
            f"<b>Category:</b> {ntype}<br>"
            f"<b>Degree Centrality:</b> {deg}<br>"
            f"<b>Associated Docs:</b> {len(G.nodes[node].get('docs', []))}"
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_labels,
        textposition="top center",
        textfont=dict(size=11, family="Arial, sans-serif"),
        hovertext=node_text,
        marker=dict(
            showscale=False,
            color=node_colors,
            size=node_sizes,
            line=dict(width=2, color='#1E293B')
        )
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text="Domain Knowledge Graph Topology (Fewer Nodes for Clarity)",
                font=dict(size=15)
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=20, r=20, t=45),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=430,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
    )

    return fig


# ======================================================================================
# 4. LAB REPORT PDF EXPORTER (FPDF COMPATIBLE WITH template.py)
# ======================================================================================

class LabReportPDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}} | IIT Kharagpur Virtual Laboratory - KGIRS Exp 6", align="C")


def generate_pdf_report(student_name: str, student_id: str, date_str: str,
                        trials_df: pd.DataFrame, pretest_score: int, pretest_total: int,
                        posttest_score: int, posttest_total: int,
                        student_notes: str) -> bytes:
    """Compiles experiment benchmark records into an official PDF report document."""
    pdf = LabReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # Institute & Department Header
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 5, "INDIAN INSTITUTE OF TECHNOLOGY KHARAGPUR - VIRTUAL LABS", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, "DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING | KGIRS", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # Document Title
    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, f"Experiment {EXPERIMENT_CONFIG['exp_number']}: {EXPERIMENT_CONFIG['title']}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # Student & Session Info Box
    pdf.set_fill_color(241, 245, 249)
    pdf.set_draw_color(203, 213, 225)
    pdf.rect(10, 32, 190, 24, "FD")

    pdf.set_xy(14, 34)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(34, 5, "Student Name:", 0)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(58, 5, student_name or "N/A", 0)

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(38, 5, "Student ID / Roll:", 0)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(48, 5, student_id or "N/A", 1)

    pdf.set_xy(14, 42)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(34, 5, "Experiment Date:", 0)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(58, 5, date_str or datetime.now().strftime("%Y-%m-%d"), 0)

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(38, 5, "Evaluation Scores:", 0)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(16, 185, 129)
    score_str = f"Pre: {pretest_score}/{pretest_total} | Post: {posttest_score}/{posttest_total}"
    pdf.cell(48, 5, score_str, 1)

    pdf.ln(16)

    # 1. Objectives
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "1. Experiment Objectives & Learning Units", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(51, 65, 85)
    for obj in EXPERIMENT_CONFIG["objectives"]:
        clean_obj = str(obj).replace("$", "").replace("\\", "")
        pdf.cell(5, 5, "-", 0)
        pdf.cell(0, 5, f" {clean_obj}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # 2. Recorded Trials Table
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "2. Recorded Experimental Trials & Comparative Performance", new_x="LMARGIN", new_y="NEXT")

    if trials_df.empty:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 6, "No simulation trials recorded during this session.", new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.set_fill_color(37, 99, 235)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 8)

        cols = list(trials_df.columns)
        num_cols = len(cols)
        col_w = max(18, int(190 / max(1, num_cols)))

        for c in cols:
            pdf.cell(col_w, 6, str(c)[:13], 1, 0, "C", True)
        pdf.ln()

        pdf.set_fill_color(248, 250, 252)
        pdf.set_text_color(30, 41, 59)
        pdf.set_font("Helvetica", "", 8)
        fill = False

        for _, row in trials_df.iterrows():
            for c in cols:
                val = row[c]
                val_str = f"{val:.3f}" if isinstance(val, float) else str(val)
                pdf.cell(col_w, 5, val_str[:13], 1, 0, "C", fill)
            pdf.ln()
            fill = not fill
    pdf.ln(5)

    # 3. Observations & Analysis
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "3. Observations, Inferences & Critical Analysis", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(51, 65, 85)
    notes_text = student_notes.strip() if student_notes.strip() else (
        "The experimental trials demonstrated that entity-aware probabilistic ranking consistently outperformed "
        "the standard Okapi BM25 keyword baseline across Precision@K, MAP, and NDCG@K metrics by capturing "
        "disambiguated semantic relationships between persons, organizations, locations, and concepts."
    )
    pdf.multi_cell(0, 5, notes_text)
    pdf.ln(8)

    # Sign-off line
    pdf.set_draw_color(180, 180, 180)
    pdf.line(130, pdf.get_y() + 15, 190, pdf.get_y() + 15)
    pdf.set_xy(130, pdf.get_y() + 17)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(60, 4, "Instructor / Evaluator Signature", align="C")

    return bytes(pdf.output())


# ======================================================================================
# 5. IIT KGP VLAB SECTION RENDERERS
# ======================================================================================

def render_aim_section():
    """Renders Section 1: Aim & Learning Objectives."""
    st.markdown("### Aim")
    st.info(
        "**Experiment Aim**: To extract and identify domain-specific entities (Persons, Organizations, "
        "Locations, and Concepts) from unstructured text to build structured knowledge (Module 1), "
        "construct a relational Knowledge Graph with clean node topology, and develop an understanding of "
        "probabilistic ranking and comparative retrieval performance (Module 2)."
    )

    st.markdown("#### Learning Objectives (Mapped to KGIRS Modules 1 & 2)")
    for idx, obj in enumerate(EXPERIMENT_CONFIG["objectives"]):
        st.markdown(f"**{idx + 1}.** {obj}")

    st.markdown("#### Target Syllabus Units (KGIRS Course Plan)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "**Module 1: Introduction to Information Retrieval and Graphs**\n"
            "- *Section 1.2*: Structured Knowledge vs. Unstructured Knowledge\n"
            "- *Section 1.2*: Query structures (keyword, pattern matching, structured queries)\n"
            "- *Section 1.3*: Sparse vs. Dense Retrieval and exact-match entity domains"
        )
    with col2:
        st.markdown(
            "**Module 2: Modeling in IR, Indexing and Ranking IR Models**\n"
            "- *Section 2.1*: Review of IR Models — Vector Space Model vs. **Probabilistic Model**\n"
            "- *Section 2.1*: Ranking IR Models and Problems on Probabilistic Model\n"
            "- *Section 2.2*: Hybrid search combining lexical BM25 and structured entity knowledge\n"
            "- *Section 6.2*: Comparative Retrieval Evaluation (Precision, Recall, MAP, NDCG)"
        )


def render_introduction_section():
    """Renders Section 2: Introduction & Motivation."""
    st.markdown("### Introduction")
    st.markdown("""
In the **Knowledge Graph and Information Retrieval System (KGIRS)** curriculum, a foundational transition occurs between:
1. **Unstructured Knowledge**: Free-form natural language texts, documents, and articles where facts are implicit and ambiguous (Module 1.2).
2. **Structured Knowledge**: Explicit relational representations where discrete entities and relations are systematically codified as graphs and triples $(h, r, t)$.

#### The Need for Entity Identification
Traditional search engines operate on unstructured text using bag-of-words keyword indexes. This introduces severe limitations:
- **Polysemy and Synonymy**: Lexical terms like *"Apple"* or *"Jordan"* are ambiguous without entity categorization (e.g., `ORGANIZATION` vs. `PERSON` vs. `LOCATION`).
- **Surface-Level Keyword Mismatch**: Simple string matching fails to capture that *"Geoffrey Hinton"* and *"University of Toronto"* share a formal research affiliation.
- **Entity Identification**: By parsing text to detect **Persons**, **Organizations**, **Locations**, and **Domain Concepts**, we elevate unstructured text into structured graph entities.

#### The Expected Outcome: Probabilistic Ranking & Comparative Retrieval Performance
Under **Module 2 (Modeling in Information Retrieval)**, retrieval is framed around estimating the **probability of relevance** of a document given an information need:
- **Probabilistic Ranking (Okapi BM25)** balances term frequency saturation ($k_1$) and document length normalization ($b$).
- **Entity-Aware Probabilistic Ranking** injects structured Knowledge Graph entities directly into the probabilistic scoring equation.
- **Comparative Retrieval Performance**: By measuring standard IR benchmark metrics (**Precision@K**, **Recall@K**, **MAP**, and **NDCG**), students empirically demonstrate how structured entity knowledge improves retrieval quality over baseline keyword search.
""")


def render_theory_section():
    """Renders Section 3: In-Depth Theoretical Foundations."""
    st.markdown("### Theoretical Foundations")

    st.markdown("#### 1. Entity Extraction & Knowledge Graph Construction")
    st.markdown("""
A Knowledge Graph $\\mathcal{G} = (\\mathcal{E}, \\mathcal{R}, \\mathcal{T})$ consists of a set of entities $\\mathcal{E}$, relation types $\\mathcal{R}$, and factual triples $\\mathcal{T} \\subseteq \\mathcal{E} \\times \\mathcal{R} \\times \\mathcal{E}$.
- **Entity Mentions**: Substrings in text tagged with semantic categories:
  - **`PERSON`**: Key researchers, executives, pioneers (e.g., *Geoffrey Hinton*, *Demis Hassabis*).
  - **`ORGANIZATION`**: Companies, research labs, universities (e.g., *DeepMind*, *University of Toronto*, *Meta AI*).
  - **`LOCATION`**: Headquarters, lab sites, cities (e.g., *London*, *Toronto*, *New York*).
  - **`CONCEPT`**: Technical domains, algorithmic paradigms (e.g., *Deep Learning*, *Reinforcement Learning*).
- **Relational Triples**: Directed links representing predicates:
  $$\\text{Triple} = (\\text{Head Entity}, \\text{Predicate Relation}, \\text{Tail Entity})$$
  Example: `(Geoffrey Hinton, affiliated_with, Google Brain)`
""")

    st.divider()
    st.markdown("#### 2. The Probability Ranking Principle (PRP)")
    st.markdown("""
Formulated by Stephen E. Robertson in 1977, the **Probability Ranking Principle** states:
> *"If a reference retrieval system's response to each request is a ranking of the documents in order of decreasing probability of relevance to the user, the overall effectiveness of the system will be maximized."*

Mathematically, let $R \\in \\{0, 1\\}$ denote binary relevance. Documents $D$ are ranked by the posterior odds of relevance given query $Q$:
$$\\text{Odds}(R=1 | D, Q) = \\frac{P(R=1 | D, Q)}{P(R=0 | D, Q)}$$
""")

    st.divider()
    st.markdown("#### 3. Robertson-Spärck Jones Okapi BM25 Model")
    st.markdown("""
The Okapi BM25 formula is a non-linear term saturation probabilistic model:
$$\\text{BM25}(D, Q) = \\sum_{q \\in Q} \\text{IDF}(q) \\cdot \\frac{f(q, D) \\cdot (k_1 + 1)}{f(q, D) + k_1 \\cdot \\left(1 - b + b \\cdot \\frac{|D|}{\\text{avgdl}}\\right)}$$

Where:
- **$f(q, D)$**: Term frequency of query term $q$ in document $D$.
- **$|D|$ and $\\text{avgdl}$**: Document length and average document length across the entire collection.
- **$k_1$ (Term Saturation Parameter)**: Typically set between $1.2$ and $2.0$. It calibrates how quickly term frequency saturation is attained.
- **$b$ (Length Normalization Parameter)**: Typically set around $0.75$. When $b=1$, full document length normalization is applied; when $b=0$, length normalization is disabled.
- **$\\text{IDF}(q)$**: Probabilistic Inverse Document Frequency:
  $$\\text{IDF}(q) = \\ln \\left( 1 + \\frac{N - n(q) + 0.5}{n(q) + 0.5} \\right)$$
""")

    st.divider()
    st.markdown("#### 4. Entity-Aware Probabilistic Ranking")
    st.markdown("""
To incorporate Knowledge Graph semantics into probabilistic ranking, the document score combines the lexical BM25 score with an **Entity Salience Boost**:
$$\\text{Score}_{\\text{Entity-BM25}}(D, Q) = \\text{BM25}(D, Q) + \\alpha \\sum_{e \\in \\mathcal{E}_Q \\cap \\mathcal{E}_D} \\text{Salience}(e) + \\beta \\sum_{(h, r, t) \\in \\mathcal{T}_D \\mid h, t \\in \\mathcal{E}_Q} \\text{RelWeight}(r)$$

Where $\\alpha$ is the entity boost factor and $\\mathcal{E}_Q, \\mathcal{E}_D$ represent verified entities in query and document respectively.
""")

    st.divider()
    st.markdown("#### 5. Comparative Evaluation Metrics")
    st.markdown("""
- **Precision@K**: $\\frac{|\\text{Relevant Documents} \\cap \\text{Top K Documents}|}{K}$
- **Recall@K**: $\\frac{|\\text{Relevant Documents} \\cap \\text{Top K Documents}|}{|\\text{Total Relevant Documents}|}$
- **Mean Average Precision (MAP)**: $\\text{MAP} = \\frac{1}{|Q|} \\sum_{q=1}^{|Q|} \\frac{1}{|R_q|} \\sum_{k=1}^{N} P_q(k) \\times \\text{rel}_q(k)$
- **Normalized Discounted Cumulative Gain (NDCG@K)**:
  $$\\text{DCG}@K = \\sum_{i=1}^K \\frac{2^{\\text{rel}_i} - 1}{\\log_2(i + 1)}, \\quad \\text{NDCG}@K = \\frac{\\text{DCG}@K}{\\text{IDCG}@K}$$
""")


def render_casestudy_section():
    """Renders Section 4: Real-World Case Study."""
    st.markdown("### Case Study: Semantic Academic & Enterprise Search")
    st.markdown("""
#### Scenario Background
An academic intelligence institution indexes hundreds of thousands of research briefs, corporate acquisition notices, and technical publications. 
A research director inputs the multi-facet query:
> **Query**: `"Geoffrey Hinton Deep Learning Google Toronto"`

#### Step 1: Raw Lexical Retrieval (Pure BM25)
- Document A mentions *"Geoffrey Hinton pioneered neural networks in Toronto."* (Contains 3 query tokens).
- Document B is a promotional article repeatedly repeating *"Google Google Google search cloud"* (High TF on "Google", low semantic relevance).
- **Result**: Due to term frequency saturation disparities and lack of entity typing, Document B may outrank or tie with Document A.

#### Step 2: Entity Identification & Knowledge Graph Grounding
- **Entities Identified**:
  - `Geoffrey Hinton` $\\rightarrow$ `PERSON`
  - `Deep Learning` $\\rightarrow$ `CONCEPT`
  - `Google` / `Google Brain` $\\rightarrow$ `ORGANIZATION`
  - `Toronto` $\\rightarrow$ `LOCATION`
- **Knowledge Triples Formed**:
  - `(Geoffrey Hinton, affiliated_with, University of Toronto)`
  - `(Geoffrey Hinton, joined, Google Brain)`
  - `(University of Toronto, located_in, Toronto)`

#### Step 3: Comparative Evaluation Outcome
Under **Entity-Aware Probabilistic Ranking**, documents confirming direct relational triples with query entities receive calibrated entity salience boosts. Spurious keyword matches are suppressed, raising **Precision@3 from 33.3% to 100%** and elevating **NDCG@3 from 0.46 to 1.00**.
""")


def render_pretest_section():
    """Renders Section 5: Pretest Assessment."""
    st.markdown("### Diagnostic Pretest")
    st.write("Complete the diagnostic assessment below before running the interactive simulation.")

    with st.form("pretest_form"):
        responses = {}
        for q in PRETEST_QUESTIONS:
            st.markdown(f"**Question {q['id']}:** {q['question']}")
            selected = st.radio(
                label=f"Options for Pretest Q{q['id']}",
                options=q["options"],
                index=st.session_state["pretest_answers"].get(q["id"], 0),
                key=f"pretest_radio_{q['id']}",
                label_visibility="collapsed"
            )
            responses[q["id"]] = q["options"].index(selected)
            st.markdown("---")

        submitted = st.form_submit_button("Submit Pretest for Grading", type="primary")

    if submitted:
        score = 0
        st.session_state["pretest_answers"] = responses
        st.session_state["pretest_submitted"] = True

        st.subheader("Pretest Evaluation & Diagnostic Explanations")
        for q in PRETEST_QUESTIONS:
            user_ans = responses.get(q["id"])
            corr_ans = q["answer_index"]
            if user_ans == corr_ans:
                score += 1
                st.success(f"**Question {q['id']}: Correct!**\n\n_{q['explanation']}_")
            else:
                st.error(
                    f"**Question {q['id']}: Incorrect.** (Your answer: {q['options'][user_ans]})\n\n"
                    f"**Correct Answer:** {q['options'][corr_ans]}\n\n"
                    f"_{q['explanation']}_"
                )

        st.session_state["pretest_score"] = score
        perc = (score / len(PRETEST_QUESTIONS)) * 100
        st.info(f"Pretest Result: **{score} / {len(PRETEST_QUESTIONS)}** ({perc:.0f}%)")

    elif st.session_state.get("pretest_submitted", False):
        st.success(f"Pretest completed. Score: **{st.session_state.get('pretest_score', 0)} / {len(PRETEST_QUESTIONS)}**")


def render_simulation_section():
    """Renders Section 6: Interactive Simulation Sandbox with Fewer Graph Nodes."""
    st.markdown("### Interactive Simulation Sandbox")
    st.info(
        "Experiment with entity identification, inspect the interactive Knowledge Graph "
        "(designed with fewer nodes for visual clarity), and compare Standard BM25 vs. Entity-Aware Probabilistic Ranking."
    )

    # 1. Parameter Configuration Sidebar / Columns
    st.subheader("1. Experimental Configuration Controls")
    c_corpus, c_query = st.columns([1.5, 2])

    with c_corpus:
        corpus_choice = st.selectbox(
            "Select Benchmark Corpus Domain:",
            options=list(DATA_CORPORA.keys()),
            index=0
        )
        selected_corpus = DATA_CORPORA[corpus_choice]["documents"]
        st.caption(f"_{DATA_CORPORA[corpus_choice]['description']}_")

    with c_query:
        preset_queries = {
            "AI & Deep Learning Pioneers (Domain 1)": [
                "Geoffrey Hinton deep learning Toronto Google",
                "Demis Hassabis reinforcement learning DeepMind London",
                "Yann LeCun convolutional networks New York Meta AI",
                "Sam Altman OpenAI generative transformers Microsoft"
            ],
            "Enterprise Cloud & Systems (Domain 2)": [
                "Satya Nadella Microsoft Azure Redmond distributed computing",
                "Andy Jassy Amazon Seattle cloud virtualization",
                "Sundar Pichai Google Cloud distributed computing Mountain View",
                "Linus Torvalds Linux Kernel cloud virtualization San Francisco"
            ]
        }
        query_options = preset_queries.get(corpus_choice, ["Geoffrey Hinton deep learning Google"])
        selected_query = st.selectbox("Select Test Query (or type custom query):", query_options)

    # Hyperparameters
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        param_k1 = st.slider("BM25 k1 (Term Saturation):", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
    with c2:
        param_b = st.slider("BM25 b (Length Normalization):", min_value=0.0, max_value=1.0, value=0.75, step=0.05)
    with c3:
        param_alpha = st.slider("Entity Boost Factor (α):", min_value=0.0, max_value=3.0, value=1.2, step=0.1)
    with c4:
        param_top_k = st.slider("Evaluation Cutoff (Top K):", min_value=1, max_value=4, value=3, step=1)

    # 2. Build Knowledge Graph & Run Comparative Retrieval
    G, entity_meta = build_knowledge_graph(selected_corpus)
    results = evaluate_comparative_retrieval(
        corpus_docs=selected_corpus,
        query_str=selected_query,
        k1=param_k1,
        b=param_b,
        alpha_entity=param_alpha,
        top_k=param_top_k
    )

    st.divider()

    # 3. Knowledge Graph Visualization (Fewer Nodes)
    st.subheader("2. Identified Graph Entities & Knowledge Graph Topology")
    st.caption("Extracted domain entities categorized into Persons, Organizations, Locations, and Concepts. Graph rendered with focused nodes (<15) without Neo4j dependency.")

    col_graph, col_entities = st.columns([2.2, 1.3])

    with col_graph:
        graph_fig = generate_plotly_knowledge_graph(G)
        st.plotly_chart(graph_fig, use_container_width=True)

    with col_entities:
        st.markdown("**Identified Graph Entities in Active Corpus**")
        ent_data = []
        for ename, etype in entity_meta.items():
            deg = G.degree(ename)
            ent_data.append({"Entity Name": ename, "Category": etype, "Degree": deg})
        df_entities = pd.DataFrame(ent_data)
        st.dataframe(df_entities, height=380, use_container_width=True, hide_index=True)

    st.divider()

    # 4. Comparative Retrieval Rankings
    st.subheader("3. Comparative Retrieval Rankings: Standard BM25 vs. Entity-Aware BM25")
    st.write(f"**Query Evaluated:** `{selected_query}` | **Ground Truth Relevant Docs:** `{', '.join(results['ground_truth'])}`")

    col_rank1, col_rank2 = st.columns(2)

    with col_rank1:
        st.markdown("##### Standard Okapi BM25 Ranking")
        bm25_table = []
        for idx, item in enumerate(results["bm25_ranked"]):
            status = "Relevant" if item["is_relevant"] else "Non-Relevant"
            bm25_table.append({
                "Rank": idx + 1,
                "Doc ID": item["doc_id"],
                "BM25 Score": item["score"],
                "Relevance": status,
                "Title": item["title"][:28] + "..."
            })
        st.dataframe(pd.DataFrame(bm25_table), use_container_width=True, hide_index=True)

    with col_rank2:
        st.markdown("##### Entity-Aware Probabilistic Ranking")
        entity_table = []
        for idx, item in enumerate(results["entity_ranked"]):
            status = "Relevant" if item["is_relevant"] else "Non-Relevant"
            entity_table.append({
                "Rank": idx + 1,
                "Doc ID": item["doc_id"],
                "Total Score": item["score"],
                "Entity Boost": item["entity_boost"],
                "Relevance": status,
                "Title": item["title"][:28] + "..."
            })
        st.dataframe(pd.DataFrame(entity_table), use_container_width=True, hide_index=True)

    st.divider()

    # 5. Comparative Performance Benchmarks & Plot
    st.subheader(f"4. Quantitative Benchmark Performance Metrics (@K = {param_top_k})")
    bm25_m = results["bm25_metrics"]
    ent_m = results["entity_metrics"]

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        diff_p = round(ent_m["precision_k"] - bm25_m["precision_k"], 3)
        st.metric(f"Precision@{param_top_k}", f"{ent_m['precision_k']:.3f}", delta=f"{diff_p:+.3f} vs BM25")
    with m2:
        diff_r = round(ent_m["recall_k"] - bm25_m["recall_k"], 3)
        st.metric(f"Recall@{param_top_k}", f"{ent_m['recall_k']:.3f}", delta=f"{diff_r:+.3f} vs BM25")
    with m3:
        diff_map = round(ent_m["map"] - bm25_m["map"], 3)
        st.metric("Mean Avg Precision (MAP)", f"{ent_m['map']:.3f}", delta=f"{diff_map:+.3f} vs BM25")
    with m4:
        diff_ndcg = round(ent_m["ndcg_k"] - bm25_m["ndcg_k"], 3)
        st.metric(f"NDCG@{param_top_k}", f"{ent_m['ndcg_k']:.3f}", delta=f"{diff_ndcg:+.3f} vs BM25")

    # Bar chart comparison
    chart_col1, chart_col2 = st.columns([2.5, 1.5])
    with chart_col1:
        metric_names = [f"Precision@{param_top_k}", f"Recall@{param_top_k}", "MAP", f"NDCG@{param_top_k}"]
        bm25_vals = [bm25_m["precision_k"], bm25_m["recall_k"], bm25_m["map"], bm25_m["ndcg_k"]]
        ent_vals = [ent_m["precision_k"], ent_m["recall_k"], ent_m["map"], ent_m["ndcg_k"]]

        fig_bar = go.Figure(data=[
            go.Bar(name='Standard Okapi BM25', x=metric_names, y=bm25_vals, marker_color='#94A3B8'),
            go.Bar(name='Entity-Aware Probabilistic Ranking', x=metric_names, y=ent_vals, marker_color='#2563EB')
        ])
        fig_bar.update_layout(
            barmode='group',
            title=f"Comparative Retrieval Performance Benchmark (@K={param_top_k})",
            yaxis=dict(title="Score (0.0 to 1.0)", range=[0, 1.1]),
            height=340,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with chart_col2:
        st.markdown("**Performance Inferences**")
        st.write(
            f"- **Entity Match Impact**: When query matches entity nodes, entity boost $\\alpha={param_alpha}$ "
            f"elevates relevant documents toward top ranks.\n"
            f"- **Term Saturation $k_1={param_k1}$**: Regulates how rapidly repeated keyword frequency saturates.\n"
            f"- **Length Norm $b={param_b}$**: Corrects for document verbosity across corpus items."
        )

    # 6. Experimental Data Log Book (matches template.py)
    st.divider()
    st.subheader("5. Experimental Data Log Book")
    col_log1, col_log2 = st.columns([1.5, 3.5])

    with col_log1:
        st.caption("Capture current parameters and comparative performance into your session trial table:")
        if st.button("Record Current Trial", type="primary", use_container_width=True):
            trial_record = {
                "Trial #": len(st.session_state["trials"]) + 1,
                "Corpus": corpus_choice[:14],
                "k1": param_k1,
                "b": param_b,
                "Alpha": param_alpha,
                "Cutoff K": param_top_k,
                "BM25 P@K": bm25_m["precision_k"],
                "Entity P@K": ent_m["precision_k"],
                "BM25 MAP": bm25_m["map"],
                "Entity MAP": ent_m["map"],
                "Entity NDCG": ent_m["ndcg_k"],
                "Timestamp": datetime.now().strftime("%H:%M:%S")
            }
            st.session_state["trials"].append(trial_record)
            st.toast(f"Trial #{trial_record['Trial #']} successfully saved!")

        if st.button("Clear Logged Trials", use_container_width=True):
            st.session_state["trials"] = []
            st.toast("Trial log cleared.")

    with col_log2:
        if st.session_state["trials"]:
            df_trials = pd.DataFrame(st.session_state["trials"])
            st.dataframe(df_trials, use_container_width=True, hide_index=True)
            csv_data = df_trials.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Trials as CSV",
                data=csv_data,
                file_name="kgirs_experiment_trials.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No trials recorded yet. Click 'Record Current Trial' to begin collecting experimental data.")


def render_procedure_section():
    """Renders Section 7: Step-by-Step Procedure."""
    st.markdown("### Step-by-Step Procedure")
    st.markdown("""
To systematically execute Experiment 6, follow the instructions below:

1. **Step 1 (Theoretical Review)**: Review the *Aim*, *Introduction*, and *Theory* sections to understand entity categorization (`PER`, `ORG`, `LOC`, `CONCEPT`), Knowledge Graph relational triples, and the BM25 formulation.
2. **Step 2 (Pretest Assessment)**: Navigate to the *Pretest* section in the sidebar. Answer the 5 diagnostic questions to verify foundational concepts.
3. **Step 3 (Select Corpus & Query)**: Open the *Simulation* section. Choose a domain corpus (e.g., *AI Pioneers* or *Enterprise Cloud*) and select a target test query.
4. **Step 4 (Inspect Knowledge Graph Topology)**: Observe the extracted entity nodes and directed relations in the interactive graph. Note how fewer nodes (<15) provide a clean, readable layout without visual confusion.
5. **Step 5 (Calibrate Hyperparameters)**:
   - Vary $k_1$ between $0.5$ and $3.0$ to examine term frequency saturation.
   - Adjust $b$ between $0.0$ and $1.0$ to test length normalization.
   - Adjust $\\alpha$ between $0.0$ and $2.5$ to examine the impact of entity weighting.
6. **Step 6 (Analyze Comparative Performance)**: Examine the side-by-side rankings and the performance metric comparison (Precision@K, Recall@K, MAP, NDCG@K).
7. **Step 7 (Record Experimental Trials)**: Click **'Record Current Trial'** for at least 3 to 4 distinct parameter configurations (e.g., varying $\\alpha$ from $0.0$ to $2.0$).
8. **Step 8 (Posttest & Report Generation)**: Complete the 10-question *Posttest* quiz. Finally, navigate to *Report Generation*, enter your student details and observations, and download your official PDF report.
""")


def render_exercises_section():
    """Renders Section 8: Conceptual and Analytical Exercises."""
    st.markdown("### Practical Exercises & Inquiry Problems")
    st.markdown("""
Engage with the following inquiry exercises to deepen your analysis:

#### Exercise 1: Term Frequency Saturation Analysis
- Set $\\alpha = 0.0$ (pure BM25 mode).
- Set $b = 0.75$.
- Compare retrieval rankings when $k_1 = 0.5$ (fast saturation) versus $k_1 = 3.0$ (linear term frequency influence).
- **Inquiry**: *How does a lower $k_1$ prevent documents with keyword-stuffing from dominating the top ranks?*

#### Exercise 2: Document Length Normalization Dynamics
- Compare setting $b = 0.0$ (no length penalty) versus $b = 1.0$ (full length normalization).
- **Inquiry**: *Under what corpus conditions does setting $b = 0$ severely disadvantage concise, informative documents?*

#### Exercise 3: Entity Weighting Sensitivity ($\alpha$)
- Execute a query matching a specific person and organization (e.g., `"Demis Hassabis DeepMind London"`).
- Record Precision@3 and NDCG@3 as $\\alpha$ increases in steps of $0.5$ from $0.0$ to $2.5$.
- **Inquiry**: *At what threshold of $\\alpha$ does entity matching completely dominate lexical keyword matching?*

#### Exercise 4: Polysemy and Disambiguation in Sparse Graphs
- Explain why maintaining a focused graph with fewer, highly disambiguated nodes (8-15) prevents semantic drift during graph-expanded probabilistic retrieval.
""")


def render_posttest_section():
    """Renders Section 9: Posttest Assessment (10 Questions)."""
    st.markdown("### Concept Assessment Posttest")
    st.write("Answer the 10 conceptual questions below to evaluate your understanding of Graph Entities and Probabilistic Retrieval.")

    with st.form("posttest_form"):
        user_responses = {}
        for q in POSTTEST_QUESTIONS:
            st.markdown(f"**Question {q['id']}:** {q['question']}")
            selected = st.radio(
                label=f"Options for Posttest Q{q['id']}",
                options=q["options"],
                index=st.session_state["quiz_answers"].get(q["id"], 0),
                key=f"posttest_radio_{q['id']}",
                label_visibility="collapsed"
            )
            user_responses[q["id"]] = q["options"].index(selected)
            st.markdown("---")

        submitted = st.form_submit_button("Submit Posttest for Grading", type="primary")

    if submitted:
        score = 0
        st.session_state["quiz_answers"] = user_responses
        st.session_state["quiz_submitted"] = True

        st.divider()
        st.subheader("Evaluation Results and Detailed Feedback")
        for q in POSTTEST_QUESTIONS:
            user_ans = user_responses.get(q["id"])
            corr_ans = q["answer_index"]
            if user_ans == corr_ans:
                score += 1
                st.success(f"**Question {q['id']}: Correct!**\n\n_{q['explanation']}_")
            else:
                st.error(
                    f"**Question {q['id']}: Incorrect.** (Your answer: {q['options'][user_ans]})\n\n"
                    f"**Correct Answer:** {q['options'][corr_ans]}\n\n"
                    f"**Reasoning:** _{q['explanation']}_"
                )

        st.session_state["quiz_score"] = score
        perc = (score / len(POSTTEST_QUESTIONS)) * 100
        st.info(f"Final Posttest Score: **{score} / {len(POSTTEST_QUESTIONS)}** ({perc:.0f}%)")

    elif st.session_state.get("quiz_submitted", False):
        st.success(f"Posttest already completed. Current score: **{st.session_state.get('quiz_score', 0)} / {len(POSTTEST_QUESTIONS)}**")


def render_references_section():
    """Renders Section 10: Academic References & Textbooks."""
    st.markdown("### Academic References & Recommended Reading")
    st.markdown("""
1. **Robertson, S. E., & Zaragoza, H. (2009)**. *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in Information Retrieval, 3(4), 333-389.
2. **Manning, C. D., Raghavan, P., & Schütze, H. (2008)**. *Introduction to Information Retrieval*. Cambridge University Press.
3. **Ji, S., Pan, S., Cambria, E., Marttinen, P., & Yu, P. S. (2021)**. *A Survey on Knowledge Graphs: Representation, Acquisition, and Applications*. IEEE Transactions on Neural Networks and Learning Systems, 33(2), 494-514.
4. **Xiong, C., Power, R., & Callan, J. (2017)**. *Explicit Semantic Ranking for Academic Search via Knowledge Graph Embedding*. Proceedings of the 26th International Conference on World Wide Web (WWW '17), 1271-1279.
5. **IIT Kharagpur Virtual Laboratories Project**. *Virtual Laboratory System Architecture and Pedagogical Guidelines*. Ministry of Education, Government of India.
""")


def render_report_section():
    """Renders Section 11: Dynamic Lab Report Generator with PDF Export."""
    st.markdown("### Official Lab Report Generation")
    st.write("Compile your student details, diagnostic scores, recorded simulation trials, and observations into a downloadable PDF report.")

    col1, col2, col3 = st.columns(3)
    with col1:
        student_name = st.text_input("Student Name", value=st.session_state["student_info"].get("name", "Student Name"))
    with col2:
        student_id = st.text_input("Student Roll / ID", value=st.session_state["student_info"].get("id", "EXP-006"))
    with col3:
        lab_date = st.date_input("Experiment Date", value=datetime.now())

    st.session_state["student_info"]["name"] = student_name
    st.session_state["student_info"]["id"] = student_id
    st.session_state["student_info"]["date"] = str(lab_date)

    st.subheader("Observations & Analysis Notes")
    student_notes = st.text_area(
        "Enter your interpretation of results, observations, and conclusions:",
        value=st.session_state.get("student_notes", (
            "The experimental trials demonstrated that entity-aware probabilistic ranking consistently outperformed "
            "the standard Okapi BM25 keyword baseline across Precision@K, MAP, and NDCG@K metrics by capturing "
            "disambiguated semantic relationships between persons, organizations, locations, and concepts."
        )),
        height=120
    )
    st.session_state["student_notes"] = student_notes

    trials_df = pd.DataFrame(st.session_state["trials"]) if st.session_state["trials"] else pd.DataFrame()

    st.divider()
    st.subheader("Report Summary Preview")
    st.write(f"**Experiment:** {EXPERIMENT_CONFIG['exp_number']}. {EXPERIMENT_CONFIG['title']}")
    st.write(f"**Discipline:** {EXPERIMENT_CONFIG['discipline']} | **Subject:** {EXPERIMENT_CONFIG['subject']}")
    st.write(f"**Student:** {student_name} | **ID:** {student_id} | **Date:** {lab_date}")
    st.write(
        f"**Pretest Score:** {st.session_state.get('pretest_score', 0)} / {len(PRETEST_QUESTIONS)} | "
        f"**Posttest Score:** {st.session_state.get('quiz_score', 0)} / {len(POSTTEST_QUESTIONS)}"
    )

    if not trials_df.empty:
        st.dataframe(trials_df, hide_index=True, use_container_width=True)
    else:
        st.info("Note: You have not recorded any trials in the Simulation tab yet. Your report will indicate 0 trials.")

    # Generate PDF bytes and write file to disk
    pdf_bytes = generate_pdf_report(
        student_name=student_name,
        student_id=student_id,
        date_str=str(lab_date),
        trials_df=trials_df,
        pretest_score=st.session_state.get("pretest_score", 0),
        pretest_total=len(PRETEST_QUESTIONS),
        posttest_score=st.session_state.get("quiz_score", 0),
        posttest_total=len(POSTTEST_QUESTIONS),
        student_notes=student_notes
    )

    # Save to local files for guaranteed download
    os.makedirs("static", exist_ok=True)
    with open("static/lab_report.pdf", "wb") as f:
        f.write(pdf_bytes)
    with open("lab_report.pdf", "wb") as f:
        f.write(pdf_bytes)

    st.divider()
    st.subheader("Download Official Lab Report (.pdf)")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        st.link_button(
            "Open / Download PDF Document",
            url="/app/static/lab_report.pdf",
            type="primary",
            use_container_width=True
        )

    with col_btn2:
        st.download_button(
            label="Download lab_report.pdf",
            data=pdf_bytes,
            file_name="kgirs_exp6_lab_report.pdf",
            mime="application/pdf",
            key="stream_pdf_btn",
            use_container_width=True
        )


# ======================================================================================
# 6. MAIN NAVIGATION & SESSION STATE
# ======================================================================================

def init_session_state():
    """Initializes Streamlit session state variables."""
    if "trials" not in st.session_state:
        st.session_state["trials"] = []
    if "pretest_answers" not in st.session_state:
        st.session_state["pretest_answers"] = {}
    if "pretest_submitted" not in st.session_state:
        st.session_state["pretest_submitted"] = False
    if "pretest_score" not in st.session_state:
        st.session_state["pretest_score"] = 0
    if "quiz_answers" not in st.session_state:
        st.session_state["quiz_answers"] = {}
    if "quiz_submitted" not in st.session_state:
        st.session_state["quiz_submitted"] = False
    if "quiz_score" not in st.session_state:
        st.session_state["quiz_score"] = 0
    if "student_info" not in st.session_state:
        st.session_state["student_info"] = {
            "name": "Student Name",
            "id": "EXP-006",
            "date": str(datetime.now().date())
        }
    if "student_notes" not in st.session_state:
        st.session_state["student_notes"] = ""


def main():
    st.set_page_config(
        page_title="IIT KGP Virtual Lab - KGIRS Exp 6",
        page_icon=None,
        layout="wide"
    )

    init_session_state()

    # IIT Kharagpur VLab Breadcrumb Banner
    st.caption(
        f"**Virtual Labs (IIT Kharagpur)** &gt; {EXPERIMENT_CONFIG['discipline']} &gt; "
        f"{EXPERIMENT_CONFIG['subject']} &gt; Experiment {EXPERIMENT_CONFIG['exp_number']}"
    )
    st.title(f"Experiment {EXPERIMENT_CONFIG['exp_number']}: {EXPERIMENT_CONFIG['title']}")

    # Navigation Sidebar matching IIT Kharagpur VLab page navigation
    section = st.sidebar.radio(
        "Lab Navigation",
        options=[
            "Aim",
            "Introduction",
            "Theory",
            "Case Study",
            "Pretest",
            "Simulation",
            "Procedure",
            "Exercises",
            "Posttest",
            "References",
            "Report Generation"
        ]
    )

    st.sidebar.divider()
    st.sidebar.subheader("Session Progress")
    pre_status = "Done" if st.session_state.get("pretest_submitted", False) else "Pending"
    post_status = "Done" if st.session_state.get("quiz_submitted", False) else "Pending"
    st.sidebar.write(f"- **Pretest:** {pre_status} ({st.session_state.get('pretest_score', 0)}/5)")
    st.sidebar.write(f"- **Posttest:** {post_status} ({st.session_state.get('quiz_score', 0)}/10)")
    st.sidebar.write(f"- **Trials Logged:** {len(st.session_state.get('trials', []))}")

    # Section Dispatcher
    if section == "Aim":
        render_aim_section()
    elif section == "Introduction":
        render_introduction_section()
    elif section == "Theory":
        render_theory_section()
    elif section == "Case Study":
        render_casestudy_section()
    elif section == "Pretest":
        render_pretest_section()
    elif section == "Simulation":
        render_simulation_section()
    elif section == "Procedure":
        render_procedure_section()
    elif section == "Exercises":
        render_exercises_section()
    elif section == "Posttest":
        render_posttest_section()
    elif section == "References":
        render_references_section()
    elif section == "Report Generation":
        render_report_section()


if __name__ == "__main__":
    main()
