r"""Capstone Checkpoint 2.1 — Retrieval Strategy Design and Baseline Implementation (starter).
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code / PyCharm / Jupytext.
"""

# %% [markdown]
# # Capstone Checkpoint 2.1 — Retrieval Strategy Design and Baseline Implementation
# **MO-LLM Module 2 / Required Capstone Checkpoint (120 minutes)**
#
# ## What this checkpoint is
#
# In Checkpoint 1.1, you showed that a plain LLM can't reliably answer questions about
# your corpus. Now, you will **add retrieval**: Design a retrieval strategy for your scenario
# and build a **baseline retrieval system** that finds the most relevant documents for
# a query, so the model can ground its answers in them.
#
# This mirrors the Module 2 labs — keyword (BM25), vector (semantic), and hybrid
# retrieval — applied to your own capstone corpus. The graded deliverable is the completed 
# Capstone Checkpoint 2.1 worksheet, which includes your written responses and evidence of your 
# retrieval system implementation and testing. This script provides a small working example of 
# baseline retrieval. Use it to understand the retrieval workflow, then adapt the code to implement 
# and test a baseline retriever using your selected capstone dataset.
#
# **Learning outcomes (Module 2):**
# 1. Design a retrieval strategy appropriate for a given dataset and query type.
# 2. Implement and test a baseline retrieval system using structured and/or semantic
#    approaches.

# %% [markdown]
# ## Step 1 — Keep your capstone scenario
#
# Use the **same scenario** you chose in Checkpoint 1.1.
#
# | Scenario | Corpus | Retrieval considerations |
# |---|---|---|
# | **Research Paper Navigator** | ~150 research-paper PDFs (`Labs/CapstoneDatasets/ResearchPapers/`) | long documents; you'll likely chunk them; questions often name a specific paper or compare papers. |
# | **Wikipedia Retrieval Engine** | ~2,400 Wikipedia HTML articles (`Labs/CapstoneDatasets/Wikipedia/`) | many short-to-medium articles; questions name a figure/place or span several articles. |
#
# A good baseline is keyword (BM25), semantic (embeddings + vector search), or a
# hybrid of both — exactly what you built in Labs 1.2–2.2.

# %% [markdown]
# ## Setup (~5 min)
#
# 1. **Python 3.11 or 3.12**
# 2. `pip install langchain-openai langchain-core python-dotenv`
# 3. Use the OpenRouter API key provided for this program. This checkpoint uses
#  the `openai/gpt-5.4-mini` model, with usage covered by the course credits. (this uses the paid gpt-5.4-mini chat model — covered by 
# your course credits — and a keyword retriever, no embeddings).
# 4. Create a `.env` file next to this script: `OPENROUTER_API_KEY=sk-or-v1-...`
#
# This runs on a tiny built-in sample corpus, so you do not need to prepare your own
# dataset. It still requires an OpenRouter API key to run the LLM (it is not offline or
# free of API calls). Your real baseline (over your full corpus) is what you describe in
# the writeup.

# %%
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import os
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi


# %%
SCRIPT_DIR = Path(__file__).resolve().parent
LAB_DIR = SCRIPT_DIR.parent
load_dotenv(LAB_DIR / ".env")

EMBEDDING_MODEL = "openai/text-embedding-3-small"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
CHROMA_DIR = SCRIPT_DIR / f"chroma_db_{CHUNK_SIZE}"
USE_HISTORY = False  # compare each question independently
CANDIDATE_POOL = 10        # Candidates pulled from EACH retriever before fusion
WEIGHT_BM25 = 0.5
WEIGHT_VECTOR = 0.5

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"  # latest small OpenAI model, fast; covered by course credits
TEMPERATURE = 0.2
TOP_K = 6
LOG_PATH = Path.cwd() / "checkpoint_2_1_retrieval.log"

# === SET THIS to the scenario you chose in Checkpoint 1.1 ===
SCENARIO = "research_papers"   # "research_papers" or "wikipedia"

ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question using ONLY the provided "
    "documents, and quote from them where you can. If the documents do not contain "
    "the answer, say so rather than guessing."
)


# %%
def check_api_key() -> str:
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Use the OpenRouter API key "
            "provided for this course, put it in a .env file next to this "
            "script, and rerun."
        )
    return key


def make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        api_key=check_api_key(),
        base_url=OPENROUTER_BASE_URL,
    )


def log(label: str, text: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {label}\n{text}\n{'-' * 72}\n")

# ─── Vector side (from Lab 2.1) ──────────────────────────────────────
def get_embeddings() -> OpenAIEmbeddings:
    api_key = check_api_key()
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        check_embedding_ctx_length=False,  # send text to OpenRouter, not token IDs
        chunk_size=128,                    # emails per embedding request
    )

def build_or_load_db(chunks: list, chroma_dir: str = CHROMA_DIR) -> Chroma:
    """Reuse the saved index, or embed the chunks and save a new one."""
    cache = Path(chroma_dir)
    if cache.is_dir() and any(cache.iterdir()):
        print(f"Loading saved vector index from {cache} (no document embedding calls).")
        return Chroma(persist_directory=str(cache), embedding_function=get_embeddings())

    # Persist the vector store to disk for reuse without re-embedding.
    vector_store = Chroma.from_documents(chunks, get_embeddings(), persist_directory=str(cache))

    # Print a message indicating that the vector index has been saved and how many emails were embedded.
    print(f"Saved vector index to {cache} (embedded {len(chunks)} chunks).")

    return vector_store


# %% [markdown]
# ## A tiny sample corpus (stands in for your real one)
#
# There are six short "documents" on distinct topics so a baseline retriever has something to
# discriminate between. Your real corpus is the PDFs/articles in `Labs/CapstoneDatasets/`,
# which came with the course in Module 1. Point your code at your local copy of that
# folder, and update the path if your checkout puts it elsewhere.


# %%
"""SAMPLE_DOCS = [
    {"id": "doc1", "text": "Program synthesis: generating programs automatically from a specification, such as input-output examples or a logical formula."},
    {"id": "doc2", "text": "The sketching approach lets a programmer write a partial program with holes, and a synthesizer fills the holes to satisfy a specification."},
    {"id": "doc3", "text": "Retrieval-augmented generation grounds a language model's answers in documents retrieved from a corpus, reducing hallucination."},
    {"id": "doc4", "text": "BM25 is a keyword ranking function that scores documents by term frequency and inverse document frequency."},
    {"id": "doc5", "text": "Vector search embeds text into dense vectors and ranks documents by cosine similarity to the query embedding."},
    {"id": "doc6", "text": "Evaluation of retrieval systems measures whether the retrieved documents actually contain the information needed to answer the query."},
]"""

PDF_DIR = SCRIPT_DIR / "ResearchPapers"

all_pages = []
for path in sorted(PDF_DIR.glob("*.pdf")):  # [:5] limit to 5 PDFs for this tiny sample
    # 1. create a PyPDFLoader for this path (cast to str for PyPDFLoader)
    pdf_loader = PyPDFLoader(str(path))

    try:
        # 2. load it (inside try/except, so a bad PDF prints a warning and the loop continues)
        pages = pdf_loader.load()

        # 3. add its pages to all_pages with extend
        all_pages.extend(pages)
    except Exception as e:
        print(f"Warning: Failed to load {path}: {e}")

print(f"Loaded {len(all_pages)} pages")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)
# Split the loaded pages into chunks for retrieval
pdf_chunks = text_splitter.split_documents(all_pages)

# Add a chunk_id to each chunk for retrieval and citation
for i, chunk in enumerate(pdf_chunks):
    # Including the source filename and a 1-based index in the chunk_id helps identify the chunk's origin
    chunk.metadata["chunk_id"] = f"{Path(chunk.metadata['source']).name}::{i+1}"

# Create a dictionary mapping chunk_id to the chunk for easy retrieval
DOC_BY_ID = {chunk.metadata["chunk_id"]: chunk for chunk in pdf_chunks}

# Build or load the vector database for retrieval
db = build_or_load_db(pdf_chunks)



# %% [markdown]
# ## Step 2 — The baseline retriever (provided)
#
# A simple keyword-overlap retriever scores each document by how many query words
# it shares, and returns the top-k. This is the smallest possible baseline (a stand-in
# for the BM25 / vector / hybrid retriever you built in the Module 2 labs). The
# `answer` function then asks the LLM using only the retrieved documents.

# %%
# ─── Provided: keyword + vector building blocks ──────────────────────
_STOPWORDS = {
    "a", "an", "the", "and", "but", "or", "nor", "so", "yet", "for",
    "in", "on", "at", "to", "of", "by", "with", "from", "into", "onto", "upon",
    "about", "above", "below", "between", "through", "during", "before", "after",
    "under", "over", "around", "along", "across", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "i", "we", "you", "he", "she", "it", "they", "me", "us", "him", "her", "them",
    "my", "our", "your", "his", "its", "their", "this", "that", "these", "those",
    "as", "if", "up", "out", "not", "no",
}


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]

def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))

def _normalize(scores: list[float], invert: bool = False) -> list[float]:
    """Min-max scale a list of scores to [0, 1]. If invert is True, flip the scores 
    so a LOW raw value (e.g., a small vector distance = very similar) becomes a HIGH 
 normalized score."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [0.5] * len(scores)  # all equal → neutral
    norm = [(s - lo) / (hi - lo) for s in scores]
    return [1.0 - n for n in norm] if invert else norm

class HybridRetriever:
    def __init__(self, chunks: list, db: Chroma):
    
        # Keyword index over the same chunks (kept in memory).
        self._ids = [c.metadata["chunk_id"] for c in chunks]
        # Prepare the text content of each chunk for BM25 scoring.
        self._contents = [c.page_content for c in chunks]
        # Tokenize the documents for BM25 scoring.
        self._tokenized_documents = [tokenize(doc) for doc in self._contents]

        # Check that the tokenized documents contain searchable tokens; if not, raise an error.
        if not any(self._tokenized_documents):
            raise ValueError("The chunks contain no searchable tokens.")

        # Initialize the BM25 retriever with the tokenized documents.
        self._bm25 = BM25Okapi(self._tokenized_documents)

        # Validate the weights for BM25 and vector retrieval. They must be nonnegative, and at least one must be positive.
        if min(WEIGHT_BM25, WEIGHT_VECTOR) < 0 or WEIGHT_BM25 + WEIGHT_VECTOR <= 0:
            # If the weights are invalid, raise a ValueError with a descriptive message.
            raise ValueError("Weights must be nonnegative and at least one must be positive.")
        # Store the vector database for later use in retrieval.
        self._db = db
        print(f"Hybrid retriever ready over {len(self._ids)} chunks.")

    # Provided single-retriever helpers. BM25: higher score = better.
    # Vector: returns a DISTANCE, lower = better.
    def _bm25_topk(self, query: str, k: int) -> list[tuple[str, str, float]]:
        tokens = tokenize(query)
        if not tokens or k <= 0:
            return []
        scores = self._bm25.get_scores(tokens)
        # Unmatched queries otherwise return arbitrary zero-score chunks. Use lexical
        # overlap, not score > 0: BM25 scores may be zero/negative in small corpora.
        query_terms = set(tokens)
        candidates = [i for i, doc in enumerate(self._tokenized_documents)
                      if query_terms.intersection(doc)]
        top = sorted(candidates, key=lambda i: scores[i], reverse=True)[:k]
        return [(self._ids[i], self._contents[i], scores[i]) for i in top]

    def _vector_topk(self, query: str, k: int) -> list[tuple[str, str, float]]:
        if not query.strip() or k <= 0:
            return []
        results = self._db.similarity_search_with_score(query, k=k)
        return [(d.metadata["chunk_id"], d.page_content, s) for d, s in results]

    def getTopK(self, query: str, k: int) -> list[tuple[str, str, float]]:
        """Return the top k chunks by combining BM25 keyword and vector search scores.

        Each method's candidate scores are normalized to 0-1 (vector distances are
        inverted so closer is better), then combined with the configured weights.
        A chunk found by only one method gets only that method's share. If one
        weight is zero, the other method's results are returned directly.

        Each result is (chunk_id, text, score), best first. A blank query or
        k <= 0 returns an empty list.
        """

        if query is None or not query.strip() or k <= 0:
            return []

        # If one of the weights is zero, return only the other retriever's results, with vector distances inverted to a score.
        if WEIGHT_BM25 <= 0:
            return [(name, content, -distance)
                                for name, content, distance in self._vector_topk(query, k)]

        # If the vector weight is zero, return only the BM25 results.
        if WEIGHT_VECTOR <= 0:
            return self._bm25_topk(query, k)

        # Retrieve top candidates from both BM25 and vector retrievers
        bm25_results = self._bm25_topk(query, CANDIDATE_POOL)
        vector_results = self._vector_topk(query, CANDIDATE_POOL)

        # Normalize the scores for both BM25 and vector results
        bm25_scores = []
        for score in bm25_results:
            bm25_scores.append(score[2])
        bm25_normalized = _normalize(bm25_scores, invert=False)

        vector_scores = []
        for score in vector_results:
            vector_scores.append(score[2])
        vector_normalized = _normalize(vector_scores, invert=True)

        # Combine the results with weighted sum
        combined_results = {}
        # Add BM25 results to the combined results dictionary with their normalized scores
        for i, (name, content, _) in enumerate(bm25_results):
            combined_results[name] = (content, bm25_normalized[i] * WEIGHT_BM25)

        # Add vector results to the combined results dictionary, updating scores if the email is already present
        for i, (name, content, _) in enumerate(vector_results):
            if name in combined_results:
                # If the email is already present in the combined results, update its score by adding the weighted vector score to the existing score
                combined_results[name] = (content, combined_results[name][1] + vector_normalized[i] * WEIGHT_VECTOR)
            else:
                # If the email is not present in the combined results, add it with its weighted vector score
                combined_results[name] = (content, vector_normalized[i] * WEIGHT_VECTOR)

        # Sort the combined results by score in descending order and return the top k
        sorted_results = sorted(combined_results.items(), key=lambda x: x[1][1], reverse=True)[:k]
        return [(name, content, score) for name, (content, score) in sorted_results]
    
retriever = HybridRetriever(pdf_chunks, db)


def retrieve(query: str, k: int = TOP_K) -> list[tuple[str, float]]:
    """Hybrid retrieval: return the top-k (chunk_id, score) pairs from HybridRetriever."""
   
    scored = retriever.getTopK(query, k)
   
    return [(chunk_id, score) for chunk_id, _, score in scored]


def answer(llm: ChatOpenAI, query: str, doc_ids: list[str]) -> str:
    ## Ask the LLM to answer the query using only the retrieved documents. 
    # Each document is labeled with its chunk_id and page number for citation. 
    # If the retrieved documents don’t contain the answer, the system prompt tells the LLM to say so rather than guess.
    context = "\n\n".join(
        f"[{i}, p. {DOC_BY_ID[i].metadata.get('page', 0) + 1}] {DOC_BY_ID[i].page_content}"
        for i in doc_ids if i in DOC_BY_ID
    )
    messages = [
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {query}"),
    ]
    return llm.invoke(messages).content


# %% [markdown]
# ## Step 3 — Your representative queries (TODO)
#
# Submission item #2 asks for **3-5 representative queries** for your scenario and the
# results your system retrieves for each. Write those queries here. Some good ones include questions that:
#
# - Are answerable from **one** document (tests precision),
# - Need **several** documents (tests recall / aggregation),
# - Have wording that **differs** from the document's wording (i.e., tests whether
#   keyword vs. semantic retrieval matters for your corpus)
#
# Return a list of 3-5 query strings.

# %%
def my_representative_queries() -> list[str]:
    """Return 3-5 representative queries for the Research Paper Navigator.

    Reuses questions from Checkpoint 1.1 (known ground truth) plus one
    paraphrased query, covering single-paper, quote, multi-paper and
    paraphrased retrieval.
    """
    return [
        # File: DBLP_conf_ppsn_KrawiecS14.pdf
        # Type: single paper, precise (exact numbers; BM25 should help)
        # Where: Abstract (last sentence) and Section 1 Introduction, final paragraph (p. 435)
        # ANSWER: 35 benchmarks, representing three different domains.
        # (Checkpoint 1.1 without retrieval: model said 11 benchmarks, 5 domains.)
        "In the paper 'Improving Genetic Programming with Behavioral Consistency "
        "Measure' (Krawiec and Solar-Lezama, PPSN 2014), how many benchmarks did "
        "the authors use to evaluate their consistency measure, and how many "
        "different domains did they represent?",

        # File: DBLP_conf_ppsn_KrawiecS14.pdf
        # Type: single paper, verbatim quote
        # Where: Abstract, first sentence (p. 434)
        # ANSWER (verbatim): "Program synthesis tasks usually specify only the
        #   desired output of a program and do not state any expectations about
        #   its internal behavior."
        "Quote, word for word, the first sentence of the abstract of 'Improving "
        "Genetic Programming with Behavioral Consistency Measure' (Krawiec and "
        "Solar-Lezama, PPSN 2014).",

        # File: DBLP_conf_ppopp_KurajSP22.pdf  (actual content: Xiang & Vaidya)
        # Type: single paper, named entities
        # Where: Abstract (p. 1)
        # ANSWER: GentleRain and Cure. Previous global stabilization designs
        #   assume full replication.
        # (Checkpoint 1.1 without retrieval: model said Cure and "Orbe".)
        "In the paper 'Global Stabilization for Causally Consistent Partial "
        "Replication' (Xiang and Vaidya), which two previous systems does the "
        "abstract name as using global stabilization, and what assumption about "
        "replication do all previous global stabilization designs make?",

        # File: DBLP_conf_acl_GuLJOLSS24.pdf
        # Type: paraphrased (avoids the paper's key term "counterfeit samples";
        #   vector search should do the work)
        # Where: Abstract and Introduction (p. 74)
        # ANSWER: Gu et al., "The Counterfeit Conundrum" (ACL 2024). Models often
        #   misclassify these subtly wrong programs as correct, predict their
        #   execution as if they were correct, and repair them at a low rate.
        "Which paper studies whether code-generating language models can recognize "
        "subtly wrong programs that they produced themselves, and what did it find?",

        # Files: DBLP_conf_ppopp_KurajSP22.pdf (Xiang & Vaidya) and
        #        DBLP_conf_pldi_Solar-LezamaJB08.pdf (actual content: Zhu et al.)
        # Type: multi-paper comparison (tests whether chunks from BOTH papers are retrieved)
        # Where: Abstracts (p. 1 of each)
        # ANSWER (key points):
        #   - Xiang & Vaidya: distributed systems; causal consistency with partial
        #     replication via global stabilization; proves optimal Global Stable
        #     Time; simulation results.
        #   - Zhu et al.: computer vision; Semantic Component-level Memory module in a
        #     Structured Sketch Recognition network for explainability; SPG and
        #     SketchIME datasets.
        #   - Unrelated fields and methods (proof + simulation vs. neural network +
        #     dataset experiments).
        "Compare these two papers: 'Global Stabilization for Causally Consistent "
        "Partial Replication' (Xiang and Vaidya) and 'Enhance Sketch Recognition's "
        "Explainability via Semantic Component-Level Parsing' (Zhu et al., AAAI 2024). "
        "What problem does each address, and how do their approaches differ?",
    ]


# %% [markdown]
# ## Step 4 — Run the baseline and capture the evidence
#
# This runs each query through the baseline retriever and the LLM, printing the
# retrieved document ids/scores and the grounded answer, and logging everything to
# `checkpoint_2_1_retrieval.log`. The retrieved documents from the output are the rest of the evidence for
# submission item #2.

# %%
def run() -> None:
    llm = make_llm()
    queries = my_representative_queries()
    print(f"Checkpoint 2.1 — baseline retrieval  |  scenario: {SCENARIO}\n")
    for i, query in enumerate(queries, 1):
        hits = retrieve(query, TOP_K)
        print("=" * 72)
        print(f"QUERY {i}: {query}")
        print(f"  retrieved: {hits}")
        if not hits:
            print("  (nothing matched — note this in your writeup)")
            continue
        ans = answer(llm, query, [doc_id for doc_id, _ in hits])
        print(f"  answer: {ans}\n")
        log(f"QUERY {i}: {query}", f"retrieved={hits}\nanswer={ans}")
    print("=" * 72)
    print("Done. Use the retrieved document results above as evidence in your writeup, and "
          "describe your REAL baseline (over your full corpus) in the submission.")


run()

# %% [markdown]
# ## Step 5 — Your written submission (the graded deliverable)
#
# Use your completed retrieval implementation and test results to complete the Capstone Checkpoint 2.1
# worksheet. In the worksheet, you will document your retrieval approach, provide evidence that your 
# system is functioning, include 3–5 representative queries and retrieved results, and reflect on where
# your approach performs well and where it struggles.  
 
# Save your completed Python file in the appropriate checkpoint folder in your GitHub repository. 
# Upload the completed worksheet only to the learning platform as your graded submission.
