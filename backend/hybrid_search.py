from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from retrieval import dense_retrieve
from bm25_retrieval import bm25_retrieve

# ---------------------------------------------------------------------------
# 1. RRF CONSTANT
# ---------------------------------------------------------------------------
# k=60 is the standard value from the original RRF paper (Cormack et al. 2009).
# It smooths the contribution of top-ranked items — rank 1 gives 1/(60+1)=0.016,
# rank 10 gives 1/(60+10)=0.014. The drop-off is gentle, not sharp.
# Lower k = top ranks dominate more aggressively.
# Higher k = more uniform weighting across ranks.
RRF_K = 60


# ---------------------------------------------------------------------------
# 2. RRF FUSION FUNCTION
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    k: int = RRF_K,
) -> list[dict[str, Any]]:
    """
    Merges multiple ranked result lists into one unified ranking using RRF.

    Args:
        ranked_lists: List of result lists. Each result list is what you'd
                      get from dense_retrieve() or bm25_retrieve() — a list
                      of dicts with at minimum "chunk_id", "content", "source", "rank".
        k:            The RRF smoothing constant (default 60).

    Returns:
        A single list of chunks sorted by RRF score descending.
        Each chunk has an added "rrf_score" field.

    How RRF works step by step:
        1. For each ranked list, look at each chunk's rank (1=best, 2=second, ...)
        2. Compute that chunk's contribution: 1 / (k + rank)
        3. If a chunk appears in multiple lists, SUM its contributions
        4. Sort all chunks by their total RRF score

    Example with 2 lists:
        Dense results:  [ChunkA(rank=1), ChunkB(rank=2), ChunkC(rank=3)]
        BM25 results:   [ChunkB(rank=1), ChunkD(rank=2), ChunkA(rank=3)]

        ChunkA: 1/(60+1) + 1/(60+3) = 0.01639 + 0.01563 = 0.03202
        ChunkB: 1/(60+2) + 1/(60+1) = 0.01613 + 0.01639 = 0.03252  ← wins
        ChunkC: 1/(60+3) = 0.01563
        ChunkD: 1/(60+2) = 0.01613

        Final order: ChunkB, ChunkA, ChunkD, ChunkC
    """
    # chunk_id → running RRF score
    rrf_scores: dict[str, float] = defaultdict(float)

    # chunk_id → the full chunk dict (so we can reconstruct results)
    # We store the first time we see each chunk; content doesn't change across lists
    chunk_store: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for item in ranked_list:
            cid = str(item["chunk_id"])

            # Add this item's RRF contribution to its running total
            rank = item["rank"]
            rrf_scores[cid] += 1.0 / (k + rank)

            # Store the chunk data (only need to do this once per chunk)
            if cid not in chunk_store:
                chunk_store[cid] = {
                    "chunk_id": item["chunk_id"],
                    "content":  item["content"],
                    "source":   item["source"],
                }

    # Sort all seen chunks by their total RRF score, highest first
    sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

    # Build the final result list
    results = []
    for rank, cid in enumerate(sorted_chunk_ids, start=1):
        chunk = chunk_store[cid].copy()
        chunk["rrf_score"] = rrf_scores[cid]
        chunk["rank"] = rank
        results.append(chunk)

    return results


# ---------------------------------------------------------------------------
# 3. SINGLE QUERY HYBRID SEARCH
# ---------------------------------------------------------------------------
def _search_one_query(query: str, top_k: int) -> list[list[dict]]:
    """
    Runs dense + BM25 for a single query variant, in parallel threads.

    Returns a list of two ranked lists: [dense_results, bm25_results]

    We use ThreadPoolExecutor to run both retrievers at the same time.
    Dense retrieval (GPU/CPU embedding + DB query) and BM25 (in-memory lookup)
    can run simultaneously — no reason to wait for one before starting the other.

    Note: Python's GIL means true parallelism only happens during I/O waits
    (DB queries, model inference that releases the GIL). For our use case,
    the DB query I/O makes threading genuinely faster than sequential.
    """
    results = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit both tasks to run concurrently
        future_dense = executor.submit(dense_retrieve, query, top_k)
        future_bm25  = executor.submit(bm25_retrieve, query, top_k)

        # Collect results as they complete
        for future in as_completed([future_dense, future_bm25]):
            result = future.result()   # blocks until this future is done
            results.append(result)

    return results  # [dense_list, bm25_list] (order may vary due to concurrency)


# ---------------------------------------------------------------------------
# 4. MAIN HYBRID SEARCH (multi-query aware)
# ---------------------------------------------------------------------------
def hybrid_search(
    queries: list[str],
    top_k: int = 20,
    final_k: int = 10,
) -> list[dict[str, Any]]:
    """
    Full hybrid search: runs dense + BM25 for all query variants, fuses with RRF.

    Args:
        queries:  List of query strings. First element is the original query.
                  Remaining elements are rewritten variants from query_rewriter.py.
                  Example: ["NADRA registration docs", "documents needed NADRA", ...]
        top_k:    How many results to fetch per retriever per query.
        final_k:  How many results to return after RRF fusion.

    Returns:
        Top final_k chunks after RRF fusion across all retrievers and queries.

    Why multiple queries?
        query_rewriter.py produces N reformulations of the original question.
        Each reformulation might retrieve different (but relevant) chunks.
        By running ALL of them and fusing with RRF, we get better recall —
        we're less likely to miss the right chunk due to vocabulary mismatch.

    The math:
        With 3 queries × 2 retrievers × 20 results = 120 candidate slots.
        After deduplication, maybe 60-80 unique chunks.
        RRF fuses all their rankings → we return the top 10.
    """
    # PERFORMANCE FIX: previously this ran queries one at a time — query 1's
    # dense+BM25 search had to fully finish before query 2 even started. With
    # query_rewriter.py producing up to 4-5 variants (original + HyDE + 3
    # sub-questions), that serialized 4-5 rounds of embedding + DB round-trips
    # back to back, which is a major, avoidable contributor to slow response
    # times. Running all query variants concurrently (each already runs its
    # own dense+BM25 pair concurrently via _search_one_query) cuts wall-clock
    # retrieval time roughly by a factor of len(queries).
    all_ranked_lists: list[list[dict]] = []

    with ThreadPoolExecutor(max_workers=max(len(queries), 1)) as executor:
        futures = [executor.submit(_search_one_query, query, top_k) for query in queries]
        for future in as_completed(futures):
            all_ranked_lists.extend(future.result())

    # Fuse all ranked lists (could be 2, 4, 6 lists depending on query count)
    fused_results = reciprocal_rank_fusion(all_ranked_lists, k=RRF_K)

    # Return only the top final_k after fusion
    return fused_results[:final_k]


# ---------------------------------------------------------------------------
# 5. CONVENIENCE WRAPPER (single query, no rewriting needed)
# ---------------------------------------------------------------------------
def hybrid_search_single(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """
    Simplified interface: pass a single query string, get hybrid results.
    Used by spot_check.py and tests where we don't need multi-query expansion.
    """
    return hybrid_search(queries=[query], final_k=top_k)


# ---------------------------------------------------------------------------
# 6. SMOKE TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Simulate what query_rewriter.py will produce: original + 2 variants
    test_queries = [
        "What documents are required for NADRA CNIC registration?",
        "NADRA CNIC application document requirements",
        "list of documents needed for national identity card",
    ]

    print(f"\nHybrid search with {len(test_queries)} query variants\n")
    results = hybrid_search(test_queries, top_k=20, final_k=10)

    print(f"RRF fused results ({len(results)} chunks):\n")
    for r in results:
        print(f"  Rank {r['rank']} | RRF: {r['rrf_score']:.5f} | Source: {r['source']}")
        print(f"  {r['content'][:100]}...")
        print()
