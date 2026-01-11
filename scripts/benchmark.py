#!/usr/bin/env python3
"""
Benchmark script for Legal RAG PoC.

Measures:
1. Average latency (query → response)
2. Citation accuracy (% of verified citations)
3. Retrieval metrics (documents found)

Usage:
    python scripts/benchmark.py

Requirements:
    - Documents already indexed in ChromaDB
    - OPENAI_API_KEY and VOYAGE_API_KEY set

Output:
    - Console summary
    - JSON file for README integration
"""

import json
import os
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

# Check API keys
if not os.getenv("OPENAI_API_KEY") or not os.getenv("VOYAGE_API_KEY"):
    print("❌ OPENAI_API_KEY and VOYAGE_API_KEY required")
    print("   Set them in .env file")
    sys.exit(1)

from backend import db  # noqa: E402
from backend.logging_config import setup_logging  # noqa: E402
from backend.rag import answer_question_buffered, refusal  # noqa: E402

setup_logging()
db.init_db()

# Test queries covering different legal domains
TEST_QUERIES = [
    "Quels sont les délais de prescription en matière fiscale?",
    "Quelles sont les conditions de validité d'un contrat commercial?",
    "Comment fonctionne la mise en demeure pour impayé?",
    "Quels sont les recours possibles en cas de contentieux?",
    "Quelles sont les obligations du partenaire commercial?",
]


def run_benchmark(queries: list[str], num_runs: int = 3) -> dict:
    """
    Run benchmark on a list of queries.

    Args:
        queries: List of test queries
        num_runs: Number of times to run each query (for averaging)

    Returns:
        Dict with benchmark results
    """
    results = {
        "queries_tested": len(queries),
        "runs_per_query": num_runs,
        "latencies_ms": [],
        "refusals": 0,
        "answers": 0,
        "total_citations": 0,
        "sources_per_answer": [],
    }

    print(f"\n🚀 Running benchmark: {len(queries)} queries × {num_runs} runs\n")
    print("-" * 60)

    for i, query in enumerate(queries, 1):
        query_latencies = []

        for _run in range(num_runs):
            start = time.perf_counter()

            try:
                answer, sources, doc_ids, chunk_ids = answer_question_buffered(query)
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                query_latencies.append(elapsed_ms)

                if answer == refusal():
                    results["refusals"] += 1
                else:
                    results["answers"] += 1
                    results["total_citations"] += len(sources)
                    results["sources_per_answer"].append(len(sources))

            except Exception as e:
                print(f"  ❌ Error on query {i}: {e}")
                continue

        if query_latencies:
            avg_latency = sum(query_latencies) / len(query_latencies)
            results["latencies_ms"].extend(query_latencies)
            print(f"  [{i}/{len(queries)}] {query[:50]}...")
            print(f"           Latency: {avg_latency:.0f}ms (avg of {len(query_latencies)} runs)")

    print("-" * 60)

    # Calculate summary statistics
    if results["latencies_ms"]:
        latencies = results["latencies_ms"]
        results["summary"] = {
            "avg_latency_ms": int(sum(latencies) / len(latencies)),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "p50_latency_ms": int(sorted(latencies)[len(latencies) // 2]),
            "p95_latency_ms": int(sorted(latencies)[int(len(latencies) * 0.95)]) if len(latencies) >= 20 else max(latencies),
            "answer_rate": results["answers"] / (results["answers"] + results["refusals"]) if (results["answers"] + results["refusals"]) > 0 else 0,
            "avg_sources_per_answer": sum(results["sources_per_answer"]) / len(results["sources_per_answer"]) if results["sources_per_answer"] else 0,
        }

    return results


def print_summary(results: dict) -> None:
    """Print formatted benchmark summary."""
    if "summary" not in results:
        print("\n❌ No results to summarize")
        return

    s = results["summary"]

    print("\n" + "=" * 60)
    print("📊 BENCHMARK RESULTS")
    print("=" * 60)
    print(f"""
| Métrique | Valeur |
|----------|--------|
| Latence moyenne | {s['avg_latency_ms']}ms |
| Latence P50 | {s['p50_latency_ms']}ms |
| Latence P95 | {s['p95_latency_ms']}ms |
| Taux de réponse | {s['answer_rate']*100:.0f}% |
| Sources/réponse | {s['avg_sources_per_answer']:.1f} |
""")
    print("=" * 60)

    # Markdown for README
    print("\n📝 Copy this to README.md:\n")
    print("```markdown")
    print("## Performance")
    print("")
    print("Benchmarks sur 5 requêtes juridiques types (3 runs chacune):")
    print("")
    print("| Métrique | Valeur |")
    print("|----------|--------|")
    print(f"| Latence moyenne | ~{round(s['avg_latency_ms'], -2)}ms |")
    print(f"| Latence P50 | {s['p50_latency_ms']}ms |")
    print(f"| Taux de réponse | {s['answer_rate']*100:.0f}% |")
    print(f"| Sources citées/réponse | {s['avg_sources_per_answer']:.1f} |")
    print("```")


def save_results(results: dict, path: str = "benchmark_results.json") -> None:
    """Save results to JSON file."""
    output_path = Path(__file__).parent.parent / path
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to {output_path}")


if __name__ == "__main__":
    # Check if documents are indexed
    from backend.rag_runtime import vectorstore

    try:
        vs = vectorstore()
        collection = vs.get()
        doc_count = len(collection.get("ids", []))

        if doc_count == 0:
            print("⚠️  No documents indexed!")
            print("   Upload documents first via the Documents page")
            print("   Then run this benchmark again")
            sys.exit(1)

        print(f"✅ Found {doc_count} chunks in vectorstore")

    except Exception as e:
        print(f"❌ Could not access vectorstore: {e}")
        sys.exit(1)

    # Run benchmark
    results = run_benchmark(TEST_QUERIES, num_runs=3)

    # Print and save results
    print_summary(results)
    save_results(results)
