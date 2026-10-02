from src.vector_store import get_collection
from src.embeddings import create_embeddings


TEST_CASES = [
    {
        "question": "What is the OSI model?",
        "expected_page": 3
    },
    {
        "question": "What is SNMP?",
        "expected_page": 18
    },
    {
        "question": "What is TCP?",
        "expected_page": 8
    },
    {
        "question": "What is SSH?",
        "expected_page": 17
    },
    {
        "question": "What is ICMP?",
        "expected_page": 19
    },
    {
        "question": "What is a network protocol?",
        "expected_page": 2
    },
    {
        "question": "What is the TCP/IP model?",
        "expected_page": 3
    },
    {
        "question": "Who invented the telephone?",
        "expected_page": None
    }
]


THRESHOLDS = [
    1.0,
    1.2,
    1.4,
    1.6,
    1.8
]


collection = get_collection()


print("Starting threshold evaluation...\n")


# Store retrieval distances once.
results_cache = []


for test_case in TEST_CASES:

    question = test_case["question"]
    expected_page = test_case["expected_page"]

    query_embedding = create_embeddings(
        [question]
    )[0]

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=3
    )

    results_cache.append({
        "question": question,
        "expected_page": expected_page,
        "pages": [
            metadata["page"]
            for metadata in results["metadatas"][0]
        ],
        "distances": results["distances"][0]
    })


# Test each threshold.
for threshold in THRESHOLDS:

    hit_at_1 = 0
    hit_at_3 = 0

    known_questions = 0

    negative_tests = 0
    negative_passed = 0

    for result in results_cache:

        expected_page = result["expected_page"]
        pages = result["pages"]
        distances = result["distances"]

        # Known-answer test
        if expected_page is not None:

            known_questions += 1

            if pages[0] == expected_page:
                hit_at_1 += 1

            relevant_pages = [
                pages[i]
                for i, distance in enumerate(distances)
                if distance <= threshold
            ]

            if expected_page in relevant_pages:
                hit_at_3 += 1

        # Negative test
        else:

            negative_tests += 1

            if all(
                distance > threshold
                for distance in distances
            ):
                negative_passed += 1


    print("=" * 50)
    print(f"Threshold: {threshold}")
    print("=" * 50)

    print(
        f"Hit@1: "
        f"{hit_at_1}/{known_questions}"
    )

    print(
        f"Hit@3 after threshold: "
        f"{hit_at_3}/{known_questions}"
    )

    print(
        f"Negative tests passed: "
        f"{negative_passed}/{negative_tests}"
    )