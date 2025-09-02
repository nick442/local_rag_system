# Templates

Use these as starting points for small, reproducible runs.

## Retrieval Test Prompts (JSON)
Structure expected by `scripts/tests/run_retrieval_tests.py`:
```
{
  "retrieval_test_prompts": {
    "test_sets": {
      "smoke": {
        "prompts": [
          {
            "id": "smoke_1",
            "query": "What documents are available in the demo corpus?",
            "expected_elements": ["document", "demo", "corpus"],
            "expected_sources": 1
          }
        ]
      }
    }
  }
}
```

Run:
`python scripts/tests/run_retrieval_tests.py --config tests/retrieval_test_prompts.json --output test_results`

