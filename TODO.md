# TODO
- [x] STEP 1 LLM runtime feasibility test
- [x] STEP 2 repo memory files
- [x] STEP 3 simulator + data model
- [x] STEP 4 tool layer + guardrails
- [x] STEP 5 test the simulator independently
- [x] STEP 6 agent + model-provider abstraction
- [x] STEP 8 baseline policies (noop, rules) + oracle for environment validation
- [x] STEP 9 evaluation harness across seeded scenarios
- [x] STEP 10 CLI
- [x] STEP 11 CSV export/ingest + text/HTML run report
- [ ] STEP 7 run the agent against a real model  <-- BLOCKED: needs `claude setup-token`
      (one-time interactive browser login; uses the existing subscription, no billing)
- [ ] after auth: tune the agent prompt against real traces, then add `llm` to the matrix
