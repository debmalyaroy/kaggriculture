"""Track P: the improved harness (operator order 2026-09-05).

The agent-improvement machine, separated from the frozen bandit lane:
corpus (bc_corpus), trainers (bc_train / bc_train_gru), architecture probe
(bc_arch_test), embedder (bc_embed), upset gauntlet, hold-shape screening,
and the Bradley-Terry ranker (bt_rank) -- the currency of the FINAL
leaderboard per the host's 2026-09-05 confirmation.

The bandit lane (agents/, submit path, live pair) is NOT touched from here;
this package produces gated candidates and reports, never submissions.
"""
