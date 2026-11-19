# Adaptive gateway experiment plan

Adaptive gateways remain a planned experiment, not part of the current index. The equal-budget offline headroom test has now been completed and failed its acceptance gate: the greedy facility-location oracle did not improve held-out recall and reduced distance computations by only 0.21%. Online adaptive promotion is therefore deferred unless a future gateway-edge or search design creates measurable oracle headroom.

## Objective

At a fixed gateway count, edge budget and search budget, select entry/routing nodes that either:

- improve recall by at least one percentage point; or
- reduce distance computations by at least 10% at equal recall.

The experiment must not increase the existing gateway density of one gateway per 512 inserted records or the gateway degree of eight.

## Phase 1: offline oracle headroom

1. Split the chronological 100k stream into the first 50k records for gateway selection and the next 50k for held-out evaluation.
2. Generate candidate gateways every 64 or 128 insertions.
3. Measure each candidate's marginal contribution on training queries:
   - reduction in level-0 distance computations;
   - recall improvement at fixed efSearch;
   - number of successful query entries;
   - semantic clusters reached that periodic gateways miss.
4. Select an equal-count greedy oracle set.
5. Compare periodic, random and oracle gateways with exactly 195 gateways and degree eight.

Stop if the oracle fails the objective. In that case gateway placement is not the current bottleneck.

## Phase 2: gateway feature dataset

For every promotion candidate, record only online-available features:

- insertion rank and elapsed time since the previous gateway;
- distance to the nearest encountered gateway;
- local k-neighbour radius from construction search;
- normalized semantic novelty `nearest_gateway_distance / local_radius`;
- number and diversity of nearby ordinary nodes;
- semantic cluster proxy or construction-neighbour diversity;
- temporal traffic density;
- current gateway degree saturation;
- temporal-scale co-eligibility with potential gateway neighbours.

Use oracle membership and measured marginal search benefit as diagnostic labels. This identifies whether useful gateways represent dense semantic regions, drift events, cluster boundaries or outliers.

## Phase 3: budgeted online policy

Within each 512-record epoch:

1. Always retain one periodic gateway as a coverage guarantee.
2. Maintain a bounded heap of adaptive candidates.
3. Score candidates using semantic novelty, represented-neighbour support and temporal novelty.
4. Penalize isolated outliers and candidates whose useful edges are unlikely to survive range filtering.
5. Promote at most one additional adaptive candidate only by replacing, not adding to, the fixed gateway budget.

A starting score is:

```text
score = alpha * normalized_semantic_novelty
      + beta  * represented_neighbor_support
      + gamma * temporal_novelty
      + delta * construction_neighbor_diversity
      - lambda * outlier_penalty
```

Thresholds should use rolling percentiles rather than fixed global values.

## Phase 4: temporally compatible gateway edges

Gateway edges should use independent 24, 168 and 720-hour candidate pools. An edge outside the maximum supported window has no strict-query utility and should not consume degree.

Compare at equal degree:

1. periodic gateways with global semantic edges;
2. periodic gateways with temporal-scale-aware edges;
3. oracle gateways with temporal-scale-aware edges;
4. online adaptive gateways with temporal-scale-aware edges.

## Evaluation

Report results by coverage and fragmentation:

- recall and precision at 100;
- mean, p50 and p95 latency;
- distance computations and nodes visited;
- gateway seeds considered and selected;
- gateway edges examined and rejected by range;
- insertion latency and gateway scoring cost;
- storage and degree distribution;
- gateway promotion rate and semantic/temporal coverage.

Run at efSearch 128, 256 and 512. Keep exact fallback at 5,000 records.

## Acceptance criteria

Implement adaptive online promotion only if:

1. the offline oracle demonstrates meaningful headroom;
2. the learned heuristic captures a substantial fraction of the oracle gain;
3. gains hold on the held-out chronological suffix;
4. storage, gateway count and degree remain equal to the periodic baseline; and
5. insertion p95 does not regress materially.

This staged process prevents adding a complex adaptive heuristic to a topology where gateway placement is not the limiting factor.
