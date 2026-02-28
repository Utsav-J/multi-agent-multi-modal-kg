# DeepEval Comparison Report: MAS vs GraphRAG

**Total queries:** 60
**Categories:** Multi-Document Aggregation, Entity Relationship Reasoning, System Level / Architectural Understanding, Single-Document Fact Lookup, Entity-Specific Attribute Retrieval, Multi-Hop Chain Reasoning

---

## 1. Per-Metric Summary


| Metric          | MAS Mean | MAS Std | GR Mean | GR Std | MAS Median | GR Median | Diff (MAS-GR) |
| --------------- | -------- | ------- | ------- | ------ | ---------- | --------- | ------------- |
| AnswerRelevancy | 0.945    | 0.115   | 0.841   | 0.220  | 1.000      | 0.930     | +0.104        |
| Bias            | 1.000    | 0.000   | 1.000   | 0.000  | 1.000      | 1.000     | +0.000        |
| Clarity         | 0.817    | 0.196   | 0.777   | 0.212  | 0.900      | 0.900     | +0.040        |
| Completeness    | 0.956    | 0.139   | 0.947   | 0.169  | 1.000      | 1.000     | +0.009        |
| Tone            | 0.977    | 0.050   | 1.000   | 0.000  | 1.000      | 1.000     | -0.023        |
| Toxicity        | 1.000    | 0.000   | 1.000   | 0.000  | 1.000      | 1.000     | +0.000        |


---

## 2. Overall Scores (Weighted Mean)

Weights: {'AnswerRelevancy': 0.3, 'Clarity': 0.2, 'Completeness': 0.25, 'Tone': 0.15, 'Bias': 0.05, 'Toxicity': 0.05}

- **MAS overall:** 0.932
- **GraphRAG overall:** 0.894
- **Difference (MAS - GraphRAG):** +0.038

---

## 3. Statistical Comparison

### Per-metric paired difference (MAS - GraphRAG)


| Metric          | Mean Diff | t               | Cohen's d |
| --------------- | --------- | --------------- | --------- |
| AnswerRelevancy | +0.104    | 3.50 (p~0.001)  | 0.45      |
| Bias            | +0.000    | 0.00 (p~1.000)  | 0.00      |
| Clarity         | +0.040    | 1.33 (p~0.187)  | 0.17      |
| Completeness    | —         | —               | —         |
| Tone            | -0.023    | -3.62 (p~0.001) | -0.47     |
| Toxicity        | +0.000    | 0.00 (p~1.000)  | 0.00      |


---

## 4. Pass Rates


| Metric          | MAS Pass | MAS Rate | GR Pass | GR Rate |
| --------------- | -------- | -------- | ------- | ------- |
| AnswerRelevancy | 60/60    | 100.0%   | 54/60   | 90.0%   |
| Bias            | 0/60     | 0.0%     | 0/60    | 0.0%    |
| Clarity         | 52/60    | 86.7%    | 50/60   | 83.3%   |
| Completeness    | 58/60    | 96.7%    | 57/60   | 95.0%   |
| Tone            | 60/60    | 100.0%   | 60/60   | 100.0%  |
| Toxicity        | 0/60     | 0.0%     | 0/60    | 0.0%    |


*Note: For Bias and Toxicity, raw score 0 means no issues (ideal). Overall scores use inverted values (1 - raw) so higher = better.*

---

## 5. Category-wise Breakdown & Comparison

### Entity Relationship Reasoning (n=10)


| Metric          | MAS Mean | GraphRAG Mean | Diff (MAS−GR) | Winner   |
| --------------- | -------- | ------------- | ------------- | -------- |
| AnswerRelevancy | 0.980    | 0.959         | +0.021        | MAS      |
| Bias            | 1.000    | 1.000         | +0.000        | Tie      |
| Clarity         | 0.780    | 0.750         | +0.030        | MAS      |
| Completeness    | 0.890    | 0.910         | -0.020        | GraphRAG |
| Tone            | 0.950    | 1.000         | -0.050        | GraphRAG |
| Toxicity        | 1.000    | 1.000         | +0.000        | Tie      |


**Overall (weighted):** MAS = 0.915, GraphRAG = 0.915, Diff = -0.000 → **GraphRAG**

**Metric wins:** MAS 2, GraphRAG 2

### Entity-Specific Attribute Retrieval (n=10)


| Metric          | MAS Mean | GraphRAG Mean | Diff (MAS−GR) | Winner   |
| --------------- | -------- | ------------- | ------------- | -------- |
| AnswerRelevancy | 0.897    | 0.639         | +0.259        | MAS      |
| Bias            | 1.000    | 1.000         | +0.000        | Tie      |
| Clarity         | 0.710    | 0.890         | -0.180        | GraphRAG |
| Completeness    | 0.944    | 1.000         | -0.056        | GraphRAG |
| Tone            | 0.940    | 1.000         | -0.060        | GraphRAG |
| Toxicity        | 1.000    | 1.000         | +0.000        | Tie      |


**Overall (weighted):** MAS = 0.888, GraphRAG = 0.870, Diff = +0.019 → **MAS**

**Metric wins:** MAS 1, GraphRAG 3

### Multi-Document Aggregation (n=10)


| Metric          | MAS Mean | GraphRAG Mean | Diff (MAS−GR) | Winner |
| --------------- | -------- | ------------- | ------------- | ------ |
| AnswerRelevancy | 0.957    | 0.883         | +0.075        | MAS    |
| Bias            | 1.000    | 1.000         | +0.000        | Tie    |
| Clarity         | 0.970    | 0.840         | +0.130        | MAS    |
| Completeness    | 1.000    | 0.860         | +0.140        | MAS    |
| Tone            | 1.000    | 1.000         | +0.000        | Tie    |
| Toxicity        | 1.000    | 1.000         | +0.000        | Tie    |


**Overall (weighted):** MAS = 0.981, GraphRAG = 0.898, Diff = +0.083 → **MAS**

**Metric wins:** MAS 3, GraphRAG 0

### Multi-Hop Chain Reasoning (n=10)


| Metric          | MAS Mean | GraphRAG Mean | Diff (MAS−GR) | Winner   |
| --------------- | -------- | ------------- | ------------- | -------- |
| AnswerRelevancy | 0.975    | 0.939         | +0.036        | MAS      |
| Bias            | 1.000    | 1.000         | +0.000        | Tie      |
| Clarity         | 0.850    | 0.760         | +0.090        | MAS      |
| Completeness    | 0.950    | 0.910         | +0.040        | MAS      |
| Tone            | 0.990    | 1.000         | -0.010        | GraphRAG |
| Toxicity        | 1.000    | 1.000         | +0.000        | Tie      |


**Overall (weighted):** MAS = 0.948, GraphRAG = 0.911, Diff = +0.037 → **MAS**

**Metric wins:** MAS 3, GraphRAG 1

### Single-Document Fact Lookup (n=10)


| Metric          | MAS Mean | GraphRAG Mean | Diff (MAS−GR) | Winner   |
| --------------- | -------- | ------------- | ------------- | -------- |
| AnswerRelevancy | 0.880    | 0.678         | +0.202        | MAS      |
| Bias            | 1.000    | 1.000         | +0.000        | Tie      |
| Clarity         | 0.760    | 0.760         | +0.000        | Tie      |
| Completeness    | 0.950    | 1.000         | -0.050        | GraphRAG |
| Tone            | 0.980    | 1.000         | -0.020        | GraphRAG |
| Toxicity        | 1.000    | 1.000         | +0.000        | Tie      |


**Overall (weighted):** MAS = 0.900, GraphRAG = 0.855, Diff = +0.045 → **MAS**

**Metric wins:** MAS 1, GraphRAG 2

### System Level / Architectural Understanding (n=10)


| Metric          | MAS Mean | GraphRAG Mean | Diff (MAS−GR) | Winner |
| --------------- | -------- | ------------- | ------------- | ------ |
| AnswerRelevancy | 0.981    | 0.948         | +0.033        | MAS    |
| Bias            | 1.000    | 1.000         | +0.000        | Tie    |
| Clarity         | 0.830    | 0.660         | +0.170        | MAS    |
| Completeness    | 1.000    | 1.000         | +0.000        | Tie    |
| Tone            | 1.000    | 1.000         | +0.000        | Tie    |
| Toxicity        | 1.000    | 1.000         | +0.000        | Tie    |


**Overall (weighted):** MAS = 0.960, GraphRAG = 0.916, Diff = +0.044 → **MAS**

**Metric wins:** MAS 2, GraphRAG 0

### Category-level summary


| Category                                   | n   | MAS Overall | GraphRAG Overall | Diff   | Winner   |
| ------------------------------------------ | --- | ----------- | ---------------- | ------ | -------- |
| Entity Relationship Reasoning              | 10  | 0.915       | 0.915            | -0.000 | GraphRAG |
| Entity-Specific Attribute Retrieval        | 10  | 0.888       | 0.870            | +0.019 | MAS      |
| Multi-Document Aggregation                 | 10  | 0.981       | 0.898            | +0.083 | MAS      |
| Multi-Hop Chain Reasoning                  | 10  | 0.948       | 0.911            | +0.037 | MAS      |
| Single-Document Fact Lookup                | 10  | 0.900       | 0.855            | +0.045 | MAS      |
| System Level / Architectural Understanding | 10  | 0.960       | 0.916            | +0.044 | MAS      |


---

## 6. Conclusions

1. **Overall winner:** MAS (overall score diff = +0.038)
2. **Metric-level:** MAS wins on 3 metrics, GraphRAG on 1.
3. **Largest gaps:** AnswerRelevancy.
4. **Paired t-test (all scores):** statistically significant (p = 0.0140).

