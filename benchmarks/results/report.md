# DeepEval Comparison Report: MAS vs GraphRAG

**Total queries:** 60
**Categories:** Multi-Hop Chain Reasoning, Entity Relationship Reasoning, System Level / Architectural Understanding, Single-Document Fact Lookup, Entity-Specific Attribute Retrieval, Multi-Document Aggregation

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

## 5. Conclusions

1. **Overall winner:** MAS (overall score diff = +0.038)
2. **Metric-level:** MAS wins on 3 metrics, GraphRAG on 1.
3. **Largest gaps:** AnswerRelevancy.
4. **Paired t-test (all scores):** statistically significant (p = 0.0140).

