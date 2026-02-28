# DeepEval Comparison Report: MAS vs GraphRAG

**Total queries:** 19
**Categories:** Multi-Document Aggregation, Single-Document Fact Lookup

---

## 1. Per-Metric Summary


| Metric          | MAS Mean | MAS Std | GR Mean | GR Std | MAS Median | GR Median | Diff (MAS-GR) |
| --------------- | -------- | ------- | ------- | ------ | ---------- | --------- | ------------- |
| AnswerRelevancy | 0.914    | 0.132   | 0.774   | 0.235  | 1.000      | 0.857     | +0.141        |
| Bias            | 1.000    | 0.000   | 1.000   | 0.000  | 1.000      | 1.000     | +0.000        |
| Clarity         | 0.863    | 0.201   | 0.795   | 0.182  | 1.000      | 0.900     | +0.068        |
| Completeness    | 0.974    | 0.044   | 0.926   | 0.217  | 1.000      | 1.000     | +0.047        |
| Tone            | 0.989    | 0.031   | 1.000   | 0.000  | 1.000      | 1.000     | -0.011        |
| Toxicity        | 1.000    | 0.000   | 1.000   | 0.000  | 1.000      | 1.000     | +0.000        |


---

## 2. Overall Scores (Weighted Mean)

Weights: {'AnswerRelevancy': 0.3, 'Clarity': 0.2, 'Completeness': 0.25, 'Tone': 0.15, 'Bias': 0.05, 'Toxicity': 0.05}

- **MAS overall:** 0.939
- **GraphRAG overall:** 0.873
- **Difference (MAS - GraphRAG):** +0.066

---

## 3. Statistical Comparison

### Per-metric paired difference (MAS - GraphRAG)


| Metric          | Mean Diff | t               | Cohen's d |
| --------------- | --------- | --------------- | --------- |
| AnswerRelevancy | +0.141    | 2.58 (p~0.019)  | 0.59      |
| Bias            | +0.000    | 0.00 (p~1.000)  | 0.00      |
| Clarity         | +0.068    | 1.51 (p~0.148)  | 0.35      |
| Completeness    | +0.047    | 0.87 (p~0.394)  | 0.20      |
| Tone            | -0.011    | -1.46 (p~0.163) | -0.33     |
| Toxicity        | +0.000    | 0.00 (p~1.000)  | 0.00      |


---

## 4. Pass Rates


| Metric          | MAS Pass | MAS Rate | GR Pass | GR Rate |
| --------------- | -------- | -------- | ------- | ------- |
| AnswerRelevancy | 19/19    | 100.0%   | 16/19   | 84.2%   |
| Bias            | 0/19     | 0.0%     | 0/19    | 0.0%    |
| Clarity         | 17/19    | 89.5%    | 17/19   | 89.5%   |
| Completeness    | 19/19    | 100.0%   | 17/19   | 89.5%   |
| Tone            | 19/19    | 100.0%   | 19/19   | 100.0%  |
| Toxicity        | 0/19     | 0.0%     | 0/19    | 0.0%    |


---

## 5. Conclusions

1. **Overall winner:** MAS (overall score diff = +0.066)
2. **Metric-level:** MAS wins on 3 metrics, GraphRAG on 1.
3. **Largest gaps:** AnswerRelevancy.
4. **Paired t-test (all scores):** statistically significant (p = 0.0090).

