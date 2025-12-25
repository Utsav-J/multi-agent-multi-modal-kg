# LLM Request Analysis for Validator Agent

## Current LLM Request Breakdown

### 1. CoverageEvaluator
- **Location**: `CoverageEvaluator.evaluate()`
- **Default sample_size**: 10 chunks
- **LLM calls per chunk**: 2 (entities + relations)
- **Total LLM calls**: 20 requests
- **Code locations**:
  - Line 514: `_extract_entities_from_chunk()` - extracts entities from chunk
  - Line 515: `_extract_relations_from_chunk()` - extracts relations from chunk

### 2. FaithfulnessEvaluator
- **Location**: `FaithfulnessEvaluator.evaluate()`
- **Default sample_size**: 50 nodes + 50 relations
- **LLM calls per node**: 1 (check grounding)
- **LLM calls per relation**: 1 (check grounding)
- **Total LLM calls**: 100 requests
- **Code locations**:
  - Line 870: `_check_node_grounding()` - checks if node is grounded in text
  - Line 884: `_check_relation_grounding()` - checks if relation is grounded in text

### 3. SemanticEvaluator
- **Location**: `SemanticEvaluator.evaluate()`
- **Default sample_size**: 100 relations
- **LLM calls per relation**: 1 (check plausibility)
- **Total LLM calls**: 100 requests
- **Code locations**:
  - Line 1009: `_check_relation_plausibility()` - checks if relation is semantically plausible

## Total LLM Requests: **220 requests** (with default settings)

All requests are made **sequentially** without rate limiting or delays, which can cause:
- API rate limit errors
- Overwhelming the API with too many concurrent requests
- Slow execution times

## Optimization Opportunities

### High Impact Optimizations

1. **Combine Entity and Relation Extraction** (CoverageEvaluator)
   - **Current**: 2 separate LLM calls per chunk
   - **Optimization**: Combine into 1 LLM call that returns both entities and relations
   - **Savings**: ~50% reduction (10 requests instead of 20)

2. **Reduce Default Sample Sizes**
   - **CoverageEvaluator**: 10 → 5 chunks (saves 10 requests)
   - **FaithfulnessEvaluator**: 50 → 25 nodes + 25 relations (saves 50 requests)
   - **SemanticEvaluator**: 100 → 50 relations (saves 50 requests)
   - **Total Savings**: ~110 requests (50% reduction)

3. **Make SemanticEvaluator Optional/Reduced Priority**
   - Semantic plausibility is less critical than coverage/faithfulness
   - Could be run less frequently or with smaller samples
   - **Savings**: Up to 100 requests (skip entirely) or 50 requests (halve sample)

### Medium Impact Optimizations

4. **Add Rate Limiting**
   - Add delays between requests (e.g., 0.1-0.5 seconds)
   - Use exponential backoff on errors
   - Prevents API throttling

5. **Batch Processing with Progress Tracking**
   - Process in smaller batches with delays between batches
   - Better error handling and recovery

6. **Parallel Processing with Rate Limiting**
   - Use asyncio or threading with semaphore to limit concurrent requests
   - Could speed up execution while respecting rate limits
   - Requires careful implementation to avoid overwhelming API

### Recommended Quick Wins

1. ✅ **Combine entity/relation extraction in CoverageEvaluator** (saves 10 requests)
2. ✅ **Reduce default sample sizes** (saves 110 requests)
3. ✅ **Add rate limiting** (prevents errors, doesn't reduce total but improves reliability)
4. ⚠️ **Make SemanticEvaluator optional** (saves 50-100 requests, but loses that metric)

### Recommended Configuration

After optimizations:
- **CoverageEvaluator**: 5 chunks × 1 LLM call = 5 requests (was 20)
- **FaithfulnessEvaluator**: 25 nodes + 25 relations = 50 requests (was 100)
- **SemanticEvaluator**: 50 relations = 50 requests (was 100)
- **New Total**: ~105 requests (52% reduction from 220)

