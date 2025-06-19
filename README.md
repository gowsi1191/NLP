# NLP

Implicit
 18-7 wins 16-15 why wins  more for 16 but mean is sum of mean is high  because 18 might win less when it wins 16 performs very poor so that mean score is less for 16 and 18 is more 
        "thresh_0.110_steep_16.5": 0.46212121212121215,
        "thresh_0.110_steep_18.0": 0.553030303030303
 for implict final report
       "config_distribution_implicit_not": {
        "thresh_0.110_steep_16.5": 15,
        "thresh_0.110_steep_18.0": 7
      }

comparitive
Consider using Roberta if P@1 and MRR are prioritized

Use BGE if P@2, P@3 or NDCG@3 are more important

Implement configuration thresh_0.120_steep_13.0 as the default, as it provides the best balanced performance

The high p-values suggest either model could be used without significant performance difference

scope
Use BGE for "scope_not" queries:

Superior precision (P@1, P@2, P@3) and MRR.

Roberta is not competitive for this query type.

Adopt thresh_0.100_steep_17.0 as default:

Best composite score, though gains over other configs are marginal.

Investigate why Roberta underperforms:

Fine-tune Roberta or explore hybrid approaches if BGE is resource-intensive.

Expand evaluation dataset:

Only 11 documents were analyzed; more data could reveal stronger trends.


prohibition
Use Roberta if your priority is first-result accuracy (e.g., search engines, chatbots).

Consider BGE if top-3 consistency matters more (e.g., recommendation systems).

Test DeBERTa-v3 if you suspect complex negations (e.g., nested "not only...but also")—Roberta may already suffice for simple prohibitions.