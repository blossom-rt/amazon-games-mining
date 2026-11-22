# Manual Review of Association Rules

## Background

`also_buy` is a recommendation slot on product pages, not a cash-register receipt. P3 derived 1,855 raw rules from 19,032 baskets; after `lift>1` and symmetric deduplication, 932 remained. This review adds two kinds of information:

- `conviction`: the counterfactual strength with which the consequent fails to appear when the antecedent appears; the higher the value, the closer the rule is to "the antecedent almost necessarily implies the consequent";
- `same_series`: after the platform, edition, and trailing numbering are stripped from the title, the first three content words are taken to decide whether the antecedent and consequent belong to the same series.

## Review Results

- Of the 932 rules, 34 are same-series, with a median `conviction` of 3.21;
- The top 8 rules are all hidden-object download-edition cross-promotions, with lift 122–134 and conviction 6.9–31.8;
- Three support/confidence thresholds: 0.02/0.15 → 0 rules; 0.01/0.20 → 21 rules; 0.005/0.10 → 932 rules;
- The rules cover only 755 frequent products; long-tail products have no rules.

## Assessment

1. High conviction does not equal a real purchase. It only shows that the antecedent and consequent are tightly bound in the `also_buy` slot, and it is still shaped by the platform's recommendation strategy.
2. Same-series rules suit combined recommendations for "series pages/bundle pages"; they should not be interpreted as cross-category purchases.
3. Most hidden-object download editions in the top rules come from the same publisher bulk-filling recommendation slots; a rule such as `Euro Trip II → House Season` has lift 134 but support of only 0.58%, so it cannot represent the overall market.
4. Raising the support threshold to 0.02 leaves 0 rules, showing that strong rules exist only in niche clusters; mainstream product pairs need lower support, at the cost of an explosion in the number of rules.

## Recommendations

- Write business interpretations under three headings — "same-series complements", "same-platform fan overlap", and "closed download-edition clusters" — and never "buying A means buying B";
- If the rules are used for recommendation, first group by `same_series` and then rank within each group, to avoid cross-category mis-recommendations;
- Real purchase baskets require order data; before order data is available, `also_buy` conclusions serve only as recommendation-slot analysis.

## Artifacts

- `results/tables/rules_top.csv`: 932 rules, including `conviction` and `same_series`;
- `results/tables/rules_summary.json`: counts, thresholds, runtime, and assertions;
- `results/tables/rules_thresholds.csv`, `rules_runtime.csv`: threshold sensitivity and algorithm runtime.
