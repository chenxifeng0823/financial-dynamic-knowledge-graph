# FinDKG Dataset Summary

## Overview

The Financial Dynamic Knowledge Graph (FinDKG) dataset is a temporal knowledge graph focused on global financial entities and their relationships over time.

## Dataset Statistics

### Size
- **Total Entities**: 13,645
- **Total Relations**: 15 types
- **Total Triplets**: 144,062
  - Training: 119,549 (82.98%)
  - Validation: 11,444 (7.94%)
  - Test: 13,069 (9.07%)

### Temporal Scope
- **Time Range**: 126 time steps (0-125)
- **Average Triplets per Time Step**: 1,143.35
- **Temporal Unit**: Weekly aggregation of events

## Entity Types

The dataset contains 12 different entity types representing various financial and economic actors:

| Entity Type | Count | Percentage |
|-------------|-------|------------|
| PERSON | 3,088 | 22.63% |
| CONCEPT | 2,807 | 20.57% |
| COMP (Company) | 1,431 | 10.49% |
| ORG (Organization) | 1,245 | 9.12% |
| EVENT | 1,068 | 7.83% |
| GPE (Geo-Political Entity) | 936 | 6.86% |
| ORG/GOV (Government Org) | 759 | 5.56% |
| PRODUCT | 699 | 5.12% |
| SECTOR | 584 | 4.28% |
| FIN_INSTRUMENT | 533 | 3.91% |
| ECON_INDICATOR | 436 | 3.20% |
| ORG/REG (Regulatory Org) | 59 | 0.43% |

## Relation Types

The dataset includes 15 semantic relation types:

1. **Relate_To** (27.55%) - General relationships
2. **Control** (22.06%) - Control/ownership relationships
3. **Operate_In** (12.96%) - Operational presence
4. **Impact** (10.97%) - General impact relationships
5. **Has** (7.59%) - Possession/attribute relationships
6. **Raise** (2.75%) - Increase actions
7. **Negative_Impact_On** (2.70%) - Adverse effects
8. **Is_Member_Of** (2.64%) - Membership relationships
9. **Announce** (2.15%) - Announcements/declarations
10. **Participates_In** (2.04%) - Participation relationships
11. **Invests_In** (1.83%) - Investment relationships
12. **Introduce** (1.36%) - Introduction/launch actions
13. **Decrease** (1.25%) - Decrease actions
14. **Produce** (1.24%) - Production relationships
15. **Positive_Impact_On** (0.92%) - Beneficial effects

## Notable Entities

Top entities include major financial and political actors:
- Donald Trump, Joe Biden (Political figures)
- United States, China, Russia (Countries)
- U.S. Federal Reserve (Regulatory body)
- Apple Inc., Meta Platforms Inc. (Tech companies)
- COVID-19 (Major event)

## Data Format

### Triplet Structure
Each triplet in the dataset follows the format:
```
subject_id  relation_id  object_id  timestamp  [ignored]
```

Example triplets:
1. `[President Trump Administration] --Control--> [Volcker rule]` at time 0
2. `[Wells Fargo Co.] --Impact--> [U.S. Federal Reserve]` at time 0
3. `[Australia] --Participates_In--> [S&P/ASX 200]` at time 0

### Files
- `train.txt` - Training set (chronologically first events)
- `valid.txt` - Validation set (middle period)
- `test.txt` - Test set (chronologically latest events)
- `entity2id.txt` - Entity name to ID mapping with types
- `relation2id.txt` - Relation name to ID mapping
- `stat.txt` - Dataset statistics (entities, relations, timestamps)

## Dataset Variants

### FinDKG (Default)
- Standard dataset for research
- 13,645 entities
- 144,062 triplets

### FinDKG-full
- Extended dataset with more events
- 13,012 entities
- ~230,000 triplets
- Includes `time2id.txt` mapping for real-world dates

## Use Cases

This dataset is suitable for:
1. **Temporal Link Prediction** - Predicting future relationships between entities
2. **Anomaly Detection** - Identifying unusual patterns in financial networks
3. **Event Forecasting** - Predicting financial events based on historical patterns
4. **Network Analysis** - Understanding structure of global financial networks
5. **Relationship Evolution** - Tracking how financial relationships change over time

## Citation

Original repository: [xiaohui-victor-li/FinDKG](https://github.com/xiaohui-victor-li/FinDKG)

Website: https://xiaohui-victor-li.github.io/FinDKG/

