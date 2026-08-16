# CookSmart: Hybrid Recipe Recommendation System

## Executive summary

This project upgrades an earlier recipe-rating regression analysis into a personalized recommendation system. The original target — a global average rating — did not reflect the fact that different users value different recipes. CookSmart instead ranks unseen recipes a user is likely to enjoy.

The project compares popularity, content-based, LightFM, and weighted-hybrid approaches. The validation-selected weighted hybrid achieved the strongest final result: **Recall@10 = 0.3671** and **NDCG@10 = 0.2311**.

## Problem framing

The revised task is: given a user's historical positive interactions, rank unseen recipes by the likelihood that the user will like them. Ratings of 4 or 5 are positive feedback. Rating 0 is not interpreted as a negative preference because it is treated as missing or invalid explicit feedback.

## Data

| Dataset | Purpose |
| --- | --- |
| `RAW_recipes.csv` | Recipe title, description, tags, ingredients, steps, nutrition, and submission date |
| `RAW_interactions.csv` | User ID, recipe ID, interaction date, rating, and review |

After matching interactions to recipes with available metadata, the analysis uses 83,782 recipes and 234,428 usable interactions. The ranking evaluation contains 10,404 users with at least two positive interactions.

## Data preparation

### Recipe features

- Parse tags, ingredients, steps, and nutrition from serialized strings.
- Expand nutrition into numeric features such as calories, fat, sugar, sodium, and protein.
- Preserve a missing-description indicator rather than dropping records.
- Combine title, description, tags, and ingredients into `recipe_text` for content modeling.

### Interaction features

- Parse interaction dates.
- Retain valid explicit ratings from 1 to 5.
- Create a binary `liked` target for ratings of 4 or 5.
- Keep only interactions that match a recipe with available metadata.

## Evaluation design

The experiment uses a time-based split. For each user with two or more positive interactions, the most recent liked recipe is held out as the test target; earlier liked recipes form the training history. Previously interacted recipes are excluded from recommendations.

Each model ranks one held-out liked recipe against 99 unseen candidates. The metrics are:

- **Recall@10**: whether the held-out liked recipe appears in the top 10.
- **NDCG@10**: rewards ranking the held-out recipe closer to the top.

This sampled-candidate setting provides an efficient, like-for-like offline comparison. It should not be interpreted as a full-catalog retrieval score.

## Models

### Popularity Baseline

Ranks recipes by the number of positive training interactions. It is deliberately non-personalized and provides a strong reference under sparse feedback.

### Content-based Baseline

Transforms recipe title, description, tags, and ingredients with TF-IDF. A user profile is the normalized aggregate of their previously liked recipes; candidates are ranked by cosine similarity.

### LightFM hybrid experiment

LightFM learns from the user–recipe interaction matrix and recipe tags/ingredients. It uses WARP loss to optimize the position of positive recipes in the recommendation list.

### Weighted Hybrid Recommender

The final model combines normalized content and popularity scores:

```text
hybrid score = α × content similarity + (1 − α) × popularity score
```

The value of α is selected on a validation split created from training interactions. The untouched test split is used only for the final result.

## Results

| Model | Recall@10 | NDCG@10 | Finding |
| --- | ---: | ---: | --- |
| Popularity Baseline | 0.3032 | 0.1823 | Strong reference under sparse feedback |
| Content-based Baseline | 0.2719 | 0.1655 | Captures recipe similarity but lacks global preference signal |
| **Weighted Hybrid (content = 0.5)** | **0.3671** | **0.2311** | Best model; combines global and individual signals |

The notebook also evaluates LightFM on the same candidate-generation protocol. The key lesson is that a more complex personalized model does not automatically beat a strong popularity baseline when users have short interaction histories.

## Interpretation

Popularity and content are complementary. Content alone is weaker because similar ingredients and tags do not guarantee that users rate two recipes similarly. Popularity alone misses individual taste. The weighted hybrid improves both metrics by balancing these signals.

This suggests a practical product strategy: use high-quality popular recipes as a fallback for new or low-history users, while incorporating content preference where useful history exists.

## Limitations and future work

- Interaction data is sparse and ratings are concentrated toward the high end.
- Evaluation uses 99 sampled candidates rather than the complete catalogue.
- The data lacks demographic and session-level user context.
- TF-IDF does not capture all semantic relationships between recipes.

Future work: add sentence embeddings for recipe text, evaluate full-catalog retrieval on a user subset, tune LightFM only on validation data, and build a two-stage candidate-generation plus reranking service.

## Reproducibility

See [CookSmart_Hybrid_Recipe_Recommendation.ipynb](../CookSmart_Hybrid_Recipe_Recommendation.ipynb) for the complete, executable workflow.
