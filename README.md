## Statutory Damages Amount Extraction

This project uses court judgment documents as the data source. Through Few-Shot Prompting and a large language model, it automatically extracts statutory damages amounts related to copyright issues and compares the results with human annotations for performance evaluation.

### 1. Data Source and Preprocessing

- The research data consists of court judgment documents stored in a MongoDB database.
- For each document, paragraphs classified as *copyright* are selected from the `content_clean_statute_new` field and concatenated into a single input text.
- To handle structured or nested JSON data, a flattening method is used to convert strings, lists, and dictionaries into unified text paragraphs.

### 2. Few-Shot Prompting

- The extraction process uses the LangChain framework and the OpenAI `gpt-5` model.
- Representative samples from each cluster in the clustering analysis results are used to construct a **few-shot prompt**. The prompt includes several examples, each consisting of an input/output pair, to demonstrate how to extract only the numerical value of the statutory damages amount from the text.
- Prompt rules:
  - Output only plain numbers, with `$` signs and commas removed.
  - If the document does not mention a statutory damages amount, output `N/A`.
  - Do not output any additional text or unnecessary content.

### 3. Automated Extraction

- The processed text of each document is passed to the LLM to obtain the predicted damages amount.
- The model output is then normalized by converting formats such as `$`, `USD`, decimals, and commas into integer strings.

### 4. True Label Annotation Comparison

- Human annotations are stored in the `annotation_GT_LW` field, with the main key being **Statutory Damages Award (Y)**.
- If the annotation structure is complex, the correct field is identified through case-insensitive matching and keyword-based fuzzy matching.
- The annotated amount is also normalized.

### 5. Performance Evaluation

**Definition of valid samples**: A sample is considered valid if both the annotated amount and the model-predicted amount are present, or if both are `N/A`, meaning the label is `N/A` and the prediction is correct.

#### Accuracy Calculation

- **Valid sample accuracy**: Calculated only on valid samples by checking whether the model prediction exactly matches the label.
- **Overall accuracy**: Calculated as the proportion of samples, excluding unprocessable documents, where the model prediction matches the label.

#### Experimental Results

| Metric | Result |
|---|---:|
| Number of evaluable samples | 44 / 48 |
| Valid sample accuracy | 93.18% |
| Overall accuracy | 85.42% |

### 6. Output

The final extraction results are saved as a **CSV file**. Each record corresponds to one court document and includes the following fields:

| Field | Description |
|---|---|
| `_id` | Unique identifier of the MongoDB document |
| `content_for_pred` | Actual input text used by the model, containing only paragraphs where `statute = copyright` |
| `predicted_raw` | Raw LLM output, including the amount and Evidence |
| `predicted_award` | Amount parsed from the raw output, as a plain number or `N/A` |
| `predicted_reason` | Evidence returned by the model |
| `manifest.json` | Output record and analysis configuration, including the selected k value |
| `true_award` | Human-annotated true amount, Statutory Damages Award |
| `predicted_award_single` | Normalized predicted amount used for comparison |
| `award_match` | Whether the prediction exactly matches the label, `True` / `False` |

---

## Few-Shot Clustering

This section describes the clustering process used to select representative samples for constructing the few-shot prompt.

### 1. Data Source and Preprocessing

The data is stored in MongoDB.

- Database: `copyright`
- Collection: `RST_Preprocessed_SBS`

Each document contains a paragraph-level annotation field named `LLMOUT_SectionLab`, where each paragraph is assigned a label such as *Facts*, *Analysis*, *Order/Summary*, and others.

This program automatically extracts paragraphs labeled as **"Order/Summary"** and removes blank or missing values to form the text corpus for analysis.

### 2. Sentence Embedding

To preserve the semantic characteristics of legal text, the **Legal-BERT** model, `nlpaueb/legal-bert-base-uncased`, is used to convert each text segment into a vector representation.

Because the model is pretrained on legal corpora, it can capture domain-specific features such as statutory language, judicial wording, and decision-related expressions.

After encoding with Legal-BERT, each text segment is represented as a 768-dimensional sentence vector, which is used for subsequent clustering analysis.

### 3. Clustering

The **K-Means algorithm** is applied to all Order/Summary vectors to identify semantically similar types of judicial outcomes.

To determine the optimal number of clusters, *k*, the **Elbow Method** is first used to observe changes in inertia, also known as SSE, under different k values. **KneeLocator** is then used to automatically identify the optimal turning point.

If the automatic detection fails, the default value is set to *k = 6*.

The cosine similarity between each sample and its cluster centroid is also calculated. This allows the program to select the sample that best represents the semantic center of each cluster as the **representative sample**.

### 4. Dimensionality Reduction and UMAP Visualization

To support manual inspection of the clustering results, **UMAP**, or Uniform Manifold Approximation and Projection, is used to reduce the high-dimensional sentence vectors into two dimensions and generate a scatter plot.

Different clusters are distinguished by color, while representative samples within each cluster are marked with a black cross, `×`.

This visualization helps reveal the distribution of different judicial language patterns.

### 5. Cluster Keyword Extraction

To analyze the textual characteristics of each cluster, **spaCy** is used for lemmatization, and a **TF-IDF** vectorization model is used to calculate term weights.

For each cluster, the top 30 keywords with the highest average TF-IDF scores are selected to represent the main semantic themes of that cluster.

### 6. Outputs

After execution, the program automatically generates the following output files. By default, they are saved in the `./outputs` folder.

| File Name | Description |
|---|---|
| `elbow_plot.png` | Inertia curve under different numbers of clusters, used to inspect the optimal k value |
| `umap_clusters.png` | Two-dimensional UMAP visualization of clustering results |
| `representatives.csv` | Most representative samples from each cluster |
| `cluster_summary.csv` | Cluster size, average similarity, and representative sample summary for each cluster |
| `cluster_keywords.csv` | Top 30 TF-IDF keywords for each cluster |
| `manifest.json` | Output record and analysis configuration, including the selected k value |
