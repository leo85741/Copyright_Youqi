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
