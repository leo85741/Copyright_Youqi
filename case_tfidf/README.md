## TF-IDF Calculation for Cited Cases in Court Decisions

The program filters records where `category` = `Cases` from the `urls_dic` field in each document, and uses `link` as the unique identifier for each cited case.

First, the program calculates the **TF** and **DF** for each case link. TF represents the total number of times a case is cited across all documents, while DF represents the number of distinct documents in which the case appears. To avoid inflating DF due to repeated citations of the same case within a single document, the calculation first groups records by “document ID + case link.”

Next, the program selects a representative `raw_text` for each case link as the output citation text. It prioritizes `raw_text` entries with more than 5 words; if multiple candidates exist, the one with the highest frequency is selected. This step is only used for display purposes and does not affect the TF-IDF calculation.

Finally, the IDF and TF-IDF values for each case are calculated based on the total number of documents `N`:

$$
IDF = \log\left(\frac{N}{DF}\right)
$$

$$
TF\_IDF = TF \times IDF
$$

The results are sorted in descending order by `TF_IDF` score and exported as `cases_TF_DF_TFIDF_clean.csv`.

| Field Name | Description |
|---|---|
| `link` | The cited case link |
| `TF` | The total number of times the case is cited across all documents |
| `DF` | The number of distinct documents in which the case appears |
| `raw_text` | The representative citation text |
| `category` | The citation category, which is `Cases` in this task |
| `source_signal` | The source signal from the original data |
| `confidence` | The confidence score from the original data |
| `rule_id` | The rule ID from the original data |
| `IDF` | The inverse document frequency calculated from the total number of documents and DF |
| `TF_IDF` | The final importance score of the cited case |
