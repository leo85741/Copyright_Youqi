## LexisNexis Case Document Parser

This program parses HTML case documents exported from LexisNexis and converts the extracted case information into a JSON output file.

In addition, the program automatically generates a citation based on the extracted `case_title`, `reporters`, `case_court`, and `case_date`. The year is extracted from the date field using a regular expression and then combined into a LexisNexis-style citation.

### Input

The input is an HTML case document exported from LexisNexis, for example:

```text
lexis_document.html
```

The HTML file path can be specified using the `--input` argument:

```bash
python lexis_html_parser.py \
  --input "lexis_document.html" \
  --output "lexis_document_parsed.json"
```

### Output

The program outputs the extracted results as a JSON file, for example:

```text
lexis_document_parsed.json
```

The output JSON file contains the following fields:

| Field Name | Description |
|---|---|
| `case_no` | Case number |
| `case_title` | Case title |
| `case_date` | Decision date or case date information |
| `case_court` | Court information |
| `reporters` | Reporter citation information |
| `prior_history` | Prior history of the case |
| `core_terms` | Core terms provided by LexisNexis |
| `case_summary` | Case summary, including `procedural_posture`, `overview`, and `outcome` |
| `counsel` | Counsel or attorney information |
| `opinion_by` | Information about the judge who wrote the opinion |
| `opinion` | Full opinion text or main body of the decision |
| `citation` | Automatically generated LexisNexis-style citation |
