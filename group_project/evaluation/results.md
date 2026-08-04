# RAGAS Evaluation Results

- Generated: `2026-08-04T04:59:46.782733+00:00`
- Golden test cases: **15**
- Generation model: `gpt-4o-mini`
- RAGAS judge model: `gpt-4o-mini`
- Embedding model: `text-embedding-3-small`

## Overall A/B comparison

| Configuration | Faithfulness | Relevancy | Recall | Precision | Average |
|---|---:|---:|---:|---:|---:|
| hybrid_rrf | 0.944 | 0.880 | 1.000 | 0.915 | 0.935 |
| dense_only | 0.944 | 0.873 | 0.900 | 0.846 | 0.891 |

## Per-question scores

### hybrid_rrf

| # | Question | Faithfulness | Relevancy | Recall | Precision | Average |
|---:|---|---:|---:|---:|---:|---:|
| 1 | How many library items may undergraduate and postgraduate students borrow? | 1.000 | 0.964 | 1.000 | 0.500 | 0.866 |
| 2 | What are the loan period and renewal allowance for undergraduate students? | 1.000 | 0.623 | 1.000 | 1.000 | 0.906 |
| 3 | How can a student book a library study room at RMIT? | 1.000 | 0.953 | 1.000 | 1.000 | 0.988 |
| 4 | How far in advance can students book a library study room? | 0.667 | 1.000 | 1.000 | 0.583 | 0.812 |
| 5 | What is the maximum study-room booking duration and how many bookings may one person make? | 1.000 | 0.990 | 1.000 | 1.000 | 0.998 |
| 6 | What happens if a student does not check in for a study-room booking on time? | 1.000 | 0.800 | 1.000 | 0.806 | 0.901 |
| 7 | Does RMIT provide airport pick-up support for international students? | 1.000 | 0.877 | 1.000 | 1.000 | 0.969 |
| 8 | When are international students assigned a buddy, and how long does the support last? | 1.000 | 0.559 | 1.000 | 1.000 | 0.890 |
| 9 | What complimentary activities does RMIT offer international students? | 1.000 | 0.996 | 1.000 | 1.000 | 0.999 |
| 10 | What main academic support services are provided by Student Academic Success? | 1.000 | 0.872 | 1.000 | 1.000 | 0.968 |
| 11 | Where is the Writing and Learning Centre at Saigon South, and what are its opening hours? | 0.500 | 0.995 | 1.000 | 1.000 | 0.874 |
| 12 | When and where is the Learning Advisor Service available at the Hanoi campus? | 1.000 | 0.939 | 1.000 | 1.000 | 0.985 |
| 13 | What counselling support does RMIT CAPS provide to enrolled students? | 1.000 | 0.898 | 1.000 | 1.000 | 0.975 |
| 14 | What must a student provide to register with Equitable Learning and Accessibility? | 1.000 | 0.863 | 1.000 | 1.000 | 0.966 |
| 15 | When must University Pathway and Higher Education students pay their tuition fees? | 1.000 | 0.864 | 1.000 | 0.833 | 0.924 |

### dense_only

| # | Question | Faithfulness | Relevancy | Recall | Precision | Average |
|---:|---|---:|---:|---:|---:|---:|
| 1 | How many library items may undergraduate and postgraduate students borrow? | 1.000 | 0.964 | 1.000 | 1.000 | 0.991 |
| 2 | What are the loan period and renewal allowance for undergraduate students? | 1.000 | 0.623 | 1.000 | 1.000 | 0.906 |
| 3 | How can a student book a library study room at RMIT? | 1.000 | 1.000 | 1.000 | 0.804 | 0.951 |
| 4 | How far in advance can students book a library study room? | 0.500 | 1.000 | 1.000 | 0.750 | 0.812 |
| 5 | What is the maximum study-room booking duration and how many bookings may one person make? | 1.000 | 0.990 | 1.000 | 1.000 | 0.998 |
| 6 | What happens if a student does not check in for a study-room booking on time? | 1.000 | 0.800 | 1.000 | 0.750 | 0.888 |
| 7 | Does RMIT provide airport pick-up support for international students? | 1.000 | 0.888 | 1.000 | 1.000 | 0.972 |
| 8 | When are international students assigned a buddy, and how long does the support last? | 1.000 | 0.559 | 1.000 | 1.000 | 0.890 |
| 9 | What complimentary activities does RMIT offer international students? | 1.000 | 0.996 | 1.000 | 0.833 | 0.957 |
| 10 | What main academic support services are provided by Student Academic Success? | 1.000 | 0.872 | 1.000 | 0.887 | 0.940 |
| 11 | Where is the Writing and Learning Centre at Saigon South, and what are its opening hours? | 0.667 | 0.905 | 0.500 | 0.333 | 0.601 |
| 12 | When and where is the Learning Advisor Service available at the Hanoi campus? | 1.000 | 0.932 | 1.000 | 1.000 | 0.983 |
| 13 | What counselling support does RMIT CAPS provide to enrolled students? | 1.000 | 0.899 | 1.000 | 0.833 | 0.933 |
| 14 | What must a student provide to register with Equitable Learning and Accessibility? | 1.000 | 0.808 | 1.000 | 1.000 | 0.952 |
| 15 | When must University Pathway and Higher Education students pay their tuition fees? | 1.000 | 0.864 | 0.000 | 0.500 | 0.591 |

## Worst performers

| Configuration | Question | Average | Main weakness |
|---|---|---:|---|
| dense_only | When must University Pathway and Higher Education students pay their tuition fees? | 0.591 | context_recall (0.000) |
| dense_only | Where is the Writing and Learning Centre at Saigon South, and what are its opening hours? | 0.601 | context_precision (0.333) |
| dense_only | How far in advance can students book a library study room? | 0.812 | faithfulness (0.500) |
| hybrid_rrf | How far in advance can students book a library study room? | 0.812 | context_precision (0.583) |
| hybrid_rrf | How many library items may undergraduate and postgraduate students borrow? | 0.866 | context_precision (0.500) |

## Analysis and recommendations

- **hybrid_rrf** achieved the highest overall average in this run.
- Improve low context precision by reducing duplicated PDF chunks and adding Markdown header metadata to the chunker.
- Improve context recall by expanding Vietnamese/English query terms before dense and BM25 retrieval.
- Review the worst questions above and add targeted documents when the corpus lacks explicit evidence.
- Re-run this same golden set after retrieval, chunking, or prompt changes to detect regressions.
