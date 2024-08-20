# MLTRL Card - RFM-Analysis - level 1

| Summary info        | Content, links       |
| -------------------------- | ------------- |
| Tech name               | RFM-Analysis - Car Repair Shop   |
| Current Level           | 1  |
| Owner(s)                | Christian Wissor                        |
| Reviewer(s)             | Christian Wissor                           |
| Main project page       | *none*   |
| Req's, V&V docs         | Master-Thesis   |
| Code repo & docs        | [Github Repository](https://github.com/SmithyW/master-thesis-mlops-study-imlpementation)   |
| Ethics checks?          | *none* |
| Coupled components      | *none*         |


**TL;DR** — The RFM-Analysis is clustering project for segmenting customer of a small car repair shop. 


### Top-level requirements

*the underlying master-thesis serves as the full requirements documentation and is not publicly available*

1. The model should segment customers in a comporehensible way.
2. Integration into third-party systems for inference should be made possible via a suitable interface.

### Model info

The model is a standard K-Means clustering algorithm, where the number of segments has to be provided beforehand.

Implementation notes:

- Implementation leverages the python library scikit-learn. 

### Intended use

- The clustering algorithm will create the customer segments based on features for a conventional RFM-Analysis.
- The allocation is intended to be accomplished with the computed cluster centers and a yet unknown distance metric. 

### Testing status

*no tests are specified or planned for now*

**Extra notes**: The model can be seen as uncritical when considering safety aspects and the entrpreneurial risk.


### Data considerations

A production dataset is available for experimenting.

The implementation expects a specific format similar to csv, but with tab symbols as delimiter. For further processing the data is merged and stored as csv. The usage inside python leverages the pandas library.

The conversion of the raw invoice data to RFM-Features is implemented without further processing (for now).

The distribution of the data allows for scaling using sklearns StandardScaler.

### Caveats, known edge cases, recommendations

- Clustering the data does not yield fully satisfying results, but is considered promising.
- Consideration must be given to whether certain criteria should be introduced for the data.
- Further processing in terms of outlier detection is recommended.

### MLTRL stage debrief

<!-- Succinct summary of stage progress – please respond to each question, link to extended material if needed... -->

1. What was accomplished, and by who?

The data was prepared so that experiments could then be carried out iteratively and the properties of the data familiarized.

2. What was punted and/or de-scoped?

    n/a

3. What was learned?

    - The model does not fully yield the expected results; possibly to unsufficient processing
    - Through visualizing the data, the processing and filtering could lead to promising results.

4. What tech debt what gained? Mitigated?

    - Initialization of the project happened with DVC and MLflow in mind, so that versioning and tracking of experiments and data is ensured.

---
