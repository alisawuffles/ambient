# AmbiEnt

This repository contains the code and data for [We're Afraid Language Models Aren't Modeling Ambiguity](https://arxiv.org/abs/2304.14399).

If you have any questions, please feel free to create a Github issue or reach out to the first author at [alisaliu@cs.washington.edu](alisaliu@cs.washington.edu).

## Summary of code

**§2 Creating AmbiEnt |** AmbiEnt includes a small set of author-curated examples (§2.1), plus a larger collection of examples created through overgeneration-and-filtering of unlabeled examples followed by linguist annotation (§2.2-2.3). The code for generating and filtering unlabeled examples is in `generation/`; code for preparing batches for expert annotation and validation are in `notebooks/linguist_annotation`. The AmbiEnt dataset and all relevant annotations are in `AmbiEnt/`.

**§3 Does Ambiguity Explain Disagreement? |** In this section, we analyze how crowdworkers behave on ambiguous input under the traditional 3-way annotation scheme for NLI, which does not account for the possibility of ambiguity. Code for creating AMT batches and computing the results is in `notebooks/crowdworker_experiment`.

**§4 Evaluating Pretrained LMs |** In our experiments, we design a suite of tests based on AmbiEnt to evaluate whether LMs can recognize ambiguity and disentangle possible interpretations. All of the code for this is in `evaluation/`, and results files are in `results/`. Code for human evaluation of LM-generated disambiguations (§4.1) are in `notebooks/human_eval`.

**§5 Evaluating Multilabel NLI Models |** Next, we investigate the performance of NLI models tuned on existing NLI data that contests the 3-way categorization (e.g., examples with soft labels). Scripts for data preprocessing and training of a multilabel NLI model (§5) are in `classification/`; for other models, please see codebases from prior work or reach out to me with questions. Results files are also in `results/`.

**§6 Case Study: Detecting Misleading Political Claims |** This experiment is done in `notebooks/political_claims_case_study.ipynb`. You can find the author annotations of ambiguity in political claims in `political-claims/`, along with results from our detection method.

For examples of how scripts are used, please see `scripts/`.

## Citation
If our work is useful to you, you can cite us with the following BibTex entry!
```
@misc{liu-etal-2023-afraid,
    title = "We're Afraid Language Models Aren't Modeling Ambiguity",
    author = "Alisa Liu and Zhaofeng Wu and Julian Michael and Alane Suhr and Peter West and Alexander Koller and Swabha Swayamdipta and Noah A. Smith and Yejin Choi",
    month = apr,
    year = "2023",
    url = "https://arxiv.org/abs/2304.14399",
}
```