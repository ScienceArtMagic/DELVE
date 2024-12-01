---
license:
- apache-2.0
- mit
language:
- multilingual
library_name: transformers
pipeline_tag: text-generation
tags:
- mamba
- pythia
- rwkv
- openlm
- supra
datasets:
- EleutherAI/pile
- cerebras/SlimPajama-627B
- oscar-corpus/oscar
- togethercomputer/RedPajama-Data-V2
- tiiuae/falcon-refinedweb
- bigcode/the-stack-dedup
- bigcode/the-stack-v2-dedup
- OpenCoder-LLM/fineweb-code-corpus
- codeparrot/github-code-clean
- opencsg/chinese-fineweb-edu
- opencsg/chinese-fineweb-edu-v2
- HuggingFaceFW/fineweb
- HuggingFaceFW/fineweb-edu
- OpenCoder-LLM/fineweb-math-corpus
- allenai/dolma
base_model:
- state-spaces/mamba-370m-hf
- state-spaces/mamba-1.4b-hf
- TRI-ML/mamba-7b-rw
- RWKV/rwkv-4-430m-pile
- RWKV/rwkv-4-1b5-pile
- RWKV/rwkv-4-7b-pile
# - TRI-ML/openlm-7b-code
- EleutherAI/pythia-410m-deduped
- EleutherAI/pythia-1b-deduped
- EleutherAI/pythia-1.4b-deduped
- EleutherAI/pythia-6.9b-deduped
---

# DELVE: Diminutive Experts Leverage Voluminous Expansion

##  Table of Contents

- [Model Details](#model-details)
  - [Model Description](#model-description)
- [Uses](#uses)
  - [Direct Use](#direct-use)
  - [Downstream Use [Optional]](#downstream-use-optional)
  - [Out-of-Scope Use](#out-of-scope-use)
- [Bias, Risks, and Limitations](#bias-risks-and-limitations)
  - [Recommendations](#recommendations)
- [Training Details](#training-details)
  - [Training Data](#training-data)
  - [Training Procedure](#training-procedure)
    - [Preprocessing](#preprocessing)
    - [Speeds, Sizes, Times](#speeds-sizes-times)
- [Evaluation](#evaluation)
  - [Testing Data, Factors & Metrics](#testing-data-factors--metrics)
    - [Testing Data](#testing-data)
    - [Factors](#factors)
    - [Metrics](#metrics)
  - [Results](#results)
- [Model Examination](#model-examination)
- [Technical Specifications [optional]](#technical-specifications-optional)
  - [Model Architecture and Objective](#model-architecture-and-objective)
  - [Compute Infrastructure](#compute-infrastructure)
    - [Hardware](#hardware)
    - [Software](#software)
- [Citation](#citation)
- [Glossary [optional]](#glossary-optional)
- [More Information [optional]](#more-information-optional)
- [Model Card Authors [optional]](#model-card-authors-optional)
- [Model Card Contact](#model-card-contact)
- [How to Get Started with the Model](#how-to-get-started-with-the-model)


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is/does. -->


- **Developed by:** 🧪🖌️🪄
- **Model type:** Language model
- **Language(s) (NLP):** mul
- **License:** MIT or Apache 2.0 (your choice)
- **Parent Model:** Mamba v1, RWKV v4, Pythia, OpenLM
- **Resources for more information:** More information needed



## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->
<!-- If the user enters content, print that. If not, but they enter a task in the list, use that. If neither, say "more info needed." -->




### Downstream Use [Optional]

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->
<!-- If the user enters content, print that. If not, but they enter a task in the list, use that. If neither, say "more info needed." -->

If you're fine-tuning with e.g. LoRA, note that the linear modules from Mamba, RWKV v4, Pythia, have already been approximated as low-rank submodules (the biggest... well, factor... in making DELVE so small).

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->
<!-- If the user enters content, print that. If not, but they enter a task in the list, use that. If neither, say "more info needed." -->

I neither know nor care whether or not this model will make a good spicy waifu.

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Significant research has explored bias and fairness issues with language models (see, e.g., [Sheng et al. (2021)](https://aclanthology.org/2021.acl-long.330.pdf) and [Bender et al. (2021)](https://dl.acm.org/doi/pdf/10.1145/3442188.3445922)). Predictions generated by the model may include disturbing and harmful stereotypes across protected classes; identity characteristics; and sensitive, social, and occupational groups.

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Hopefully the model won't be a douche... It is trained on the internet though, so...

## Training Details

### Training Data

<!-- This should link to a Data Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

More information on training data needed

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing

More information needed

#### Speeds, Sizes, Times

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

More information needed
 
## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Data Card if possible. -->

More information needed

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

More information needed

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

More information needed

### Results 

More information needed

## Model Examination

More information needed

## Technical Specifications [optional]

### Model Architecture and Objective

Autoregressive hybrid of Mamba v1 SSM, RWKV v4 RNN, and two decoder-only Transformer architectures (Pythia and OpenLM) - all based on the GPT-NeoX 20b tokenizer.

Combined single model upcycled from these individual pretrained models, after each goes through SVD low-rank approximation for extreme parameter reduction.

### Compute Infrastructure

More information needed

#### Hardware

More information needed

#### Software

More information needed

## Citation

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

More information needed

**APA:**

More information needed

## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

More information needed

## More Information [optional]

More information needed

## Model Card Authors [optional]

<!-- This section provides another layer of transparency and accountability. Whose views is this model card representing? How many voices were included in its construction? Etc. -->

[@stereoplegic](https://huggingface.co/stereoplegic)

## Model Card Contact

[🤗 stereoplegic](https://huggingface.co/stereoplegic)
[🦋 ScienceArtMagic](https://ScienceArtMagic.bsky.social)

## How to Get Started with the Model

Use the code below to get started with the model.

<details>
<summary> Click to expand </summary>

More information needed

</details>