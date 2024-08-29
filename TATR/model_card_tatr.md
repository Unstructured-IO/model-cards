---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards

---

# Model Card for microsoft/table-transformer-structure-recognition-v1.1-all

<!-- Provide a quick summary of what the model is/does. -->

Table Transformer (TATR) model trained on PubTables1M and FinTabNet.c for table structure recognition, introduced in the paper 'Aligning benchmark datasets for table structure recognition' by Smock et al. The model is designed to take table image as input and output bounding boxes of rows, columns, cells of the table. Data used for training contains mostly images of tables from science or financial documents.

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

The Table Transformer is a Transformer-based object detection model designed to recognize table structures (like rows and columns) in documents. It is trained on aligned datasets PubTables1M and FinTabNet using a canonicalization process to address label inconsistencies. The model is suitable for extracting table structures in unstructured documents.

- **Developed by:** Microsoft Research
- **Funded by [optional]:** Microsoft Research
- **Shared by [optional]:** Microsoft Research
- **Model type:** Object Detection
- **Language(s) (NLP):** N/A (Trained on Mostly English)
- **License:** MIT
- **Finetuned from model [optional]:** Table Transformer v1 (trained on the same datasets)

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** https://github.com/microsoft/table-transformer
- **Paper [optional]:** https://arxiv.org/abs/2301.12345
- **Demo [optional]:** Example (done by community): https://huggingface.co/spaces/nielsr/tatr-demo

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

Detecting table structures like rows and columns in documents.

### Downstream Use [optional]

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

Applied in document processing pipeline for parsing table image into HTML representation.

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

When applied on image without table can predict non existing structure.

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Potential inaccuracies when applied to unseen table layouts or document types not covered during training. Inconsistencies may arise when encountering heavily stylized or multi-layered tables.

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users should be aware of the model’s potential biases related to dataset-specific training. Users should also verify extracted results manually for critical use cases.

## How to Get Started with the Model

Use the code below to get started with the model.

```python
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
import torch
from PIL import Image

file_path = hf_hub_download(repo_id="nielsr/example-pdf", repo_type="dataset", filename="example_pdf.png")
image = Image.open(file_path).convert("RGB")

image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition-v1.1-all")
image_processor.size['shortest_edge'] = 800

model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition-v1.1-all")

inputs = image_processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
target_sizes = torch.tensor([image.size[::-1]])
results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[
    0
]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {model.config.id2label[label.item()]} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )
```


## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

- PubTables1M (https://huggingface.co/datasets/bsmock/pubtables-1m): 947,642 fully annotated tables including text content and complete location (bounding box) information for table structure recognition and functional analysis.
- FinTabNet (https://huggingface.co/datasets/bsmock/FinTabNet.c): 112,887 tables with cell structure complex tables from the annual reports of S&P 500 companies with detailed table structure annotations to help train and test structure recognition.


### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing [optional]

Data canonicalization procedure applied to make annotation consistent across datasets


#### Training Hyperparameters

- **Training regime:** Mixed precision (fp16) used for efficiency during training. <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

#### Speeds, Sizes, Times [optional]

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

[More Information Needed]

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

Test splits from table datasets:
  - PubTables1M (https://huggingface.co/datasets/bsmock/pubtables-1m)
  - FinTabNet (https://huggingface.co/datasets/bsmock/FinTabNet.c)
  - ICDAR-2013 (https://huggingface.co/datasets/bsmock/ICDAR-2013.c)


#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

Evaluated across different table layouts and document types, disaggregating performance based on table complexity and structural variation.

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

- Typical object detection metrics mAP (mean Average Precision)
- Specialized object detection metrics crafted for table structure analysis like GriTS (https://arxiv.org/abs/2203.12555) which analyses how well the structure is preserved in the predicted table


### Results

The model outperforms previous versions in terms of mAP and robustness to label inconsistencies, achieving over 90% mAP on aligned datasets.

#### Summary

Significant improvements in table structure recognition due to enhanced dataset alignment

## Model Examination [optional]

<!-- Relevant interpretability work for the model goes here -->

[More Information Needed]

## Environmental Impact [optional]

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** NVidia Tesla V100 GPU (information available for V1 TATR, not available for v1.1)
- **Hours used:** Not available
- **Cloud Provider:** Not available
- **Compute Region:** Not available
- **Carbon Emitted:** Not available

## Technical Specifications [optional]

### Model Architecture and Objective

Transformer-based architecture using DETR (Detection Transformer)

### Compute Infrastructure

[More Information Needed]

#### Hardware

NVIDIA GPUs recommended for inference due to computational demands of the Transformer architecture.

#### Software

Transformers library from Hugging Face, PyTorch backend.

## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:** **[optional]**

@article{smock2023aligning,
  title={Aligning benchmark datasets for table structure recognition},
  author={Smock, Brandon and Pesala, Rohith and Abraham, Robin},
  booktitle={International Conference on Document Analysis and Recognition},
  pages={371--386},
  year={2023},
  organization={Springer}
}


**APA:** **[optional]**

Smock, B., Pesala, R., & Abraham, R. (2023). Aligning benchmark datasets for table structure recognition. International Conference on Document Analysis and Recognition, 371–386.

## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

[More Information Needed]

## More Information [optional]

[More Information Needed]

## Model Card Authors [optional]

Unstructured.io

## Model Card Contact

shane@unstructured.io