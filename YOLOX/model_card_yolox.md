---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards

---

# Model Card for yolox

<!-- Provide a quick summary of what the model is/does. -->

The yolox model trained by Unstructured on custom DocLayNet dataset for layout detection tasks. The model is designed to take document image as input and output bounding boxes of layout elements on original DocLayNet classes. YoloX architecture was introduced in paper 'YOLOX: Exceeding YOLO Series in 2021' by Zheng Ge et al.

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

The yolox model name comes from YoloX architecture used to train it, an advanced version of the YOLO object detection architecture. Introduced in paper 'YOLOX: Exceeding YOLO Series in 2021' by Zheng Ge et al, developed as part of the SuperGradients repository by Deci-AI, extended with custom post-processing by Unstructured, is a cutting-edge object detection model that focuses on improving speed and accuracy. With various enhancements and optimizations, it enables efficient and precise object detection across a range of applications. The model is trained on 5K sampled open-source DocLayNet datasets. The model is publicly available through the Unstructured open-source library.

- **Developed by:** Unstructured.io
- **Funded by [optional]:** Unstructured.io
- **Shared by [optional]:** Unstructured.io
- **Model type:** Object Detection
- **Language(s) (NLP):** N/A (Trained on Mostly English)
- **License:** Apache License 2.0
- **Finetuned from model [optional]:** YOLO series: YOLOX

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** https://github.com/Megvii-BaseDetection/YOLOX
- **Paper [optional]:** https://arxiv.org/abs/2107.08430
- **Demo [optional]:** Not available

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

Used for document layout elements detection tasks in various domains documents including science articles, finance documents, instructions, patents, web pages.

### Downstream Use [optional]

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

Applied in document processing pipeline for parsing document elements into json representation.

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

Potential inaccuracies when applied to document types not covered during training. Inconsistencies may arise when encountering heavily stylized or multi-layered documents.

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Users should be aware of the model’s potential biases related to dataset-specific training. Users should also verify extracted results manually for critical use cases.

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Regularly update the training dataset with diverse samples, and conduct fairness audits by testing the model across varied scenarios.

## How to Get Started with the Model

Use the code below to get started with the model.

```python
from unstructured_inference.models.yolox import MODEL_TYPES, UnstructuredYoloXModel
from PIL import Image

file_path = "/path/to/image"
image = Image.open(file_path)
image = image.convert("RGB")

model = UnstructuredYoloXModel()
model.initialize(MODEL_TYPES["yolox"]["model_path"], MODEL_TYPES["yolox"]["label_map"])
layout_elements = model.predict(image)
print(layout_elements)
```


## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

The training data is DocLayNet dataset. Classes include: caption, footnote, formula, list-item, page-footer, page-header, picture, section-header, table, text, title. Check https://github.com/DS4SD/DocLayNet for more details

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing [optional]

Preprocessing methods like random resizing, random cropping, random horizontal flipping, and mosaic augmentation.


#### Training Hyperparameters

- **Training regime:** The model was trained using Stochastic Gradient Descent (SGD) with a variable learning rate adjusted according to a specific schedule. <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

#### Speeds, Sizes, Times [optional]

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

[More Information Needed]

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

N/A

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

N/A

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

N/A

### Results

The model is a solid benchmark for document layout detection task.

#### Summary

The model is a solid benchmark for document layout detection task.

## Model Examination [optional]

<!-- Relevant interpretability work for the model goes here -->

[More Information Needed]

## Environmental Impact [optional]

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** Training conducted on NVIDIA GPUs.
- **Hours used:** N/A
- **Cloud Provider:** N/A
- **Compute Region:** Not available
- **Carbon Emitted:** Not available

## Technical Specifications [optional]

### Model Architecture and Objective

Object detection model with YOLOX architecture, trained on DocLayNet for layout detection tasks.

### Compute Infrastructure

[More Information Needed]

#### Hardware

CUDA-compatible GPUs, ideally NVIDIA's higher-end models for efficient training and inference.

#### Software

Python 3.x, PyTorch, yolox

## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:** **[optional]**

@article{yolox2021, title={YOLOX: Exceeding YOLO Series in 2021}, author={Ge, Zheng and others}, journal={arXiv preprint arXiv:2107.08430}, year={2021}}

**APA:** **[optional]**

Ge, Zheng, et al. (2021). YOLOX: Exceeding YOLO Series in 2021. arXiv preprint arXiv:2107.08430.

## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

[More Information Needed]

## More Information [optional]

[More Information Needed]

## Model Card Authors [optional]

Unstructured.io

## Model Card Contact

shane@unstructured.io