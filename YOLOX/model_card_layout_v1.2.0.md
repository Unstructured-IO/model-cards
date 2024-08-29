---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards

---

# Model Card for layout_v1.2.0

<!-- Provide a quick summary of what the model is/does. -->

The layout_v1.2.0 is YoloX model trained by Unstructured on custom dataset for layout detection tasks. The model is designed to take document image as input and output bounding boxes of layout elements: paragraph, page_number, image, paragraphs_in_image, title, table, paragraphs_in_table, other, page_header, subheading, formulas, page_footer, paragraphs_in_form, checkbox, checkbox_checked, form, radio_button_checked, radio_button, code_snippet. YoloX architecture was introduced in paper 'YOLOX: Exceeding YOLO Series in 2021' by Zheng Ge et al.

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

The layout_v1.2.0 is YoloX architecture model, an advanced version of the YOLO object detection architecture. Introduced in paper 'YOLOX: Exceeding YOLO Series in 2021' by Zheng Ge et al, developed as part of the SuperGradients repository by Deci-AI, extended with custom post-processing by Unstructured, is a cutting-edge object detection model that focuses on improving speed and accuracy. With various enhancements and optimizations, it enables efficient and precise object detection across a range of applications. The model is designed to take document images as input and output bounding boxes for a list of proprietary layout elements from Unstructured. This design ensures that the inference can be generalized to various types of documents. It is an enhanced version of layout_v1.1.0, with improvements in data quality and an increase in the quantity of training data. The model is trained on a 70K custom dataset specifically for layout detection tasks. The model is not available for public use.

- **Developed by:** Unstructured.io
- **Funded by [optional]:** Unstructured.io
- **Shared by [optional]:** N/A (Model is not available for public use)
- **Model type:** Object Detection
- **Language(s) (NLP):** N/A (Trained on Mostly English)
- **License:** N/A (Model is not available for public use)
- **Finetuned from model [optional]:** YOLO series: YOLOX

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** https://github.com/Unstructured-IO/od-modelling-super-gradients (not available for public use)
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
# Example code to load and use layout_v1.2.0 model
# To use this code it is necessary to have access to the private Unstructured repository
import onnxruntime

from huggingface_hub import hf_hub_download
from super_gradients.training.processing.processing import ComposeProcessing
from super_gradients.training.utils.predict import DetectionPrediction, ImageDetectionPrediction

from unstructured_sg.model_configs import MODEL_CONFIGS_ONNX, UnstructuredODModelConfigOnnx
from unstructured_sg.utils.inference import load_document, save_detection_prediction_to_json

model_name = "internal_hf_model_name_for_layout_v1.2.0"  # alias for layout_v1.2.0
model_config = MODEL_CONFIGS_ONNX[model_name]

# Start onnx session
onnx_session = load_session(model_config.model_name)
# Load document
file_path = hf_hub_download(repo_id="nielsr/example-pdf", repo_type="dataset", filename="example_pdf.png")
images = load_document(file_path)
for page_count, image_array in enumerate(images):
  image_processor = get_image_processor(model_config)
  preprocessed_image, processing_metadata = image_processor.preprocess_image(image=image_array.copy())

  inputs = [o.name for o in onnx_session.get_inputs()]
  outputs = [o.name for o in onnx_session.get_outputs()]
  output = onnx_session.run(outputs, {inputs[0]: np.expand_dims(preprocessed_image, 0)})

  pred_boxes = output[0][:, 0:4]
  pred_scores = output[0][:, 4]
  pred_classes = output[0][:, 5]

  prediction = DetectionPrediction(bboxes=pred_boxes, bbox_format="xyxy", confidence=pred_scores, labels=pred_classes, image_shape=image_array.shape[:2])
  postprocessed_prediction = image_processor.postprocess_predictions(prediction, processing_metadata)
  image_detection_prediction = ImageDetectionPrediction(image=image_array, prediction=postprocessed_prediction, class_names=layout_classes)
  image_output = image_detection_prediction.draw()

  suffix = Path(file_name).suffix
  output_image_path = f"{file_name.removesuffix(suffix)}_{page_count}.png"
  cv2.imwrite(str(output_image_path), image_output)
  save_detection_prediction_to_json(image_detection_prediction, output_image_path)


## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

The training data is a custom dataset proprietary to Unstructured. The data is not available to the public.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing [optional]

Image resizing, and data augmentation techniques such as random cropping, blurring and color jittering are used.


#### Training Hyperparameters

- **Training regime:** The model was trained using Stochastic Gradient Descent (SGD) with a variable learning rate adjusted according to a specific schedule. Multiple GPUs were employed for distributed training. <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

#### Speeds, Sizes, Times [optional]

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

[More Information Needed]

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

The training data is a custom dataset proprietary to Unstructured. The data is not available to the public.

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

Evaluated across different documents layouts and types.

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

mAP (mean Average Precision), F1 score, Precision, Recall.

### Results

The model outperforms several internal and sampled public datasets.

#### Summary

Significant improvements in layout element detection.

## Model Examination [optional]

<!-- Relevant interpretability work for the model goes here -->

[More Information Needed]

## Environmental Impact [optional]

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** Training conducted on NVIDIA GPUs: H100 or A100.
- **Hours used:** Not available
- **Cloud Provider:** modal.com
- **Compute Region:** Not available
- **Carbon Emitted:** Not available

## Technical Specifications [optional]

### Model Architecture and Objective

Object detection model with YOLOX architecture, trained on custom dataset for layout detection tasks.

### Compute Infrastructure

[More Information Needed]

#### Hardware

CUDA-compatible GPUs, ideally NVIDIA's higher-end models for efficient training and inference.

#### Software

Python 3.x, PyTorch, unstructured_sg.

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

Untructured.io

## Model Card Contact

shane@unstructured.io