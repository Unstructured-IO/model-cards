
---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
Tesseract
---

# Model Card for Tesseract

<!-- Provide a quick summary of what the model is/does. -->

Tesseract is an open-source optical character recognition (OCR) engine that can recognize and read text in images. It supports a wide variety of languages and can be trained to recognize new fonts or languages.

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

Tesseract is a powerful and versatile open-source OCR engine that uses neural networks to recognize text in images. It was originally developed by Hewlett-Packard in the 1980s and has been maintained by Google since 2006. Tesseract can process a wide variety of image formats and output plain text or more structured formats like hOCR, PDF, and TSV.

-   **Developed by:** Originally by Hewlett-Packard, now maintained by Google and the open-source community
-   **Funded by:** Google (current maintenance)
-   **Shared by:** Google via GitHub
-   **Model type:** Optical Character Recognition (OCR) Engine
-   **Language(s) (NLP):** Supports over 100 languages
-   **License:** Apache License 2.0
-   **Finetuned from model:** Not applicable (Tesseract is a standalone OCR engine)

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

-   **Repository:** [https://github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract)
-   **Paper:** Various papers have been published on Tesseract's development and improvements
-   **Demo:** Various online demos are available, but not officially provided by the Tesseract project

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->
Tesseract can be used directly for:

-   Extracting text from scanned documents
-   Digitizing printed books
-   Reading text from images in various applications
-   Assisting in document processing workflows

### Downstream Use [optional]

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

Tesseract can be integrated into larger systems for:

-   Automated document processing pipelines
-   Search engines for scanned document repositories
-   Assistive technologies for visually impaired users
-   Data entry automation from printed forms

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->
Tesseract is not designed for:

-   Recognizing handwritten text (it's optimized for printed text)
-   Real-time OCR on video streams (it's not optimized for speed)
-   Recognizing text in highly stylized fonts or complex backgrounds

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

-   Tesseract's accuracy can vary depending on the quality of the input image
-   Performance may be lower for languages with limited training data
-   It may struggle with complex layouts, tables, or mixed text and graphics
-   OCR errors can potentially introduce misinformation if results are not verified

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users should be aware of Tesseract's limitations and verify important OCR results manually. For critical applications, consider using Tesseract in conjunction with human review. Regular retraining with diverse datasets can help improve performance across different languages and document types.

## How to Get Started with the Model

Use the code below to get started with the model in python.

```
import pytesseract
from PIL import Image

# If you haven't added Tesseract to your PATH, you need to specify the path:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Open an image file
image = Image.open('path/to/your/image.png')

# Perform OCR on the image
text = pytesseract.image_to_string(image)

print(text)
```

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

Tesseract is pre-trained on a large corpus of text in various languages. The specific details of the training data are not publicly available, but it includes a wide range of fonts and styles for each supported language.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

Tesseract uses a two-pass approach: the first pass tries to recognize words, and the second pass refines the results. It employs long short-term memory (LSTM) networks for text line recognition.


#### Preprocessing [optional]

Tesseract performs several preprocessing steps on input images, including binarization, noise removal, and layout analysis. The exact preprocessing steps can vary depending on the image quality and complexity.

#### Training Hyperparameters

- **Training regime:** Tesseract uses a combination of traditional machine learning techniques and neural networks. The LSTM networks are typically trained using backpropagation through time (BPTT).

#### Speeds, Sizes, Times [optional]

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

The speed of Tesseract varies greatly depending on the hardware, image size, and complexity. On a modern CPU, Tesseract can process a typical page in a few seconds to a minute. The LSTM model files for each language are typically a few megabytes in size. Training times for new languages or specialized models can range from hours to days, depending on the dataset size and computing resources.

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

Tesseract is typically evaluated on standard OCR datasets, which include scanned documents, book pages, and various printed materials.

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

Evaluation considers factors such as image quality, font types, document layout, and language
#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

Common evaluation metrics include accuracy, character error rate (CER), and word error rate (WER).

### Results

Specific results vary depending on the dataset and language. In general, Tesseract performs well on clear, high-quality scans of printed text, with accuracy often exceeding 95% for English text under good conditions.

#### Summary

Tesseract is a highly effective open-source OCR engine that excels in recognizing and extracting text from images, supporting multiple languages and formats. Its continuous development and adaptability make it a valuable tool for various text recognition tasks, though its performance can vary based on the quality of input images and the complexity of the text.

## Model Examination [optional]

<!-- Relevant interpretability work for the model goes here -->

Tesseract's accuracy heavily relies on preprocessing steps such as binarization, deskewing, and noise reduction. The better the preprocessing, the more interpretable and accurate the results. It can integrate post-OCR error correction tools that use language models or dictionaries to adjust and correct recognized text. This post-processing step helps make Tesseract's output more interpretable by identifying and fixing common recognition errors.

## Environmental Impact [optional]

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->
Tesseract can be used on both CPU and GPU hardwares. Below you can select any type of GPU Hardware to estimate the carbon emissions. 
Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** GPU (NVIDIA V100 or similar)
- **Hours used:** [More Information Needed]
- **Cloud Provider:** AWS, Google Cloud, etc.
- **Compute Region:** [More Information Needed]
- **Carbon Emitted:** [More Information Needed]

## Technical Specifications [optional]

### Model Architecture and Objective

Tesseract uses a series of processing steps, including layout analysis, line finding, baseline fitting, and text line recognition using LSTM neural networks.

### Compute Infrastructure

Tesseract can run on a wide range of hardware, from personal computers to servers. It's implemented in C++ for efficiency.

#### Hardware

Tesseract can run on any modern multi-core CPU, and performance improves with faster processors and more cores, especially when processing large batches of documents.


#### Software

-   **Linux**: Tesseract is widely supported on Linux distributions (e.g., Ubuntu, Debian, Fedora). Most package managers offer easy installation, or you can compile it from the source.
-   **Windows**: Tesseract has a Windows build, and you can download precompiled binaries. Tools like Chocolatey also offer simplified installation.
-   **macOS**: Tesseract can be installed on macOS through Homebrew (`brew install tesseract`).

## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:** **[optional]**

@INPROCEEDINGS{4376991,
  author={Smith, R.},
  booktitle={Ninth International Conference on Document Analysis and Recognition (ICDAR 2007)}, 
  title={An Overview of the Tesseract OCR Engine}, 
  year={2007},
  volume={2},
  number={},
  pages={629-633},
  keywords={Optical character recognition software;Search engines;Testing;Open source software;Text recognition;Filters;Prototypes;Independent component analysis;Pipelines;Inspection},
  doi={10.1109/ICDAR.2007.4376991}}


## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

[More Information Needed]

## More Information [optional]

[More Information Needed]

## Model Card Authors [optional]

[More Information Needed]

## Model Card Contact

[More Information Needed]