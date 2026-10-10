Aerial Image Semantic Segmentation
A Deep Learning project for semantic segmentation of aerial imagery, focused on identifying and classifying different types of terrain and structures in rural areas.
The project includes the creation of a custom dataset, image annotation, preprocessing, model training and evaluation, comparison between different segmentation architectures, and a graphical interface for testing trained models on new aerial images.
Overview
The main objective of this project is to apply Artificial Intelligence and Computer Vision techniques to semantic segmentation of aerial images.
A custom dataset was created specifically for the problem under study, since the publicly available datasets that were analysed did not adequately match the required classes, visual characteristics, or geographical context.
Two Deep Learning architectures were evaluated:
- U-Net
- DeepLabV3+
The models receive aerial images as input and generate segmentation masks in which each pixel is assigned to one of the defined classes.
Segmentation Classes
The dataset contains six segmentation classes:
- Background
- Agricultural fields
- Vegetation
- Water bodies
- Roads
- Structures
Each class is represented in the segmentation masks through a unique identifier.
Dataset
The dataset was created specifically for this project.
Image Collection
Aerial images were collected using Google Earth, with the goal of obtaining visually diverse rural environments containing different combinations of the defined classes.
During image selection, attention was given to:
- Diversity of terrain
- Different vegetation patterns
- Different agricultural areas
- Different road types
- Different building and roof characteristics
- Water bodies with different visual appearances
- Variations in lighting and image composition
Annotation
The collected images were manually annotated using CVAT (Computer Vision Annotation Tool).
Each relevant region was assigned to one of the predefined classes, producing a segmentation mask corresponding to each aerial image.
The annotation process was particularly important in regions where:
- Class boundaries were ambiguous
- Different classes had similar colours or textures
- Objects had irregular shapes
- Small elements were difficult to delimit accurately
Data Preprocessing
Before training, the images and segmentation masks are prepared for use by the Deep Learning models.
The preprocessing pipeline includes:
1. Loading the aerial images and their corresponding masks
2. Converting image data to the required format
3. Organising image-mask pairs
4. Splitting large images into smaller patches
5. Preparing class labels
6. Normalising image values when required
7. Dividing the dataset into training, validation, and test sets
Image Patches
The original images are divided into patches of:
256 x 256 pixels
Using patches reduces the amount of data processed simultaneously by the model and allows large aerial images to be handled more efficiently.
Each image patch has a corresponding mask patch, preserving the spatial relationship between the input image and its expected segmentation result.
The final training configuration does not use data augmentation.
Dataset Split
The dataset is divided into three subsets:
- Training set — used to update the model parameters
- Validation set — used to monitor model performance during training
- Test set — used for final evaluation on unseen data
The separation between these subsets helps evaluate the ability of the models to generalise to images that were not directly used during training.
Models
U-Net
U-Net is the main segmentation architecture used in this project.
It follows an encoder-decoder structure:
- The encoder progressively extracts higher-level visual features while reducing spatial resolution.
- The decoder reconstructs the spatial information and generates the final segmentation output.
- Skip connections transfer information from encoder layers directly to corresponding decoder layers.
These skip connections help preserve spatial details that may otherwise be lost during downsampling, improving the localisation of class boundaries.
The final layer performs multi-class pixel classification using a Softmax activation.
DeepLabV3+
DeepLabV3+ was implemented as a comparison architecture.
The purpose of including DeepLabV3+ was to evaluate whether a more complex segmentation architecture could provide better results with the available dataset.
In the experiments performed during the project, U-Net achieved the best overall performance and was also faster to retrain during the iterative development of the dataset.
Training Process
The general training workflow is:
Aerial Images + Segmentation Masks
                ↓
          Preprocessing
                ↓
         256x256 Patches
                ↓
   Train / Validation / Test Split
                ↓
           Model Training
                ↓
      Validation During Training
                ↓
           Model Evaluation
                ↓
        Segmentation Predictions
During training, the model receives an aerial image patch and produces a prediction for each pixel.
The prediction is compared with the corresponding ground-truth segmentation mask, and the model parameters are updated in order to reduce the prediction error.
This process is repeated across multiple epochs.
Evaluation Metrics
The models are evaluated using metrics suitable for semantic segmentation.
Pixel Accuracy
Pixel Accuracy measures the proportion of correctly classified pixels relative to the total number of pixels.
Although simple and intuitive, this metric can be misleading when the dataset contains significant class imbalance.
Intersection over Union
Intersection over Union (IoU), also known as the Jaccard Index, measures the overlap between the predicted region and the corresponding ground-truth region.
A value of:
- 1 represents perfect overlap
- 0 represents no overlap
Mean IoU
Mean IoU calculates the average IoU across all segmentation classes and provides a more balanced overview of model performance.
In addition to quantitative metrics, predicted masks are visually analysed in order to identify common segmentation errors.
Graphical Interface
A graphical interface was developed to simplify model testing.
The interface allows the user to:
- Select one or more aerial images
- Select the trained model to use
- Run semantic segmentation
- Generate predicted masks
- Save the resulting masks to a selected folder
This avoids the need to manually modify or execute individual prediction scripts every time a new image needs to be tested.
The graphical interface also serves as a starting point for a future large-scale system capable of analysing aerial imagery at municipal scale.
Patch Reconstruction
Large images are processed using smaller patches.
After inference, the predicted patches are reconstructed to generate the final segmentation mask.
One limitation of this approach is the possible appearance of inconsistencies at patch boundaries, since each patch is processed independently.
Future improvements may include:
- Overlapping patches
- Weighted blending between neighbouring predictions
- Improved reconstruction strategies
- Larger context windows
Technologies
The main technologies and tools used in this project are:
Programming and Deep Learning
- Python
- TensorFlow
- Keras
Image Processing and Numerical Computing
- OpenCV
- NumPy
- Matplotlib
Dataset Creation
- CVAT
- Google Earth
Version Control
- Git
- GitHub
Project Structure
The exact repository structure may vary as the project evolves, but the project is generally organised around the following areas:
project/
│
├── dataset/
│   ├── images/
│   └── masks/
│
├── models/
│   └── trained models
│
├── src/
│   ├── preprocessing
│   ├── training
│   ├── evaluation
│   └── prediction
│
├── gui/
│   └── graphical interface
│
├── requirements.txt
└── README.md
Note: folder names may differ from the structure shown above depending on the current repository organisation.

Installation
1. Clone the Repository
git clone <repository-url>
cd <repository-folder>
2. Create a Virtual Environment
Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1
macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
3. Install Dependencies
If the repository contains a requirements.txt file:
pip install -r requirements.txt
Usage
The project contains scripts for the main stages of the segmentation pipeline, including:
- Dataset preprocessing
- Patch generation
- Model training
- Model evaluation
- Prediction on new images
- Graphical interface execution
Because script names and paths may vary depending on the current repository version, run the appropriate script from the project directory using:
python <script_name>.py
For example:
python train.py
or:
python gui.py
Replace the example filenames with the actual filenames available in the repository.

Trained Models
Trained models are stored using the Keras model format:
.keras
These files can later be loaded for evaluation or prediction without retraining the network.
Due to their size, trained models may not always be included directly in the Git repository.
Known Limitations
The main limitations identified during the project include:
- Limited dataset size
- Class imbalance
- Visual similarity between some segmentation classes
- Possible inconsistencies in manual annotations
- Difficulty identifying small objects
- Patch-boundary artefacts during reconstruction
- Limited computational resources for extensive experimentation
- Increased training time as the dataset grows
Some common class-confusion cases include:
- Dark or green water being confused with vegetation or shadows
- Dirt roads being confused with agricultural fields
- Buildings being confused with roads
- Blue or green rooftops being confused with water
- Vegetation being confused with agricultural areas
Future Work
Several improvements can be explored in future versions of the project:
- Increase the size and diversity of the dataset
- Collect images from a wider variety of geographical areas
- Improve annotation consistency
- Review ambiguous class boundaries
- Explore additional data augmentation strategies
- Use pre-trained models and transfer learning
- Perform deeper hyperparameter optimisation
- Test alternative loss functions for class imbalance
- Improve reconstruction of overlapping patches
- Reduce patch-boundary artefacts
- Extend the graphical interface
- Evaluate additional semantic segmentation architectures
A major future objective is the development of an application capable of detecting and segmenting specific classes in large-scale aerial imagery, including municipal-scale areas.
The current graphical interface provides an initial foundation for this solution, since it already supports model selection, image processing, segmentation prediction, and mask generation.
Scaling the system to municipal-level imagery would require improvements in:
- Large-image processing
- Patch management
- Reconstruction quality
- Computational performance
- Result visualisation
- Territorial analysis workflows
Academic Context
This project was developed in an academic context with the objective of studying and applying concepts related to:
- Artificial Intelligence
- Machine Learning
- Deep Learning
- Computer Vision
- Convolutional Neural Networks
- Semantic Segmentation
- Aerial Image Analysis
The project includes both the technical implementation and an academic study of the methods, models, metrics, limitations, and results obtained.
Use of Generative AI
Generative Artificial Intelligence tools were used as complementary support during the project, particularly for language review, text reformulation, organisation of ideas, and development assistance.
All technical decisions, implementation work, experimental results, analysis, and conclusions were reviewed and validated according to the work effectively carried out during the project.
License
No license has been explicitly defined for this repository yet.
If the project is intended to be publicly reusable, an appropriate open-source license should be added, such as:
- MIT License
- Apache License 2.0
- GPLv3
Authors
Developed as part of an academic project focused on semantic segmentation of aerial imagery using Deep Learning.
