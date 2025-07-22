# Hands-on Workshop - License Plate Recognition

A comprehensive hands-on workshop for building and deploying License Plate Recognition (LPR) models using Cloudera AI (CAI) platform. This workshop covers the complete machine learning lifecycle from data exploration to model deployment.

## Workshop Overview.

This workshop demonstrates how to leverage Cloudera AI for rapid ML prototyping and operationalization, focusing on a real-world License Plate Recognition use case. Participants will learn best practices for computer vision projects, model experimentation, tracking, and deployment strategies.

### What You'll Learn

- How to ingest and explore image datasets in Cloudera AI
- Build and train license plate detection models
- Evaluate and optimize model performance
- Deploy models as REST APIs
- Integrate with Cloudera AI's advanced features

### Business Applications

- Smart parking and toll systems
- Security and access control
- Smart city traffic management
- Law enforcement applications

## Quick Start

### Prerequisites

- Access to Cloudera AI (CAI) platform
- Basic Python programming knowledge
- Familiarity with Jupyter notebooks

### Environment Setup

1. **Access Cloudera AI Platform**

   - Log into your Cloudera AI workspace
   - Create a new project or use an existing one
   - Ensure you have GPU-enabled compute resources available
2. **Clone the repository**

   ```bash
   git clone git@github.com:masorg/hol_workshop_lpr.git
   cd hol_workshop_lpr
   ```
3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```
4. **Verify CAI environment**

   ```bash
   python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
   python -c "import tensorflow as tf; print(f'GPU available: {len(tf.config.list_physical_devices(\"GPU\")) > 0}')"
   ```

### Data Setup

The workshop datasets will be automatically downloaded when you execute the notebooks. No manual data preparation is required.

## Workshop Structure

### Part 1: Exploratory Data Analysis (EDA)

**File**: `Part1_EDA.ipynb`

**Learning Objectives**:

- Data loading and inspection
- License plate crop extraction
- Data quality assessment
- Length distribution analysis
- Visualization and insights

**Key Topics**:

- Understanding the LPR dataset structure
- Identifying data quality issues
- Analyzing license plate characteristics
- Preparing data for model training

### Part 2: Model Training

**File**: `Part2_ModelTraining.ipynb`

**Learning Objectives**:

- Bounding box detection architecture design
- Data preprocessing pipeline
- Model training with variable-length support
- Validation and performance metrics
- Model optimization techniques

**Key Topics**:

- Building regression models for coordinate prediction
- Implementing IoU (Intersection over Union) evaluation
- Training deep learning models for object detection
- Model checkpointing and early stopping

### Part 3: Model Inference & Testing

**File**: `Part3_ModelInference.ipynb`

**Learning Objectives**:

- Model inference pipeline development
- Real-time prediction system implementation
- Performance benchmarking
- Deployment considerations

**Key Topics**:

- Loading trained models from multiple formats
- Comprehensive evaluation on test datasets
- Visual validation of predictions
- Performance analysis and error handling

## Technical Stack

### Core Technologies

- **Deep Learning**: TensorFlow 2.16+, Keras 3.0+
- **Computer Vision**: OpenCV 4.7+, Pillow 9.0+
- **Data Science**: NumPy, Pandas, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Development**: Jupyter, IPython

### Cloudera AI Platform Integration

- **Scalable GPU Resources**: Leverage CAI's GPU-enabled compute for faster training
- **Experiment Tracking**: Built-in MLflow integration for model versioning
- **Model Deployment**: Automated pipeline support for production deployment
- **Enterprise Monitoring**: Real-time model performance and health monitoring
- **Collaborative Workspace**: Team-based development and model sharing

## Expected Outcomes

By the end of this workshop, participants will have:

1. **Practical Experience**: Hands-on experience with real computer vision problems
2. **Model Development**: Complete understanding of the ML development lifecycle
3. **Performance Optimization**: Skills in model evaluation and optimization
4. **Deployment Knowledge**: Understanding of production deployment considerations
5. **Platform Proficiency**: Familiarity with Cloudera AI capabilities and workflows

## Troubleshooting

### Common Issues

**GPU Not Available in CAI**

```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

- Ensure you're using a GPU-enabled compute profile in CAI
- Contact your CAI administrator if GPU resources are not available

**Memory Issues**

- Reduce batch size in training notebooks
- Use smaller image dimensions for testing
- Enable memory growth for TensorFlow
- Consider using CAI's memory-optimized compute profiles

**Import Errors**

```bash
# Reinstall specific packages if needed
pip install --upgrade tensorflow opencv-python
```

### CAI-Specific Considerations

- **Resource Management**: Monitor your compute usage through CAI dashboard
- **Data Persistence**: Use CAI's persistent storage for datasets and models
- **Collaboration**: Share notebooks and models with team members through CAI
- **Version Control**: Use CAI's built-in Git integration for code management

### Getting Help

- Check the notebook outputs for specific error messages
- Verify all dependencies are installed correctly
- Ensure sufficient disk space for dataset downloads
- Consult CAI documentation for platform-specific issues

## Workshop Prerequisites

### Technical Skills

- Basic Python programming
- Familiarity with Jupyter notebooks
- Understanding of machine learning concepts
- Basic knowledge of computer vision (helpful but not required)

### CAI Platform Requirements

- **Access**: Valid Cloudera AI workspace access
- **Compute**: GPU-enabled compute profile recommended
- **Storage**: Sufficient workspace storage for datasets and models
- **Permissions**: Ability to install packages and run notebooks

## Contributing

This workshop is designed for educational purposes. For questions or improvements:

1. Review the existing notebooks
2. Test the workshop flow
3. Submit issues for bugs or improvements
4. Contribute enhancements through pull requests

## License

This workshop is provided as-is for educational purposes. Please refer to individual library licenses for their respective terms.

---

**Note**: This workshop is specifically designed for Cloudera AI platform and leverages its unique capabilities for enterprise ML development. The AgenticAI folder contains separate content and is not part of the main LPR workshop.
