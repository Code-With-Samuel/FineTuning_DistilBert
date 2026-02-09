# FineTuning DistilBERT

Fine-tuning DistilBERT for sentiment analysis on the Yelp Polarity dataset. This project demonstrates a complete machine learning workflow for text classification using the Hugging Face Transformers library.

## 📋 Project Overview

This project fine-tunes a pre-trained **DistilBERT** model on the **Yelp Polarity dataset** for binary sentiment classification (positive/negative reviews). DistilBERT is a lightweight, faster, and cheaper version of BERT that retains ~97% of BERT's performance while being 40% smaller and 60% faster.

### Key Features
- Load and fine-tune DistilBERT base model
- Dataset tokenization and preprocessing using Hugging Face Datasets
- Training with custom hyperparameters using the Trainer API
- Model evaluation on test dataset
- Model saving for future inference

## 📦 Requirements

- Python 3.7+
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Gradio (optional, for interface)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/Code-With-Samuel/FineTuning_DistilBert.git
cd FineTuning_DistilBert
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
FineTuning_DistilBert/
├── finetune_DistilBERT.ipynb       # Main Jupyter notebook with all steps
├── fine_tuned_yelp_model/          # Fine-tuned model (generated after training)
│   ├── config.json                 # Model configuration
│   ├── tokenizer.json              # Tokenizer vocabulary and settings
│   ├── tokenizer_config.json       # Tokenizer configuration
│   └── model.safetensors           # Model weights
├── results/                         # Training checkpoints and results
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🚀 Usage

### Running the Notebook

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `finetune_DistilBERT.ipynb` and run cells in order:
   - **Section 1**: Load the Model and Tokenizer
   - **Section 2**: Load the Yelp Polarity Dataset
   - **Section 3**: Tokenize the Dataset
   - **Section 4**: Configure Training Parameters
   - **Section 5**: Initialize the Trainer
   - **Section 6**: Fine-tune the Model
   - **Section 7**: Evaluate Model Performance
   - **Section 8**: Save the Fine-tuned Model

### Training Details

- **Model**: DistilBERT (uncased, base)
- **Dataset**: Yelp Polarity (binary classification)
- **Training Subset**: 1,000 samples
- **Test Subset**: 500 samples
- **Learning Rate**: 2e-5
- **Batch Size**: 16
- **Epochs**: 1
- **Evaluation**: At end of each epoch

## 📊 Model Output

After training, the fine-tuned model is saved to `fine_tuned_yelp_model/` with:
- Model weights in `model.safetensors` format
- Tokenizer configuration for preprocessing inference text
- Model configuration for reproducibility

Training checkpoints are saved to `results/` directory.

## ⚠️ Important Notes

**Large Files**: The `model.safetensors` files are not included in version control as they exceed GitHub's recommended file sizes. To use the pre-trained model for inference, you'll need to train the model locally first by running the notebook, or download it from Hugging Face Model Hub.

For production deployments and sharing large model files, consider using:
- [Hugging Face Model Hub](https://huggingface.co/models)
- [Git LFS](https://git-lfs.github.com/)
- Cloud storage (AWS S3, Google Cloud Storage, etc.)

## 🔧 Dependencies

| Package | Purpose |
|---------|---------|
| `transformers` | Pre-trained models and training utilities |
| `datasets` | Dataset loading and preprocessing |
| `torch` | Deep learning framework |
| `gradio` | Optional UI for model inference |

## 📈 Results

Training results and metrics are saved during the training process. Evaluation metrics (accuracy, loss, etc.) are displayed at the end of the `trainer.evaluate()` step.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Open issues for bugs or feature requests
- Submit pull requests with improvements
- Add support for other datasets or models
- Optimize training parameters

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

**Code-With-Samuel**

Feel free to reach out with questions or feedback!

## 📚 References

- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Yelp Polarity Dataset](https://huggingface.co/datasets/yelp_polarity)