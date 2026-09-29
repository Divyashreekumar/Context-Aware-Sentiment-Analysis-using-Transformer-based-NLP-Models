# Context-Aware Sentiment Analysis using Transformer-Based NLP Models

## Introduction

**Context-Aware Sentiment Analysis using Transformer-Based NLP Models** is a Natural Language Processing (NLP) project that analyzes text and classifies its sentiment into three categories: **Positive, Negative, and Neutral**.

The project uses a **fine-tuned BERT model** with Hugging Face Transformers to understand the context and semantic meaning of text rather than relying only on individual keywords. This helps the system identify sentiment from different types of user-generated content more effectively.

The application provides an interactive **Streamlit** interface for analyzing individual comments, YouTube comments, and Amazon product reviews. YouTube comments can be collected using the **YouTube Data API**, while Amazon product reviews are retrieved using the **Unwrangle API**.

---

## Features

* **Single Comment Analysis**
  Enter a comment or text and predict its sentiment.

* **YouTube Comments Analysis**
  Fetch comments from a YouTube video and analyze their overall sentiment.

* **Amazon Product Review Analysis**
  Retrieve Amazon product reviews and classify their sentiments.

* **Three Sentiment Categories**
  Classifies text as:

  * Positive
  * Negative
  * Neutral

* **Context-Aware Analysis**
  Uses a fine-tuned BERT model to capture contextual and semantic information from text.

* **Interactive Web Interface**
  Provides an easy-to-use interface built with Streamlit.

---

## Technologies Used

* Python
* Streamlit
* BERT
* Hugging Face Transformers
* Natural Language Processing (NLP)
* YouTube Data API
* Unwrangle API
* Pandas
* NumPy
* Scikit-learn
* PyTorch

---

## Project Structure

```text
Context-Aware-Sentiment-Analysis/
│
├── page.py
├── predict.py
├── amazon_review.py
├── requirements.txt
├── readme.md
├── comments_sentiments.csv
├── data.csv
│
├── intent classification1/
│   ├── config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   ├── training_args.bin
│   └── vocab.txt
│
├── bert_finetune.ipynb
├── experiment.ipynb
└── product_review.ipynb
```

---

## How It Works

The system follows a simple workflow:

```text
User Input / YouTube Comments / Amazon Reviews
                    ↓
              Text Processing
                    ↓
             BERT Tokenization
                    ↓
          Fine-Tuned BERT Model
                    ↓
         Sentiment Classification
                    ↓
       Positive / Negative / Neutral
```

The BERT model processes the input text while considering the relationship between words and their surrounding context. The trained model then predicts the corresponding sentiment category.

---

## Applications

This project can be useful for:

* Social media sentiment analysis
* YouTube comment analysis
* Product review analysis
* Customer feedback analysis
* Opinion mining
* Understanding user sentiment from text data

---

# Instructions to Run the Project

1. **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    ```

2. **Activate the Virtual Environment**:
    - On Windows:
      ```bash
      venv\Scripts\activate
      ```
    - On macOS/Linux:
      ```bash
      source venv/bin/activate
      ```

3. **Install Requirements**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Streamlit Application**:
    ```bash
    streamlit run page.py
    ```

## Future Enhancements

* Improve model performance with larger and more diverse datasets
* Add multilingual sentiment analysis
* Add sentiment visualization and analytics
* Support additional review and social media platforms
* Deploy the application as a cloud-based service

---

## Conclusion

This project demonstrates the application of **Transformer-based NLP models** for context-aware sentiment analysis. By combining a fine-tuned BERT model with real-world data sources such as YouTube comments and Amazon product reviews, the system provides an interactive approach to understanding sentiment from textual data.
