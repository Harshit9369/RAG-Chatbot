# RAG Chatbot for Lung Cancer Data

This repository contains a Retrieval-Augmented Generation (RAG) chatbot trained on lung cancer medical data. The chatbot uses Pinecone Vector Database for embedding storage and is deployed using Streamlit and FastAPI for inference purposes.

## Getting Started

Follow these steps to set up and run the application.

### Prerequisites

Ensure you have Python installed on your system. You can download it from [python.org](https://www.python.org/).

### Step 1: Create a Virtual Environment

Create a virtual environment to manage dependencies.

```bash
python -m venv venv
```

Activate the virtual environment:

- On Windows:
    ```bash
    .\venv\Scripts\activate
    ```
- On macOS and Linux:
    ```bash
    source venv/bin/activate
    ```

### Step 2: Install Dependencies

Install the required dependencies using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### Step 3: Run the Application

You can run the application using either Streamlit or FastAPI.

#### Using Streamlit

To run the application with Streamlit, use the following command:

```bash
streamlit run app.py
```

#### Using FastAPI

To run the application with FastAPI, use the following command:

```bash
uvicorn app:app --reload
```

## Additional Information

- **Pinecone Vector Database**: Used for embedding storage.
- **Streamlit**: Used for creating the web interface.
- **FastAPI**: Used for creating the API endpoints.

For any issues or contributions, please open an issue or submit a pull request.
