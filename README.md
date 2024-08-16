# Assignment - 3 - Nike Shopping Bot

Welcome to the Nike Product Search Bot! This application helps you search for Nike products using both text queries and image uploads. Built with Streamlit, it leverages the CLIP model for embeddings and Pinecone for vector database management.

## Video Demo

[![Video Demo](https://img.shields.io/badge/Video%20Demo-<COLOR>?style=for-the-badge&logo=<LOGO>&logoColor=white)](https://youtu.be/FX2NfVkHWtU)

## Documentation

[![codelabs](https://img.shields.io/badge/codelabs-4285F4?style=for-the-badge&logo=codelabs&logoColor=white)](https://codelabs-preview.appspot.com/?file_id=1rxwEOAuZc-nYXXGWnDTdTcSZ_Ha2OzUyzep-SCzlBuo#0)

## Features

### Product Search
<img width="1276" alt="Product Search" src="https://github.com/deveshcode/shopping-multimodal-rag/assets/37287532/6de1a49e-242e-4836-bcf8-dca0e9346eba">

It performs search functionality as follows:

<img width="1280" alt="Search Functionality" src="https://github.com/deveshcode/shopping-multimodal-rag/assets/37287532/ce713ad3-4df3-4067-8004-5019404546c2">

### Virtual Try-On
You can also virtually try it on by providing your image and product image:

<img width="1280" alt="Virtual Try-On" src="https://github.com/deveshcode/shopping-multimodal-rag/assets/37287532/97c1d403-7dee-4bee-9f39-0ede08e56b55">

## Installation

1. Git clone the repository:

```bash
git clone https://github.com/deveshcode/shopping-multimodal-rag.git
cd shopping-multimodal-rag
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Create a .env file in the root directory and add your API keys:

```bash
PINECONE_API_KEY=your_pinecone_api_key
OPENAI_API_KEY=your_open_ai_key
API_HOST=http://localhost:8001
BUCKET_NAME=gcs_bucket_name 
GOOGLE_APPLICATION_CREDENTIALS=gcloud_creds_json_file
azure_cv_key=azure_cv_key
azure_cv_endpoint=azure_cv_endpoint
```

## Data Scraping and Embedding Pipeline

To collect data and create embeddings for the Nike Product Search Bot, you can follow the steps below:

1. Run the `01_scrapper.py` script to scrape data from the Nike website and save it as a CSV file.

2. Upload the CSV file to an S3 bucket using the `02_upload_to_s3.py` script. Make sure you have the necessary AWS credentials configured.

3. Use the `03_pinecone_storing_data.py` script to create a Pinecone index and upload the embeddings to Pinecone. This will allow you to perform efficient similarity searches on the data.

By following this pipeline, you can ensure that the Nike Product Search Bot has up-to-date data and accurate embeddings for product search and virtual try-on functionalities.

Remember to customize the scripts according to your specific requirements and configurations.

## Usage

To run the REST API server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```

To run the Streamlit app:

```bash
streamlit run app.py
```

Open the provided URL in your web browser to access the application.

## Tools and Technologies

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![CLIP](https://img.shields.io/badge/CLIP-FF4B4B?style=for-the-badge&logo=openai&logoColor=white)](https://beta.openai.com/docs/)
[![Pinecone](https://img.shields.io/badge/Pinecone-FF4B4B?style=for-the-badge&logo=pinecone&logoColor=white)](https://www.pinecone.io/docs/)
[![Google Cloud Storage](https://img.shields.io/badge/Google%20Cloud%20Storage-FF4B4B?style=for-the-badge&logo=google-cloud&logoColor=white)](https://cloud.google.com/storage/docs)
[![Azure Cognitive Services](https://img.shields.io/badge/Azure%20Cognitive%20Services-FF4B4B?style=for-the-badge&logo=microsoft-azure&logoColor=white)](https://docs.microsoft.com/en-us/azure/cognitive-services/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/)

## Data Sources
Data collected from the following sources:
Nike Website : https://www.nike.com/


## Project Team
**Snehil Aryan** - 

https://www.linkedin.com/in/snehil-aryan-158777164/

aryan.s@northeastern.edu

**Sanjay Bhaskar Kashyap** - 

https://www.linkedin.com/in/skashyap11/

kashyap.sanj@northeastern.edu

Lissa Rodrigues - 

https://www.linkedin.com/in/lissar/

rodrigues.li@northeastern.edu



## References
1. OpenAI API Documentation: https://beta.openai.com/docs/ 
2. Pinecone Documentation: https://www.pinecone.io/docs/ 
3. FastAPI Documentation: https://fastapi.tiangolo.com/ 
4. Streamlit Documentation: https://docs.streamlit.io/ 
5. Google Cloud Storage Documentation: https://cloud.google.com/storage/docs 
6. Azure Cognitive Services Documentation: https://docs.microsoft.com/en-us/azure/cognitive-services/ 
