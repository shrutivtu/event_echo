# EventEcho: Combined App API - Setup and Installation Guide 

This repository contains the setup and usage instructions for the **Combined App API**, which provides functionality to create a vector store and query audio data. The API is built with Python and exposes endpoints for creating and querying the vector store.

## Table of Contents
- [Installation](#installation)
- [Running the API](#running-the-api)

## Installation

### Prerequisites
Make sure you have the following installed on your system:
- Python 3.7 or later
- pip (Python package installer)
- Node
- npm
- Mistral API
- LMNT API

### Steps

1. **Clone the repository**:
   Clone the repository to your local machine using Git.

   ```bash
   git clone <repository-url>
   cd <repository-directory>

2. **Create and activate a virtual environment** (optional but recommended):
   To ensure the correct dependencies are installed without affecting your global Python environment, create a virtual environment:

   ```bash
   python -m venv venv

3. **Install requirements.txt** 
Install the dependencies by installing the requirements.txt file.
   ```bash
   pip install -r requirements.txt 

## Running the API

4. **Start the API**:
   To run the API in the backend, enter the following command. 

   ```bash
   uvicorn combined_app:app --host 0.0.0.0 --port 8000 

5. **Run the Web App:**
    Enter the UI App folder. Run the web app with the following commands.
    ```bash
    run npm install
    run npm start
