# ProtoAI

### ProtoAI is an open-source project developed during the Chilean OpenAI Hackathon in 2024. This project aims to solve challenges related to prosthetics and other issues associated with various human disabilities.

## How to Run the FastAPI Project

To set up and run the FastAPI application, follow these steps:

### 1. Prerequisites
- Ensure you have Python 3.9+ installed on your system.
- Install `pip` for managing Python packages.
- (Optional) Use a virtual environment for project isolation.

### 2. Clone the Repository
Clone the repository and navigate into the project folder:
```bash
git clone https://github.com/yourusername/protoai.git
```
then `cd` into it
```bash
cd ProtoAI
```

### 3. Install the Dependencies
Install the required packages using:
```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables (Optional)
  ```bash
touch .env
  ```
write your apikey into the `.env` file
  ```bash
echo OPENAI_API_KEY="your_api_key_here" > .env
  ```

### 5. Run the Application
Start the FastAPI server:
```bash
uvicorn main:app --reload
```
- The `--reload` flag enables hot reloading for development.
- Once the server starts, access the application at:
  - [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) (interactive API documentation)

### 7. License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

