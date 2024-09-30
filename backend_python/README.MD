pip install fastapi uvicorn transformers torch


phobert_server/
│
├── app/
│   ├── __init__.py           # Initialize the app as a package
│   ├── main.py               # Main FastAPI application
│   ├── models/               # Folder for any machine learning models
│   │   ├── __init__.py       # Initialize the models module
│   │   └── phobert_model.py  # PhoBERT model loading and handling logic
│   └── routes/               # Folder for route handlers (API endpoints)
│       ├── __init__.py       # Initialize the routes module
│       └── text_processing.py # Route for handling text processing
│
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── .gitignore                # Files and directories to ignore in version control

///////////////// Dockerfile

Step 3: Check Docker Status
docker info
Step 4: Retry Building the Docker Image
cd app




docker build -t phobert-fastapi-app .
mysql -h 10.60.10.241 -u test -p
Doantotnghiep@123

//Check ip address enabled
sudo lsof -i -n -P | grep mysqld



 uvicorn main:app --host 10.60.10.52 --port 8086 --reload


============== DOCKER
1.	Specify the correct path to the Dockerfile
docker build -t reviews -f app/Dockerfile 
2. Move to the correct directory.
cd app
docker build -t reviews .
3. Next Steps After Fixing the Dockerfile Path
docker build -t reviews .
docker run -d --name reviews-container -p 8086:8086 reviews

4. logs
docker logs reviews-container
=========
5. docker login
5. docker push vinhvo6494/reviews:latest
6. docker search vinhvo6494/reviews
7. docker pull vinhvo6494/reviews:latest
===========
docker run -d --name python_reviews -p 8086:8086 vinhvo6494/reviews:latest