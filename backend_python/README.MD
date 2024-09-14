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



 uvicorn main:app --host 172.21.220.172 --port 8086 --reload
