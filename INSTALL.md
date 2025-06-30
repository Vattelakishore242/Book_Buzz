📘 Book Buzz – Installation & Setup Guide
Welcome to Book Buzz, a personalized book recommendation system with integrated library management and topic-based suggestions using content filtering and external APIs.

This guide will walk you through setting up and running the application locally.

✅ Requirements
Python 3.7+

OS: Windows, Linux, or macOS

Recommended: Use a virtual environment

📦 Step-by-Step Installation
1. 📁 Clone or Download the Project
If using Git:

bash
Copy
Edit
git clone https://github.com/your-username/book_buzz.git
cd book_buzz
Or manually download and unzip the project folder.

2. 🧪 Create a Virtual Environment (Recommended)
bash
Copy
Edit
python -m venv env
Activate it:

Windows:

bash
Copy
Edit
.\env\Scripts\activate
macOS/Linux:

bash
Copy
Edit
source env/bin/activate
3. 📥 Install Required Dependencies
Run:

bash
Copy
Edit
pip install -r requirements.txt
If scikit-surprise causes issues, try:

bash
Copy
Edit
pip install scikit-surprise --no-binary :all:
Or on conda:

bash
Copy
Edit
conda install -c conda-forge scikit-surprise
▶️ Running the Application
Simply run:

bash
Copy
Edit
python main.py
