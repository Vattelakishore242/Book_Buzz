# 📚 Book Buzz

A comprehensive **Library Management System** and **Book Recommendation Engine** built with Python (Tkinter GUI) that supports book inventory, borrowing/returning functions, and personalized book recommendations using **Collaborative Filtering (SVD)**, **Content-Based Filtering (TF-IDF)**, and **OpenLibrary API integration**.

---

## 🚀 Features

- ✅ Add, delete, borrow, and return books
- ✅ Track student transactions with automatic CSV logging
- ✅ View and search books dynamically
- ✅ View borrowed books with borrower details
- ✅ Content-Based Book Recommendations using TF-IDF and cosine similarity
- ✅ Collaborative Filtering using SVD (Matrix Factorization)
- ✅ Integration with OpenLibrary API for external book data
- ✅ GUI built with Tkinter
- ✅ Calendar widget for date picking
- ✅ Evaluation function for model precision/recall

---

## 🧠 Core Technologies

| Category       | Tools/Technologies             |
|----------------|-------------------------------|
| GUI            | Python, Tkinter, tkcalendar   |
| ML Algorithms  | SVD (Surprise), TF-IDF (sklearn) |
| NLP & Similarity | Scikit-learn, cosine similarity |
| Data Handling  | Pandas, CSV                   |
| APIs           | OpenLibrary REST API          |
| Visualization  | Treeview tables in Tkinter    |

---

## 📊 Algorithms & Methods

### 1. 📘 **Collaborative Filtering** (SVD via Surprise)

- Used for generating book recommendations based on user-item interaction matrix
- Predicts unseen book preferences using matrix factorization
- Key functions:
  - `train_recommender_model()`
  - `get_book_recommendations(user_id)`

### 2. 📙 **Content-Based Filtering** (TF-IDF + Cosine Similarity)

- Recommends similar books based on textual content (title, author, year)
- Computes similarity of book descriptions with borrowed books
- Key functions:
  - `recommend_books_based_on_content(student_name)`
  - `recommend_books(user_input, books_df, top_n)`

### 3. 🔍 **Book Fetching with API + Local CSV**

- Merges results from OpenLibrary API and local CSV
- Fields fetched: title, author, publisher, subjects
- Key function:
  - `fetch_books_from_openlibrary_and_csv(query)`

---

## 🛠️ Functionality Breakdown

| Function | Description |
|---------|-------------|
| `add_book()` | Adds a new book entry to the local CSV |
| `delete_book()` | Deletes a book by title-author-year |
| `borrow_book()` | Logs a book borrow transaction |
| `return_book()` | Marks a book as returned and updates logs |
| `view_books()` | Displays books in a Treeview UI |
| `search_books()` | Searches for books in CSV |
| `view_borrowed_books()` | Displays all borrowed books in a separate window |
| `recommend_books()` | Fetches content-based recommendations |
| `get_recommendations()` | Gets books based on user-entered topics using TF-IDF |
| `evaluate_recommendation_model()` | Computes Precision and Recall for recommender model |

---


