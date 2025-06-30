import os
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import Calendar
import pandas as pd
import re
import requests # type: ignore
from surprise import SVD, Reader, Dataset # type: ignore
from surprise.model_selection import train_test_split # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Path to the CSV files
csv_file = 'lms_second.csv'
# Lists to track newly added, borrowed, and returned books
newly_added_books = []
borrowed_books = []
returned_books = []

# Sample user-item interaction matrix (adjust for real data)
user_item_matrix = pd.DataFrame({
    'User1': [1, 1, 0, 0, 1],
    'User2': [1, 0, 1, 0, 0],
    'User3': [0, 1, 1, 1, 0],
    'User4': [1, 1, 0, 1, 0],
}, index=['Book1', 'Book2', 'Book3', 'Book4', 'Book5'])

# Function to train the collaborative filtering model (SVD)
def train_recommender_model():
    reader = Reader(rating_scale=(0, 1))  # type: ignore # Rating scale is from 0 to 1
    data = Dataset.load_from_df(user_item_matrix.stack().reset_index(name='interaction'), reader) # type: ignore
    trainset, testset = train_test_split(data, test_size=0.2)
    model = SVD() # type: ignore
    model.fit(trainset)
    return model

# Function to get book recommendations
def get_book_recommendations(user_id):
    model = train_recommender_model()
    unseen_books = [book_id for book_id in range(len(user_item_matrix.columns)) if user_item_matrix[user_id].iloc[book_id] == 0]
    predictions = [model.predict(user_id, book_id) for book_id in unseen_books]
    sorted_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)
    recommended_books = [f"Book {prediction.iid + 1}" for prediction in sorted_predictions[:5]]
    return recommended_books

# Function to handle book data (add, delete, view, etc.)
def add_book():
    title = title_entry.get()
    author = author_entry.get()
    pub_year = year_entry.get()
    if title and author and pub_year:
        new_book = pd.DataFrame([[title, author, pub_year]], columns=['Title', 'Author', 'Publisher/Year'])
        try:
            df = pd.read_csv(csv_file)
            df = pd.concat([df, new_book], ignore_index=True)
        except FileNotFoundError:
            df = new_book
        df.to_csv(csv_file, index=False)
        newly_added_books.append([title, author, pub_year])
        messagebox.showinfo("Success", "Book added successfully")
        title_entry.delete(0, tk.END) # type: ignore
        author_entry.delete(0, tk.END) # type: ignore
        year_entry.delete(0, tk.END) # type: ignore
    else:
        messagebox.showwarning("Input Error", "Please fill in all fields")

# Function to view borrowed books
def view_borrowed_books():
    view_window = tk.Toplevel(root)
    view_window.title("Borrowed Books")
    view_window.configure(bg='lightblue')

    if not borrowed_books:
        messagebox.showinfo("No Borrowed Books", "No books are currently borrowed.")
        return

    tree = ttk.Treeview(view_window, columns=("Title", "Author", "Publiser/Year", "Student ID", "Issue Date"), show='headings')
    tree.heading("Title", text="Title")
    tree.heading("Author", text="Author")
    tree.heading("Publiser/Year", text="Publiser/Year")
    tree.heading("Student ID", text="Student ID")
    tree.heading("Issue Date", text="Issue Date")
    tree.pack(fill=tk.BOTH, expand=True)

    for borrowed in borrowed_books:
        tree.insert("", tk.END, values=[borrowed[0], borrowed[1], borrowed[2], borrowed[3], borrowed[4]])

# Function to show the calendar and pick the issue date
def select_issue_date():
    def on_date_selected():
        date = cal.get_date()  # Get the selected date
        # Update the issue_date_entry with the selected date in format YYYY-MM-DD
        issue_date_entry.delete(0, tk.END)
        issue_date_entry.insert(0, date)
        calendar_window.destroy()

    # Create a new window for calendar
    calendar_window = tk.Toplevel(root)
    calendar_window.title("Select Issue Date")
    
    # Calendar for selecting date
    cal = Calendar(calendar_window, selectmode='day', date_pattern='dd-mm-yyyy')
    cal.pack(pady=10)

    # Button to confirm selection
    select_button = tk.Button(calendar_window, text="Select", command=on_date_selected)
    select_button.pack(pady=10)

# Function to return a book
def return_book():
    title = title_entry.get()
    author = author_entry.get()
    pub_year = year_entry.get()
    student_id = student_id_entry.get()
    issue_date = issue_date_entry.get()

    if title and author and pub_year and student_id and issue_date:
        # Search for the book in the borrowed_books list
        borrowed_book = None
        for borrowed in borrowed_books:
            if borrowed[0] == title and borrowed[1] == author and borrowed[2] == pub_year and borrowed[3] == student_id:
                borrowed_book = borrowed
                break
        
        if borrowed_book:
            # Remove the book from borrowed_books
            borrowed_books.remove(borrowed_book)
            # Add the book to the returned_books list
            returned_books.append(borrowed_book)
            messagebox.showinfo("Success", f"{title} by {author} returned successfully.")
            
            # Clear the input fields after returning
            title_entry.delete(0, tk.END)
            author_entry.delete(0, tk.END)
            year_entry.delete(0, tk.END)
            student_id_entry.delete(0, tk.END)
            issue_date_entry.delete(0, tk.END)
        else:
            messagebox.showwarning("Book Not Borrowed", "This book was not borrowed or does not match the entered details.")
    else:
        messagebox.showwarning("Input Error", "Please fill in all fields to return a book.")
def borrow_book():
    title = title_entry.get()
    author = author_entry.get()
    pub_year = year_entry.get()
    student_id = student_id_entry.get()
    issue_date = issue_date_entry.get()

    if title and author and pub_year and student_id and issue_date:
        # Check if the book is already borrowed
        for borrowed in borrowed_books:
            if borrowed[0] == title and borrowed[1] == author and borrowed[2] == pub_year:
                messagebox.showwarning("Book Already Borrowed", "This book is already borrowed.")
                return
        
        # Add the book to borrowed_books list
        borrowed_books.append([title, author, pub_year, student_id, issue_date])
        messagebox.showinfo("Success", f"{title} by {author} borrowed successfully.")
        
        # Save transaction to CSV
        transaction_file = 'student_transactions.csv'
        new_transaction = pd.DataFrame([{
            "Student_Name": student_id,
            "Book_Title": title,
            "Borrow_Date": issue_date,
            "Return_Date": ""  # Can be updated later when book is returned
        }])

        if os.path.exists(transaction_file):
            existing_df = pd.read_csv(transaction_file)
            updated_df = pd.concat([existing_df, new_transaction], ignore_index=True)
        else:
            updated_df = new_transaction

        updated_df.to_csv(transaction_file, index=False)

        # Clear the input fields after borrowing
        title_entry.delete(0, tk.END)
        author_entry.delete(0, tk.END)
        year_entry.delete(0, tk.END)
        student_id_entry.delete(0, tk.END)
        issue_date_entry.delete(0, tk.END)
    else:
        messagebox.showwarning("Input Error", "Please fill in all fields to borrow a book.")


# Function to view books
def view_books(search_query=None):
    view_window = tk.Toplevel(root)
    view_window.title("View Books")
    view_window.configure(bg='lightblue')

    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        messagebox.showinfo("No Books", "No books available")
        return

    if df.empty:
        messagebox.showinfo("No Books", "No books available")
        return

    tree = ttk.Treeview(view_window, columns=("Title", "Author", "Publiser/Year", "Acc. No."), show='headings')
    tree.heading("Acc. No.", text="Acc. No.")
    tree.heading("Title", text="Title")
    tree.heading("Author", text="Author")
    tree.heading("Publiser/Year", text="Publiser/Year")
    tree.pack(fill=tk.BOTH, expand=True)

    if search_query:
        df = df[df.apply(lambda row: bool(re.search(search_query, row['Title'])) or bool(re.search(search_query, row['Publiser/Year'])), axis=1)]

    for i, row in df.iterrows():
        tree.insert("", tk.END, values=[row['Title'], row['Author'], row['Publiser/Year'], i+1])

# Function to search books
def search_books():
    search_query = search_entry.get().strip()

    if search_query:
        # Open a new window for displaying search results
        view_window = tk.Toplevel(root)
        view_window.title("Search Results")
        view_window.configure(bg='lightblue')

        try:
            df = pd.read_csv(csv_file)
        except FileNotFoundError:
            messagebox.showinfo("No Books", "No books available.")
            return

        # Using the search query to filter by Title or Author with case-insensitive matching
        search_pattern = re.compile(re.escape(search_query), re.IGNORECASE)

        # Create a treeview to show search results
        tree = ttk.Treeview(view_window, columns=("Title", "Author", "Publiser/Year", "Acc. No."), show='headings')
        tree.heading("Acc. No.", text="Acc. No.")
        tree.heading("Title", text="Title")
        tree.heading("Author", text="Author")
        tree.heading("Publiser/Year", text="Publiser/Year")
        tree.pack(fill=tk.BOTH, expand=True)

        found_books = False
        for index, row in df.iterrows():
            title = str(row['Title']) if pd.notna(row['Title']) else ''
            author = str(row['Author']) if pd.notna(row['Author']) else ''
            publisher_year = str(row['Publiser/Year']) if pd.notna(row['Publiser/Year']) else ''

            if search_pattern.search(title) or search_pattern.search(author):
                tree.insert("", tk.END, values=[title, author, publisher_year, index + 1])
                found_books = True
        
        if not found_books:
            messagebox.showinfo("No Results", "No books found matching your search query.")
    else:
        messagebox.showwarning("Input Error", "Please enter a search query.")



def delete_book():
    title = title_entry.get() # type: ignore
    author = author_entry.get() # type: ignore
    pub_year = year_entry.get() # type: ignore
    if title and author and pub_year:
        try:
            df = pd.read_csv(csv_file)
            df = df[~((df['Title'] == title) & (df['Author'] == author) & (df['Publisher/Year'] == pub_year))]
            df.to_csv(csv_file, index=False)
            messagebox.showinfo("Success", "Book deleted successfully")
            title_entry.delete(0, tk.END) # type: ignore
            author_entry.delete(0, tk.END) # type: ignore
            year_entry.delete(0, tk.END) # type: ignore
        except FileNotFoundError:
            messagebox.showwarning("File Error", "No book data file found")
    else:
        messagebox.showwarning("Input Error", "Please fill in all fields to delete a book")



def recommend_books_based_on_content(student_name, book_data_file='lms second.csv', transaction_file='student_transactions.csv', top_n=5):
    try:
        # Load book data
        df = pd.read_csv(book_data_file)

        # Fill missing values and prepare text for vectorization
        df['text'] = df['Title'].fillna('') + ' ' + df['Author'].fillna('') + ' ' + df['Publiser/Year'].fillna('')

        # Vectorize book text using TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(df['text'])

        # Load past student transactions
        if os.path.exists(transaction_file):
            transactions_df = pd.read_csv(transaction_file)
        else:
            return "No transaction history found."

        # Filter transactions for the given student
        borrowed_books = transactions_df[transactions_df['Student_Name'] == student_name]['Book_Title'].tolist()

        if not borrowed_books:
            return "No borrowed books found for this student."

        # Get indices of borrowed books
        borrowed_indices = [df[df['Title'] == title].index[0] for title in borrowed_books if title in df['Title'].values]

        if not borrowed_indices:
            return "Borrowed books not found in book database."

        # Compute similarity between borrowed books and all books
        borrowed_vectors = tfidf_matrix[borrowed_indices]
        cosine_sim = cosine_similarity(borrowed_vectors, tfidf_matrix)

        # Accumulate similarity scores
        similarity_scores = {}
        for scores in cosine_sim:
            for idx, score in enumerate(scores):
                if idx not in borrowed_indices:  # Exclude already borrowed
                    similarity_scores[idx] = similarity_scores.get(idx, 0) + score

        # Sort and get top N recommendations
        recommended_books = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
        recommendations = [df.iloc[idx]['Title'] for idx, _ in recommended_books[:top_n]]

        return recommendations if recommendations else "No recommendations found."

    except Exception as e:
        print(f"Error: {e}")
        return "Error occurred while generating recommendations."

# GUI components (Tkinter setup)
def recommend_books():
    student_name = student_id_entry.get() # type: ignore
    if student_name:
        recommended_books = recommend_books_based_on_content(student_name)
        if isinstance(recommended_books, str):
            messagebox.showinfo("Recommendation Error", recommended_books)
        else:
            messagebox.showinfo("Recommended Books", f"Books recommended for {student_name}: {', '.join(recommended_books)}")
    else:
        messagebox.showwarning("Input Error", "Please enter a valid Student ID.")

        
transaction_file = 'student_transactions.csv'
if os.path.exists(transaction_file): # type: ignore
    transaction_file_df = pd.read_csv(transaction_file) # type: ignore
else:
    transaction_df = pd.DataFrame(columns=["Student_Name","Book_Title","Borrow_Date","Return_Date"])
    transaction_file = "student_transaction.csv"
    borrow_date = datetime.today().strftime('%Y-%m-%d')
    return_date = (datetime.today() + timedelta(days=7)).strftime('%Y-%m-%d')  # Example: 7-day loan

def fetch_books_from_openlibrary_and_csv(query, book_data_file='lms_second.csv', max_results=20):
    try:
        # === Load and filter from CSV ===
        try:
            df_csv = pd.read_csv(book_data_file)

            # Only fill NA in string columns
            string_cols = df_csv.select_dtypes(include='object').columns
            df_csv[string_cols] = df_csv[string_cols].fillna('')

            # Filter matching rows
            csv_filtered = df_csv[
                df_csv['Title'].str.contains(query, case=False, na=False) |
                df_csv['Author'].str.contains(query, case=False, na=False) |
                df_csv['Publiser/Year'].str.contains(query, case=False, na=False)
            ]

            # Prepare CSV book data
            csv_books = csv_filtered[['Title', 'Author', 'Publiser/Year']].copy()
            csv_books.rename(columns={
                'Title': 'title',
                'Author': 'author',
                'Publiser/Year': 'publisher'
            }, inplace=True)
            csv_books['description'] = 'From local CSV'
            csv_books['rating'] = None
            csv_books['author_score'] = None

        except Exception as e:
            print(f"CSV Error: {e}")
            csv_books = pd.DataFrame(columns=['title', 'author', 'publisher', 'description', 'rating', 'author_score'])

        # === Fetch from OpenLibrary API ===
        api_books = []
        try: 
            fields = "title,author_name,publisher,subject"
            url = f"https://openlibrary.org/search.json?q={query}&limit={max_results}&fields={fields}"
            headers = {"User-Agent": "BookRecommender/1.0 (your_email@example.com)"}
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                data = response.json()
                for doc in data.get("docs", []):
                    api_books.append({
                        "title": doc.get("title", "N/A"),
                        "author": ", ".join(doc.get("author_name", ["Unknown"])) if doc.get("author_name") else "Unknown",
                        "publisher": ", ".join(doc.get("publisher", ["Unknown"])) if doc.get("publisher") else "Unknown",
                        "description": " ".join(doc.get("subject", [])) if doc.get("subject") else "From OpenLibrary",
                        "rating": None,
                        "author_score": None
                    })
            else:
                print("OpenLibrary API returned a non-200 status code.")

        except Exception as e:
            print(f"API Error: {e}")

        # Convert to DataFrame
        df_api = pd.DataFrame(api_books)

        # ✅ Add source labels before combining
        csv_books['source'] = 'CSV'
        df_api['source'] = 'API'

        # ✅ Combine both
        combined_books = pd.concat([csv_books, df_api], ignore_index=True)
        return combined_books

    except Exception as e:
        print(f"Unexpected error: {e}")
        return pd.DataFrame(columns=['title', 'author', 'publisher', 'description', 'rating', 'author_score', 'source'])



# --- 2. Recommend Books Based on Content ---
def recommend_books(user_input, books_df, top_n=5):
    books_df = books_df.dropna(subset=['description']).reset_index(drop=True)
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(books_df['description'])

    user_vec = vectorizer.transform([user_input])
    cos_sim = cosine_similarity(user_vec, tfidf_matrix).flatten()

    books_df['similarity'] = cos_sim
    books_df['normalized_rating'] = 0  # No ratings available
    books_df['author_score'] = 0  # No author scores either

    # Final score only based on similarity
    books_df['final_score'] = books_df['similarity']

    top_books = books_df.sort_values(by='final_score', ascending=False).head(top_n)
    return top_books[['title', 'author', 'publisher', 'similarity', 'final_score', 'source']]


# Function to handle the recommendation process
def get_recommendations():
    user_topic = entry.get()
    if not user_topic:
        messagebox.showwarning("Input Error", "Please enter a topic.")
        return
    try:
        books = fetch_books_from_openlibrary_and_csv(user_topic)
        if books.empty:
            messagebox.showinfo("No Results", "No books found for the given topic.")
            return
        recommendations = recommend_books(user_topic, books)
        display_recommendations(recommendations)
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Function to display recommendations in a new window
def display_recommendations(recommendations):
    result_window = tk.Toplevel(root)
    result_window.title("Book Recommendations")
    for idx, row in recommendations.iterrows():
        info = f"{row['title']} by {row['author']}\nPublisher: {row['publisher']}\nSimilarity Score: {row['similarity']:.2f} | Final Score: {row['final_score']:.2f}\n"
        label = tk.Label(result_window, text=info, justify="left", anchor="w")
        label.pack(fill="both", padx=10, pady=5)



def show_recommendations():
    student_name = student_id_entry.get()

    recommendations = recommend_books_based_on_content(student_name)

    if isinstance(recommendations, list):
        display_text = "Recommended Books Based on Past Borrowing:\n\n" + "\n".join(recommendations)
    else:
        display_text = recommendations  # Show error or fallback message

    recommendation_label.config(text=display_text) # type: ignore
# Function to handle book data (add, delete, view, etc.)
# Main window setup# ... [previous code with functions]


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def evaluate_recommendation_model(student_name, book_data_file='lms second.csv', transaction_file='student_transactions.csv', top_k=5):
    # Load book data and prepare TF-IDF text
    books_df = pd.read_csv(book_data_file)
    books_df['text'] = books_df['Title'].fillna('') + ' ' + books_df['Author'].fillna('') + ' ' + books_df['Publiser/Year'].fillna('')
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(books_df['text'])

    # Load transactions
    transactions_df = pd.read_csv(transaction_file)

    # Get books borrowed by the student
    student_transactions = transactions_df[transactions_df['Student_Name'] == student_name]
    actual_books = student_transactions['Book_Title'].dropna().unique().tolist()

    if not actual_books:
        print(f"No transaction history for student: {student_name}")
        return

    # Build student profile from borrowed books
    student_profile_text = ' '.join(actual_books)
    user_vec = vectorizer.transform([student_profile_text])

    # Compute cosine similarity
    similarity_scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
    books_df['similarity'] = similarity_scores

    # Get Top-K recommended books
    top_books = books_df.sort_values(by='similarity', ascending=False).head(top_k)
    recommended_books = top_books['Title'].tolist()

    # Compute evaluation metrics
    actual_set = set(actual_books)
    recommended_set = set(recommended_books)
    matched_books = actual_set.intersection(recommended_set)

    precision = len(matched_books) / len(recommended_set) if recommended_set else 0.0
    recall = len(matched_books) / len(actual_set) if actual_set else 0.0

    # Print evaluation
    print(f"\nEvaluation for student: {student_name}")
    print(f"Precision@{top_k}: {precision:.2f}")
    print(f"Recall@{top_k}: {recall:.2f}")
    print(f"Actual Books: {actual_books}")
    print(f"Recommended Books: {recommended_books}")
    print(f"Matched Books: {list(matched_books)}")

# 🔧 Call the function with student name



# Main window setup
root = tk.Tk()
root.title("Library Management System")
root.geometry("900x650")
root.configure(bg="#1E88E5")  # Blue background for the main window

# Top title label
top_label = tk.Label(root, text="Book Buzz", font=("Arial", 24, "bold"), bg="#1E88E5", fg="white")
top_label.grid(row=0, column=0, columnspan=3, pady=20)

# Book Title
title_label = tk.Label(root, text="Book Title:", font=("Arial", 12), bg="#1E88E5", fg="white")
title_label.grid(row=1, column=0, padx=10, pady=10, sticky="e")
title_entry = tk.Entry(root, width=30, font=("Arial", 12))
title_entry.grid(row=1, column=1, padx=10)

# Author
author_label = tk.Label(root, text="Author:", font=("Arial", 12), bg="#1E88E5", fg="white")
author_label.grid(row=2, column=0, padx=10, pady=10, sticky="e")
author_entry = tk.Entry(root, width=30, font=("Arial", 12))
author_entry.grid(row=2, column=1, padx=10)

# Year
year_label = tk.Label(root, text="Year:", font=("Arial", 12), bg="#1E88E5", fg="white")
year_label.grid(row=3, column=0, padx=10, pady=10, sticky="e")
year_entry = tk.Entry(root, width=30, font=("Arial", 12))
year_entry.grid(row=3, column=1, padx=10)

# Student ID
student_id_label = tk.Label(root, text="Student ID:", font=("Arial", 12), bg="#1E88E5", fg="white")
student_id_label.grid(row=4, column=0, padx=10, pady=10, sticky="e")
student_id_entry = tk.Entry(root, width=30, font=("Arial", 12))
student_id_entry.grid(row=4, column=1, padx=10)

# Issue Date
issue_date_label = tk.Label(root, text="Issue Date:", font=("Arial", 12), bg="#1E88E5", fg="white")
issue_date_label.grid(row=5, column=0, padx=10, pady=10, sticky="e")
issue_date_entry = tk.Entry(root, width=30, font=("Arial", 12))
issue_date_entry.grid(row=5, column=1, padx=10)

calendar_button = tk.Button(root, text="Select Date", font=("Arial", 12), bg="#4CAF50", fg="white", command=select_issue_date)
calendar_button.grid(row=5, column=2, padx=10, pady=10)

# Search
search_label = tk.Label(root, text="Search Book:", font=("Arial", 12), bg="#1E88E5", fg="white")
search_label.grid(row=6, column=0, padx=10, pady=10, sticky="e")
search_entry = tk.Entry(root, width=30, font=("Arial", 12))
search_entry.grid(row=6, column=1, padx=10)
search_button = tk.Button(root, text="Search", font=("Arial", 12), bg="#9C27B0", fg="white", command=search_books)
search_button.grid(row=6, column=2, padx=10, pady=10)

# Topic/subject entry for external API recommendations
topic_label = tk.Label(root, text="Enter a topic or subject:", font=("Arial", 12), bg="#1E88E5", fg="white")
topic_label.grid(row=7, column=0, padx=10, pady=10, sticky="e")
entry = tk.Entry(root, width=30, font=("Arial", 12))
entry.grid(row=7, column=1, padx=10)
topic_button = tk.Button(root, text="Get Recommendations", font=("Arial", 12), bg="#3F51B5", fg="white", command=get_recommendations)
topic_button.grid(row=7, column=2, padx=10, pady=10)

# Action buttons
add_button = tk.Button(root, text="Add Book", font=("Arial", 12), bg="#4CAF50", fg="white", command=add_book)
add_button.grid(row=8, column=0, padx=20, pady=15, sticky="ew")

delete_button = tk.Button(root, text="Delete Book", font=("Arial", 12), bg="#F44336", fg="white", command=delete_book)
delete_button.grid(row=8, column=1, padx=20, pady=15, sticky="ew")

borrow_button = tk.Button(root, text="Borrow Book", font=("Arial", 12), bg="#2196F3", fg="white", command=borrow_book)
borrow_button.grid(row=9, column=0, padx=20, pady=15, sticky="ew")

return_button = tk.Button(root, text="Return Book", font=("Arial", 12), bg="#FFC107", fg="black", command=return_book)
return_button.grid(row=9, column=1, padx=20, pady=15, sticky="ew")

view_button = tk.Button(root, text="View Books", font=("Arial", 12), bg="#673AB7", fg="white", command=view_books)
view_button.grid(row=10, column=0, padx=20, pady=15, sticky="ew")

view_borrowed_button = tk.Button(root, text="View Borrowed Books", font=("Arial", 12), bg="#009688", fg="white", command=view_borrowed_books)
view_borrowed_button.grid(row=10, column=1, padx=20, pady=15, sticky="ew")

recommend_button = tk.Button(root, text="Student Recommendations", font=("Arial", 12), bg="#FF5722", fg="white", command=recommend_books)
recommend_button.grid(row=11, column=0, columnspan=2, padx=20, pady=15, sticky="ew")

# Start the application
root.mainloop()







