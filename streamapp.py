import streamlit as st
import mysql.connector
from mysql.connector import Error
import bcrypt

# Connect to MySQL
try:
    connection = mysql.connector.connect(
        host='localhost',
        database='user_authentication_db',
        user='root',
        password='root'
    )

    if connection.is_connected():
        db_info = connection.get_server_info()
        st.write(f"Connected to MySQL Server version {db_info}")

except Error as e:
    st.error(f"Error while connecting to MySQL: {e}")

# Streamlit UI for login
def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_button = st.button("Login")

    if login_button:
        try:
            cursor = connection.cursor()
            cursor.execute("SELECT password FROM users WHERE username = %s", (username,))
            user = cursor.fetchone()
            if user:
                hashed_password = user[0].encode('utf-8')
                if bcrypt.checkpw(password.encode('utf-8'), hashed_password):
                    st.success(f"Logged in as {username}")
                else:
                    st.error("Invalid username or password")
            else:
                st.error("Invalid username or password")
        except Error as e:
            st.error(f"Error while querying MySQL: {e}")

# Streamlit UI for signup
def signup():
    st.title("Signup")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    signup_button = st.button("Signup")

    if signup_button:
        hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
        hashed_password_str = hashed_password.decode('utf-8')  # Convert bytes to string
        try:
            cursor = connection.cursor()
            cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (new_username, hashed_password_str))
            connection.commit()
            st.success("Signup successful! You can now login.")
        except Error as e:
            st.error(f"Error while inserting into MySQL: {e}")

# Main App
def main():
    st.title("Login/Signup App")
    mode = st.radio("Choose an option", ("Login", "Signup"))

    if mode == "Login":
        login()
    elif mode == "Signup":
        signup()

if __name__ == "__main__":
    main()
