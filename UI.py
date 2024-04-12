import streamlit as st
import time
import front
from front import eyetrack

# Define the username and password
CORRECT_USERNAME = "user"
CORRECT_PASSWORD = "123"

# Define MCQ questions and options
MCQ_QUESTIONS = {
    "What is the capital of France?": ["Paris", "London", "Berlin", "Rome"],
    "What is the largest planet in our solar system?": ["Jupiter", "Saturn", "Mars", "Earth"]
}
CORRECT_ANSWERS = {
    "What is the capital of France?": "Paris",
    "What is the largest planet in our solar system?": "Jupiter"
}

def login():
    st.title("Login")
    username = st.text_input("Username", key="username_input_login")
    password = st.text_input("Password", type="password", key="password_input_login")
    if st.button("Login", key="login_button"):
        if username == CORRECT_USERNAME and password == CORRECT_PASSWORD:
            st.session_state.logged_in = True
            st.experimental_rerun()  # Rerun the app to go to the home page
            st.success("Logged in successfully!")
        else:
            st.error("Invalid username or password.")
            st.session_state.logged_in = False

def home():
    st.title("Home")
    st.write("Welcome to the Eye Tracking App!")
    if st.button("Begin Test", key="begin_test_button"):
        st.session_state.current_page = "Test"
        st.session_state.start_time = time.time()  # Initialize start time when test begins


def test():
    st.title("Test")
    st.write("This is the eye tracking test page.")
    elapsed_time_placeholder = st.empty()  # Create a placeholder for the timer
    answers = {}
    for question, options in MCQ_QUESTIONS.items():
        answer = st.radio(question, options, key=f"{question}_radio")  # Use unique key for radio buttons
        answers[question] = answer

    st.write("")  # Add some space before the submit button
    if st.button("Submit", key="submit_button"):  # Add a submit button
        st.session_state.current_page = "Results"
        st.session_state.test_answers = answers
        st.experimental_rerun()  # Rerun the app to go to the results page

    # Display elapsed time and webcam feed while the test is ongoing
    elapsed_time = 0
    while st.session_state.current_page == "Test":
        elapsed_time = round(time.time() - st.session_state.start_time, 2)
        elapsed_time_placeholder.text("Elapsed Time: {} seconds".format(elapsed_time))
        frame = front.eyetrack()  # Update the webcam feed
        st.image(frame, channels="RGB")  # Display the webcam feed
        time.sleep(1)  # Update the timer every second

def show_results():
    st.title("Test Results")
    st.write("Here are the correct answers:")
    for question, correct_answer in CORRECT_ANSWERS.items():
        st.write(f"{question}: {correct_answer}")

def main():
    # Initialize session state variables
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Login"

    if not st.session_state.logged_in:
        login()
    else:
        if st.session_state.current_page == "Login":
            st.session_state.current_page = "Home"

    if st.session_state.logged_in:
        if st.session_state.current_page == "Home":
            home()
        elif st.session_state.current_page == "Test":
            test()
        elif st.session_state.current_page == "Results":
            show_results()

if __name__ == "__main__":
    main()

