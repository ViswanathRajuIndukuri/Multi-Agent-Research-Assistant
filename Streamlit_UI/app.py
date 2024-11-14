import streamlit as st
import requests
import os
from fpdf import FPDF
import tempfile

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Set page layout at the top
st.set_page_config(page_title="Research Agent", page_icon="📄", layout="wide")

# Initialize session state variables
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'token' not in st.session_state:
    st.session_state.token = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'page' not in st.session_state:
    st.session_state.page = "home"
if 'selected_index' not in st.session_state:
    st.session_state.selected_index = None
if 'prev_selected_index' not in st.session_state:
    st.session_state.prev_selected_index = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'user_input' not in st.session_state:
    st.session_state.user_input = ''
# Add to your existing session state initialization
if 'query_count' not in st.session_state:
    st.session_state.query_count = 0
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []

def create_qa_pdf():
    pdf = FPDF()
    pdf.add_page()
    
    # Set font for title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Research Query Report", ln=True, align='C')
    pdf.ln(10)
    
    # Set font for content
    pdf.set_font("Arial", size=12)
    
    for qa in st.session_state.qa_history:
        # Add question
        pdf.set_font("Arial", "B", 12)
        pdf.multi_cell(0, 10, f"Q: {qa['question']}")
        pdf.ln(5)
        
        # Add answer
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 10, f"A: {qa['answer']}")
        pdf.ln(10)
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as fp:
        pdf.output(fp.name)
        return fp.name


def initial_home():
    # Apply custom CSS to center the content and adjust chat message font sizes
    st.markdown(
        """
        <style>
        /* Center the main content */
        .main .block-container {
            max-width: 80%;
            padding-top: 1rem; /* Reduced padding */
            padding-left: 5%;
            padding-right: 5%;
            margin-left: auto;
            margin-right: auto;
        }
        /* Style for features list */
        .features {
            font-size: 1rem; /* Reduced from 1.2rem */
            margin-left: 1.5rem; /* Reduced margin */
        }
        .features li {
            margin-bottom: 0.3rem; /* Reduced spacing */
        }
        /* Style for the footer */
        .footer {
            text-align: center;
            font-size: 0.9rem; /* Reduced font size */
            color: #888888;
            margin-top: 1.5rem; /* Reduced margin */
        }
        /* Style for buttons */
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            padding: 0.4rem 0.8rem; /* Reduced padding */
            margin: 0.4rem 0;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .stButton > button:hover {
            background-color: #45a049;
        }
        /* Adjust font size in chat messages */
        div[data-testid="stChatMessage"] {
            font-size: 16px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Create columns to position the login/logout button at the top right
    top_col1, top_col2 = st.columns([8, 1])  # Adjusted column widths

    def go_to_signin():
        st.session_state.page = "signin"

    def logout():
        st.session_state.logged_in = False
        st.session_state.token = None
        st.session_state.username = None
        st.session_state.page = "home"
        st.session_state.selected_index = None
        st.session_state.prev_selected_index = None
        st.session_state.history = []
        st.session_state.user_input = ''
        st.session_state.query_count = 0  # Reset query count
        st.session_state.qa_history = []  # Reset QA history

    with top_col2:
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        if st.session_state.logged_in:
            st.button("🚪 Logout", key="logout_button", on_click=logout)
        else:
            st.button("🔒 Login", key="login_button", on_click=go_to_signin)

    # Display content based on login status
    if st.session_state.logged_in:
        # Top heading
        st.markdown(
            f'<div style="text-align: center; font-size: 2.5rem;">Welcome, {st.session_state.username}!</div>',
            unsafe_allow_html=True)
        st.markdown(
            '<div style="text-align: center; font-size: 2rem;">Your Research Assistant</div>',
            unsafe_allow_html=True)
        st.write("---")  # Optional separator

        # Display the index selection
        st.title("Select Index")

        headers = {
            "Authorization": f"Bearer {st.session_state.token}"
        }
        response = requests.get(f"{API_URL}/pinecone-indexes", headers=headers)

        if response.status_code == 200:
            index_list = response.json().get("indexes", [])
            if index_list:
                selected_index = st.selectbox(
                    "Pinecone Indexes:",
                    index_list,
                    key='selected_index'
                )
                st.write(f"You selected: **{selected_index}**")

                # Check if the selected index has changed
                if st.session_state.selected_index != st.session_state.prev_selected_index:
                    st.session_state.history = []  # Reset chat history
                    st.session_state.query_count = 0  # Reset query count
                    st.session_state.qa_history = []  # Reset QA history
                    st.session_state.prev_selected_index = st.session_state.selected_index
            else:
                st.info("No indexes available in Pinecone.")
        elif response.status_code == 401:
            st.error("Your session has expired. Please log in again.")
            # Log the user out
            logout()
        else:
            error_detail = response.json().get('detail', 'Unknown error')
            st.error(f"Failed to retrieve Pinecone indexes: {error_detail}")

        # Check if an index is selected
        if st.session_state.get('selected_index'):
            st.write("---")  # Separator
            st.title("Chat with Assistant")

            # Display query count
            remaining_queries = 5 - st.session_state.query_count
            st.info(f"Remaining queries: {remaining_queries}")

            # Input box for user message
            user_input = st.chat_input("Enter your query:", 
                              disabled=remaining_queries <= 0)

            # Display chat messages
            chat_container = st.container()
            with chat_container:
                for message in st.session_state.history:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

            if user_input and remaining_queries > 0:
                with st.chat_message("user"):
                    st.markdown(user_input)
                st.session_state.history.append({"role": "user", "content": user_input})

                # Call the backend API to get assistant's reply
                data = {
                    "query": user_input,
                    "index_name": st.session_state.selected_index  # Use the session state variable
                }
                headers = {
                    "Authorization": f"Bearer {st.session_state.token}",
                    "Content-Type": "application/json"
                }
                research_response = requests.post(
                    f"{API_URL}/run-research-query",
                    json=data,
                    headers=headers
                )

                if research_response.status_code == 200:
                    result = research_response.json().get("result", "")

                    # Display the assistant's response
                    with st.chat_message("assistant"):
                        st.markdown(result)
                    st.session_state.history.append({"role": "assistant", "content": result})

                    # Store Q&A in history
                    st.session_state.qa_history.append({
                        "question": user_input,
                        "answer": result
                    })
                    
                    # Increment query count
                    st.session_state.query_count += 1


                elif research_response.status_code == 401:
                    st.error("Your session has expired. Please log in again.")
                    # Log the user out
                    logout()
                else:
                    error_detail = research_response.json().get('detail', 'Unknown error')
                    st.error(f"Failed to run research query: {error_detail}")

            if st.session_state.qa_history:
                if st.button("Export to PDF"):
                    pdf_path = create_qa_pdf()
                    with open(pdf_path, "rb") as pdf_file:
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_file,
                            file_name="research_report.pdf",
                            mime="application/pdf"
                        )
                # else:
                #     st.info("Please select an index above to start chatting.")

    else:
        # The rest of your code for the non-logged-in user remains the same
        st.markdown(
            '<div style="text-align: center; font-size: 3.5rem; font-weight: bold;">Research Assistant Using LangGraph</div>',
            unsafe_allow_html=True)
        st.markdown(
            '<div style="text-align: center; font-size: 2.5rem;">Your AI-powered Research Assistant</div>',
            unsafe_allow_html=True)
        st.write("---")
        st.markdown('''
                <div style="text-align: center;">
                    <div style="margin-bottom: 1rem;">
                        📚 <strong>Document Selection</strong>: Choose from pre-processed documents for research.
                    </div>
                    <div style="margin-bottom: 1rem;">
                        🔬 <strong>Arxiv Agent</strong>: Search and retrieve relevant research papers.
                    </div>
                    <div style="margin-bottom: 1rem;">
                        🌐 <strong>Web Search Agent</strong>: Conduct online research for broader context.
                    </div>
                    <div style="margin-bottom: 1rem;">
                        💡 <strong>RAG Agent</strong>: Answer queries based on document content using Pinecone and Langraph.
                    </div>
                    <div style="margin-bottom: 1rem;">
                        📝 <strong>User Interaction Interface</strong>: Ask questions and interact with research findings.
                    </div>
                    <div style="margin-bottom: 1rem;">
                        📄 <strong>Export Results</strong>: Export findings as professional PDF reports or in Codelabs format.
                    </div>
                </div>
                ''', unsafe_allow_html=True)
        st.write("---")
        st.markdown(
            '<div style="text-align: center; font-size: 2.5rem;">🚀 Get Started</div>',
            unsafe_allow_html=True)
        st.markdown('<p style="text-align:center;">Click the <strong>🔒 Login</strong> button at the top right to sign in or create a new account.</p>', unsafe_allow_html=True)
        st.markdown('<div class="footer">© 2024 Research Assistant. All rights reserved.</div>', unsafe_allow_html=True)

def register():
    # Add a "Back" button at the top right to return to the sign-in page
    col1, col2 = st.columns([8, 1.5])  # Adjusted column widths

    with col1:
        st.markdown(
            '<div style="text-align: left; font-size: 3rem; font-weight: bold; margin-top: 1rem; margin-bottom: 1rem;">🔑 Sign Up</div>',
            unsafe_allow_html=True)
    with col2:
        st.button("🔙 Back", on_click=lambda: st.session_state.update(page="signin"))

    # Use unique keys for input fields
    st.write("Please fill in the details below to create a new account.")
    email = st.text_input("📧 Email", key="signup_email")
    username = st.text_input("👤 Username", key="signup_username")
    password = st.text_input("🔒 Password", type="password", key="signup_password")
    password_confirm = st.text_input("🔒 Confirm Password", type="password", key="signup_password_confirm")

    # Define callback functions
    def handle_register():
        if not email or not username or not password or not password_confirm:
            st.warning("⚠️ Please fill in all fields")
        elif password != password_confirm:
            st.warning("⚠️ Passwords do not match")
        else:
            response = requests.post(
                f"{API_URL}/register",
                json={
                    "email": email,
                    "username": username,
                    "password": password
                }
            )
            if response.status_code == 200:
                st.success("🎉 Registration successful! Please sign in.")
                # Clear the input fields
                st.session_state.signup_email = ''
                st.session_state.signup_username = ''
                st.session_state.signup_password = ''
                st.session_state.signup_password_confirm = ''
                st.session_state.page = "signin"
            else:
                error_detail = response.json().get('detail', 'Unknown error')
                st.error(f"❌ Registration failed: {error_detail}")

    # Registration button
    st.button("✅ Register", on_click=handle_register)

def login():
    # Add a "Back" button at the top right to return to the welcome page
    col1, col2 = st.columns([8, 1.5])  # Adjusted column widths

    with col1:
        st.markdown(
            '<div style="text-align: left; font-size: 3rem; font-weight: bold; margin-top: 1rem; margin-bottom: 1rem;">🔐 Sign In</div>',
            unsafe_allow_html=True)
    with col2:
        st.button("🔙 Back", on_click=lambda: st.session_state.update(page="home"))

    st.write("Please enter your credentials to sign in.")

    # Use unique keys for input fields
    username = st.text_input("👤 Username", key="signin_username")
    password = st.text_input("🔒 Password", type="password", key="signin_password")

    # Define callback functions
    def handle_login():
        if not username or not password:
            st.warning("⚠️ Please enter both username and password")
        else:
            response = requests.post(
                f"{API_URL}/login",
                data={
                    "username": username,
                    "password": password
                }
            )
            if response.status_code == 200:
                token = response.json().get("access_token")
                st.session_state.token = token
                st.session_state.logged_in = True
                st.session_state.username = username
                # Clear the input fields
                st.session_state.signin_username = ''
                st.session_state.signin_password = ''
                st.session_state.page = "home"
                st.success("✅ Logged in successfully!")
            else:
                error_detail = response.json().get('detail', 'Unknown error')
                st.error(f"❌ Login failed: {error_detail}")

    def go_to_signup():
        # Clear the input fields when switching pages
        st.session_state.signin_username = ''
        st.session_state.signin_password = ''
        st.session_state.page = "signup"

    st.button("➡️ Sign In", on_click=handle_login)
    st.write("Don't have an account?")
    st.button("📝 Sign Up", on_click=go_to_signup)

def main():
    if st.session_state.page == "home":
        initial_home()
    elif st.session_state.page == "signin":
        login()
    elif st.session_state.page == "signup":
        register()

if __name__ == "__main__":
    main()
