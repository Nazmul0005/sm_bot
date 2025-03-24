import os
import re
import random
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
## Uncomment the following files if you're not using pipenv as your virtual environment manager
#from dotenv import load_dotenv, find_dotenv
#load_dotenv(find_dotenv())

DB_FAISS_PATH="vectorstore/db_faiss"

# Company information for quick responses
COMPANY_INFO = {
    "ceo": "The CEO of SM Technology is MD. Monir Hossain, who is also the CEO of the parent company, bdCalling IT.",
    "web_development_price": "Website development starts at $2,500, depending on project requirements.",
    "about_assistant": "I am an assistant developed to help you with your questions related to the portfolio, client feedback, and the management team of SM Technology.",
    "digital_marketing": "Digital marketing refers to the promotion of products or services using digital channels such as search engines, social media, email, and websites. It involves tactics such as search engine optimization, content marketing, email marketing, social media marketing, and online advertising to reach and engage with a target audience."
}

@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm(huggingface_repo_id, HF_TOKEN):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=.8,
        model_kwargs={"token":HF_TOKEN,
                      "max_length":"1024"}
    )
    return llm

def is_greeting(text):
    """Check if the input is a greeting."""
    greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", 
                "good evening", "howdy", "what's up", "hola", "namaste"]
    
    text_lower = text.lower().strip()
    
    for greeting in greetings:
        if text_lower == greeting or text_lower.startswith(greeting + " "):
            return True
    return False

def is_help_request(text):
    """Check if the input is asking for help."""
    help_phrases = ["can you help", "help me", "i need help", "assist me", 
                   "need assistance", "could you help", "support me"]
    
    text_lower = text.lower().strip()
    
    for phrase in help_phrases:
        if phrase in text_lower:
            return True
    return False

def is_about_you_question(text):
    """Check if the user is asking about the bot."""
    about_phrases = ["who are you", "what are you", "tell me about yourself", 
                    "what's your name", "what is your name", "what can you do"]
    
    text_lower = text.lower().strip()
    
    for phrase in about_phrases:
        if phrase in text_lower:
            return True
    return False

def is_appreciation(text):
    """Check if the user is expressing appreciation."""
    appreciation_phrases = ["thank", "thanks", "appreciate", "helpful", "good job", 
                           "well done", "great", "awesome", "excellent"]
    
    text_lower = text.lower().strip()
    
    for phrase in appreciation_phrases:
        if phrase in text_lower:
            return True
    return False

def is_specific_query(text):
    """Check if the query matches specific company information we have."""
    
    # Check for CEO information
    if re.search(r'\bceo\b|\bfounder\b|\bowner\b|\bleader\b', text.lower()):
        return "ceo"
        
    # Check for pricing questions
    if re.search(r'\bprice\b|\bcost\b|\brate\b|\bfee\b', text.lower()) and re.search(r'\bweb\b|\bwebsite\b|\bdevelopment\b', text.lower()):
        return "web_development_price"
        
    # Check for questions about the assistant
    if is_about_you_question(text):
        return "about_assistant"
        
    # Check for digital marketing
    if "digital marketing" in text.lower():
        return "digital_marketing"
        
    return None

def get_greeting_response():
    """Return a friendly greeting response."""
    greetings = [
        "Hello! How can I assist you today with SM Technology solutions?",
        "Hi there! I'm your SM Technology AI assistant. What can I do for you?",
        "Greetings! I'm here to help with your AI, mobile app, or web development needs.",
        "Hello! How may I help you with our technology solutions today?",
        "Hi! I'm your SM Technology virtual assistant. How can I support you?"
    ]
    
    return random.choice(greetings)

def get_help_response():
    """Return a response to a help request."""
    help_responses = [
        "I'd be happy to help! Please tell me what you need assistance with.",
        "I'm here to help you. Could you please specify what you're looking for?",
        "I'm ready to assist you. What specifically do you need help with?",
        "Sure thing! I'm here to help with your questions about SM Technology solutions. What would you like to know?",
        "I'm at your service. Please let me know how I can assist you with our technology services."
    ]
    
    return random.choice(help_responses)

def get_appreciation_response():
    """Return a response to user appreciation."""
    appreciation_responses = [
        "Thank you for your kind words! I'm glad I could be of assistance.",
        "I appreciate your feedback! Is there anything else I can help you with?",
        "Thank you! I'm here to make your experience with SM Technology as smooth as possible.",
        "You're welcome! Feel free to reach out if you need any further assistance.",
        "I'm delighted to hear that! Don't hesitate to ask if you have more questions."
    ]
    
    return random.choice(appreciation_responses)

def main():
    st.title("🤖 SM Technology AI Solutions")
    st.subheader("Your partner in cutting-edge AI, mobile app, and web development.")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])
    
    prompt = st.chat_input("Pass your prompt here")
    
    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})
        
        # Check for specific information queries first
        specific_info_key = is_specific_query(prompt)
        if specific_info_key:
            response = COMPANY_INFO.get(specific_info_key)
            st.chat_message('assistant').markdown(response)
            st.session_state.messages.append({'role':'assistant', 'content': response})
            return
        
        # Check for greetings
        elif is_greeting(prompt):
            response = get_greeting_response()
            st.chat_message('assistant').markdown(response)
            st.session_state.messages.append({'role':'assistant', 'content': response})
            return
        
        # Check for help requests
        elif is_help_request(prompt):
            response = get_help_response()
            st.chat_message('assistant').markdown(response)
            st.session_state.messages.append({'role':'assistant', 'content': response})
            return
        
        # Check for appreciation/feedback
        elif is_appreciation(prompt):
            response = get_appreciation_response()
            st.chat_message('assistant').markdown(response)
            st.session_state.messages.append({'role':'assistant', 'content': response})
            return
        
        # If none of the above, process with RAG
        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Keep your tone friendly, professional and helpful.
        Use a conversational style when responding.
        Be concise but informative in your responses.
        If the question is about SM Technology, focus on highlighting the company's strengths.
        
        Context: {context}
        Question: {question}
        
        Start the answer directly.
        """
       
        HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN=os.environ.get("HF_TOKEN")
        
        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return
                
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )
            
            response = qa_chain.invoke({'query': prompt})
            result = response["result"]
            
            # Post-process the result
            if result.strip().startswith("I don't know") or result.strip().startswith("I don't have"):
                result = f"I'm sorry, but {result.lower()} Would you like me to connect you with a team member who might have more information about this?"
            
            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({'role':'assistant', 'content': result})
            
        except Exception as e:
            error_message = f"I apologize, but I encountered an error while processing your request. Please try again or contact our support team at support@smtechnology.com if the issue persists."
            st.chat_message('assistant').markdown(error_message)
            st.session_state.messages.append({'role':'assistant', 'content': error_message})
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()