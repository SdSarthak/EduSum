import requests
import PyPDF2
import gradio as gr
import os
import re
import time
from typing import Optional, Tuple, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Environment variables for API keys (replace with your keys or set them as environment variables)
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY", "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")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "bxF55MY5TaRN9r9tWoG16aHTxDhEGXMCWCbNFXwu")

# Constants
MAX_PDF_SIZE_MB = 10
MAX_TEXT_LENGTH = 10000  # Characters limit for API requests
SUMMARY_MAX_TOKENS = 300
CHATBOT_MAX_TOKENS = 500

def simple_fallback_response(query: str) -> str:
    """Provides a simplified fallback response."""
    try:
        url = "https://api.cohere.ai/v1/generate"
        headers = {"Authorization": f"Bearer {COHERE_API_KEY}", "Content-Type": "application/json"}
        data = {
            "model": "command-light",
            "prompt": f"Answer briefly: {query}",
            "max_tokens": 300,
            "temperature": 0.5,
        }
        response = requests.post(url, headers=headers, json=data, timeout=20)
        if response.status_code == 200:
            return response.json().get("generations", [{}])[0].get("text", "").strip()
        else:
            return "I'm having trouble connecting to my knowledge base. Please try again later."
    except Exception as e:
        logging.error(f"Fallback error: {str(e)}")
        return "I'm having trouble generating a response. Please try again later."

class AITeachingAssistant:
    """Main class for the AI/ML Virtual Teaching Assistant"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """Extracts text from a PDF file with error handling."""
        try:
            file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
            if file_size_mb > MAX_PDF_SIZE_MB:
                return f"Error: PDF file is too large ({file_size_mb:.1f}MB). Maximum size is {MAX_PDF_SIZE_MB}MB."
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                if len(reader.pages) == 0:
                    return "Error: The PDF file appears to be empty or corrupted."
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"
                if not text.strip():
                    return "Error: Could not extract any text from the PDF. The file might be scanned or image-based."
                if len(text) > MAX_TEXT_LENGTH:
                    logging.warning(f"PDF text truncated from {len(text)} to {MAX_TEXT_LENGTH} characters")
                    text = text[:MAX_TEXT_LENGTH] + "...[truncated due to length]"
                return text
        except Exception as e:
            logging.error(f"PDF extraction error: {str(e)}")
            return f"Error extracting PDF text: {str(e)}"

    @staticmethod
    def summarize_text(text: str) -> str:
        """Summarizes educational content using Cohere API with improved prompting."""
        if not text or text.startswith("Error:"):
            return text
        url = "https://api.cohere.ai/v1/generate"
        headers = {"Authorization": f"Bearer {COHERE_API_KEY}", "Content-Type": "application/json"}
        prompt = (
            "You are an educational content summarizer specialized in AI, ML, and LLMs. "
            "Create a concise, well-structured summary of the following educational content. "
            "Focus on key concepts, main ideas, and important details. "
            "Format the summary with clear sections and bullet points where appropriate.\n\n"
            f"Content to summarize:\n{text}"
        )
        data = {
            "model": "command",
            "prompt": prompt,
            "max_tokens": SUMMARY_MAX_TOKENS,
            "temperature": 0.4,
            "k": 0,
            "p": 0.75
        }
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            summary = result.get("generations", [{}])[0].get("text", "")
            summary = summary.strip()
            summary = re.sub(r"Content to summarize:.*?\n", "", summary, flags=re.DOTALL)
            return summary if summary else "No summary could be generated."
        except requests.exceptions.RequestException as e:
            logging.error(f"Cohere API error: {str(e)}")
            return f"Error connecting to summarization service: {str(e)}"
        except Exception as e:
            logging.error(f"Summarization error: {str(e)}")
            return f"Error during summarization: {str(e)}"

    @staticmethod
    def chatbot_response(query: str) -> str:
        """Fetches AI/ML explanations with improved handling for complete responses."""
        if not query or not query.strip():
            return "Please enter a question to get a response."
        url = "https://api.cohere.ai/v1/generate"
        headers = {"Authorization": f"Bearer {COHERE_API_KEY}", "Content-Type": "application/json"}
        prompt = (
            "You are an AI/ML educational assistant. Provide a complete, well-structured answer "
            "to the following question about artificial intelligence, machine learning, or language models. "
            "Make sure your answer is comprehensive but concise.\n\n"
            f"Question: {query.strip()}\n\n"
            "Answer:"
        )
        data = {
            "model": "command",
            "prompt": prompt,
            "max_tokens": 800,
            "temperature": 0.7,
            "k": 0,
            "stop_sequences": [],
            "return_likelihoods": "NONE"
        }
        try:
            response = requests.post(url, headers=headers, json=data, timeout=45)
            response.raise_for_status()
            result = response.json()
            answer = result.get("generations", [{}])[0].get("text", "")
            answer = answer.strip()
            if not answer:
                return "I couldn't generate a response to your question. Please try rephrasing it."
            return answer
        except requests.exceptions.RequestException as e:
            logging.error(f"API error: {str(e)}")
            return simple_fallback_response(query)
        except Exception as e:
            logging.error(f"Chatbot error: {str(e)}")
            return f"Error generating response: {str(e)}"

    @staticmethod
    def provide_resources(category: str = "all") -> str:
        """Returns curated AI/ML and LLM learning resources by category with improved formatting."""
        resources = {
            "books": [
                "📖 [Deep Learning Book by Ian Goodfellow et al.](https://www.deeplearningbook.org/) - Comprehensive free online book",
                "📖 [Neural Networks and Deep Learning by Michael Nielsen](http://neuralnetworksanddeeplearning.com/) - Excellent free online book",
                "📖 [Dive into Deep Learning](https://d2l.ai/) - Interactive deep learning book with code examples"
            ],
            "courses": [
                "📺 [Fast.ai Practical Deep Learning Course](https://course.fast.ai/) - Hands-on approach to deep learning",
                "📺 [Stanford CS224N: NLP with Deep Learning](http://web.stanford.edu/class/cs224n/) - Full lecture videos and materials",
                "📺 [Stanford CS231n: CNN for Visual Recognition](http://cs231n.stanford.edu/) - Computer vision fundamentals",
                "📺 [MIT 6.S191: Introduction to Deep Learning](http://introtodeeplearning.com/) - MIT's introductory course"
            ],
            "tutorials": [
                "📝 [Hugging Face Transformers Course](https://huggingface.co/course/) - Learn to use state-of-the-art models",
                "📝 [TensorFlow Tutorials](https://www.tensorflow.org/tutorials) - Official TensorFlow guides",
                "📝 [PyTorch Tutorials](https://pytorch.org/tutorials/) - Official PyTorch learning resources",
                "📝 [Kaggle Learn](https://www.kaggle.com/learn) - Free courses on ML, deep learning, and more"
            ],
            "llm_resources": [
                "🤖 [Papers with Code](https://paperswithcode.com/) - Find state-of-the-art ML papers with code",
                "🤖 [LLM University by Cohere](https://docs.cohere.com/docs/llmu) - Free LLM course",
                "🤖 [OpenAI Documentation](https://platform.openai.com/docs) - Learn about GPT models and capabilities",
                "🤖 [EleutherAI](https://www.eleuther.ai/) - Open AI research on large language models"
            ],
            "communities": [
                "👥 [AI Stack Exchange](https://ai.stackexchange.com/) - Q&A for AI researchers and practitioners",
                "👥 [r/MachineLearning](https://www.reddit.com/r/MachineLearning/) - Active Reddit community",
                "👥 [Hugging Face Community](https://discuss.huggingface.co/) - Forums for NLP and transformers",
                "👥 [Papers with Code Discord](https://discord.com/invite/paperswithcode) - Discussion community"
            ]
        }
        category = category.lower() if isinstance(category, str) else "all"
        try:
            if category == "all":
                output = "# AI/ML Learning Resources\n\n"
                for cat_name, items in resources.items():
                    display_name = cat_name.replace('_', ' ').title()
                    output += f"## {display_name}\n\n"
                    for item in items:
                        output += f"- {item}\n"
                    output += "\n"
                return output
            elif category in resources:
                display_name = category.replace('_', ' ').title()
                output = f"# {display_name} Resources\n\n"
                for item in resources[category]:
                    output += f"- {item}\n"
                return output
            else:
                available_cats = ", ".join(list(resources.keys()) + ["all"])
                return f"Category not found. Available categories: {available_cats}"
        except Exception as e:
            logging.error(f"Resource error: {str(e)}")
            return f"Error retrieving resources: {str(e)}"

# Generator functions for progress updates
def summarize_pdf_with_progress(pdf):
    """Process PDF summarization with progress updates."""
    assistant = AITeachingAssistant()
    if pdf is None:
        yield "Please upload a PDF file to summarize."
        return
    yield "<span class='loading'>⏳ Analyzing PDF file...</span>"
    time.sleep(0.5)
    yield "<span class='loading'>⏳ Extracting text from PDF...</span>"
    pdf_text = assistant.extract_text_from_pdf(pdf.name)
    if pdf_text.startswith("Error:"):
        yield pdf_text
        return
    yield "<span class='loading'>⏳ Generating summary...</span>"
    time.sleep(0.5)
    summary = assistant.summarize_text(pdf_text)
    yield summary

def chatbot_response_with_progress(query):
    """Process chatbot response with progress updates."""
    assistant = AITeachingAssistant()
    if not query or not query.strip():
        yield "Please enter a question to get a response."
        return
    yield "<span class='loading'>⏳ Processing your question...</span>"
    time.sleep(0.5)
    yield "<span class='loading'>⏳ Generating response...</span>"
    reply = assistant.chatbot_response(query)
    yield reply

def get_resources_with_progress(category):
    """Process resource retrieval with progress updates."""
    assistant = AITeachingAssistant()
    yield "<span class='loading'>⏳ Fetching resources...</span>"
    time.sleep(0.3)
    resources = assistant.provide_resources(category)
    yield resources

def create_interface():
    """Create and configure the Gradio web interface with improved layout and custom styling."""
    custom_css = """
    <style>
    body, .gradio-container, .gradio-app {
        font-family: 'Arial', sans-serif;
    }
    .card {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 20px;
        margin-bottom: 20px;
        background-color: #ffffff;
    }
    .custom-button {
        background: linear-gradient(90deg, #4776E6 0%, #8E54E9 100%);
        border: none;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .custom-button:hover {
        opacity: 0.9;
        transform: scale(1.05);
    }
    .output-container {
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        padding: 15px;
        background-color: #f9f9f9;
        min-height: 200px;
    }
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    .loading {
        animation: pulse 1.5s infinite;
        color: #4776E6;
        font-weight: bold;
    }
    .tab-active {
        border-bottom: 2px solid #4776E6;
    }
    </style>
    """
    with gr.Blocks(theme=gr.themes.Soft(), title="AI/ML Virtual Teaching Assistant") as iface:
        gr.HTML(custom_css)
        gr.Markdown(
            """
            # 📚 AI/ML Virtual Teaching Assistant
            Welcome! This tool helps you quickly understand AI, ML, and LLM concepts.
            Use the tabs below to switch between summarizing PDFs, asking questions, or exploring learning resources.
            """
        )
        with gr.Tabs():
            with gr.Tab("Summarize PDF"):
                with gr.Row():
                    with gr.Column(scale=5):
                        pdf_file = gr.File(label="Upload a PDF (Max 10MB)", file_types=[".pdf"])
                    with gr.Column(scale=2):
                        summarize_btn = gr.Button("Summarize", variant="primary", elem_id="summarize-btn")
                summary_output = gr.Markdown(elem_id="summary-output")
                summarize_btn.click(
                    fn=summarize_pdf_with_progress,
                    inputs=[pdf_file],
                    outputs=[summary_output]
                )
            with gr.Tab("Ask AI Chatbot"):
                with gr.Row():
                    with gr.Column(scale=5):
                        user_question = gr.Textbox(
                            label="Your Question", 
                            placeholder="Enter your question about AI, ML, or LLMs...",
                            lines=3
                        )
                    with gr.Column(scale=2):
                        ask_btn = gr.Button("Ask", variant="primary", elem_id="ask-btn")
                chatbot_output = gr.Markdown(elem_id="chatbot-output")
                ask_btn.click(
                    fn=chatbot_response_with_progress,
                    inputs=[user_question],
                    outputs=[chatbot_output]
                )
            with gr.Tab("Learning Resources"):
                with gr.Row():
                    with gr.Column(scale=5):
                        resource_category = gr.Radio(
                            choices=["All", "Books", "Courses", "Tutorials", "LLM Resources", "Communities"],
                            label="Select Resource Category",
                            value="all",
                            info="Choose a category to filter learning resources."
                        )
                    with gr.Column(scale=2):
                        resources_btn = gr.Button("Get Resources", variant="primary", elem_id="resources-btn")
                resources_output = gr.Markdown(elem_id="resources-output")
                resources_btn.click(
                    fn=get_resources_with_progress,
                    inputs=[resource_category],
                    outputs=[resources_output]
                )
        gr.Markdown(
            """
            ---
            **About This Tool:**  
            This virtual teaching assistant is built using Gradio and leverages Cohere's Command model to summarize text and answer questions.
            Explore the tabs to get summaries, ask questions, or find curated AI/ML learning resources.
            """
        )
    return iface

if __name__ == "__main__":
    iface = create_interface()
    iface.launch(share=False)
