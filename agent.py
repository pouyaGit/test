# %%
# ==============================================================================
# SIMPLE FINANCIAL ANALYST CHATBOT WITH STREAMING AUDIO - MINIMAL PERSIAN UI
# ==============================================================================

import os
import re
import threading
import queue
import atexit
import glob
import gradio as gr
import numpy as np
import sounddevice as sd
from openai import OpenAI

# Try to import frontmatter, use fallback if not available
try:
    import frontmatter
    FRONTMATTER_AVAILABLE = True
    print("✅ Frontmatter support enabled")
except ImportError:
    FRONTMATTER_AVAILABLE = False
    print("⚠️ python-frontmatter not found. Using basic markdown loading.")

from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma



# %%
# ==============================================================================
# CONFIGURATION
# ==============================================================================

# API Configuration
API_KEY = "aaWFUJo6Ih9WpKpeMZNj48rPf168iVVYvZ"
BASE_URL = "https://api.avalai.ir/v1"

# Model Configuration
LLM_MODEL = "gpt-4o-mini"
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "onyx"
EMBEDDING_MODEL = "text-embedding-3-small"

# Database Configuration
DB_DIR = "vector_db"
KNOWLEDGE_BASE_PATH = "knowledge-base"

# Audio Configuration
SAMPLE_RATE = 24000
CHANNELS = 1
DTYPE = 'int16'
SENTENCE_BATCH_SIZE = 5  # Increased to reduce logs

# UI Configuration
SHOW_SOURCES = True
PLAIN_TEXT_RESPONSE = True

# --- FFmpeg Configuration ---
FFMPEG_PATH = "C:\\ffmpeg\\bin" 

# Configure FFmpeg
try:
    if FFMPEG_PATH and os.path.exists(FFMPEG_PATH):
        os.environ["PATH"] += os.pathsep + FFMPEG_PATH
        AudioSegment.converter = os.path.join(FFMPEG_PATH, "ffmpeg.exe")
        print("✅ FFmpeg configured successfully.")
    else:
        print("⚠️ Warning: FFmpeg path not found.")
except Exception as e:
    print(f"❌ Error configuring FFmpeg: {e}")
    
# System Rules
# SYSTEM_RULES = (
#     "You are a helpful financial analyst assistant. "
#     "All financial amounts mentioned in the documents and responses are in million IRR (Iranian Rial) "
#     "unless explicitly stated otherwise. Do not convert values to full Rial, Toman, or any other unit. "
#     "Only use a different unit if the document explicitly indicates a different scale with high certainty. "
#     "Provide clear, concise responses in plain text format."
# )
SYSTEM_RULES = (
    "You are an expert financial analyst assistant. Your task is to answer user questions based *only* on the provided documents. "
    "First, synthesize a coherent and comprehensive answer from all relevant document snippets. "
    "Then, ensure every piece of information is followed by its correct citation. "
    "If the documents do not contain the answer, you must clearly state that the information is not available in the knowledge base. "
    "All financial amounts are in million IRR unless stated otherwise. Do not change the units."
)

# ==============================================================================
# GLOBAL VARIABLES
# ==============================================================================

# Initialize clients
try:
    llm_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    tts_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    print("✅ Clients initialized successfully.")
except Exception as e:
    print(f"❌ Could not initialize clients: {e}")
    llm_client = None
    tts_client = None

# Audio streaming variables
sentence_queue = queue.Queue()
audio_worker_thread = None
is_audio_running = False
conversation_history = []

# Database variables
embeddings = None
vectorstore = None
retriever = None



# %%
# ==============================================================================
# DATABASE FUNCTIONS
# ==============================================================================

def load_markdown_documents(path):
    """Load all markdown files recursively"""
    print(f"🔎 Searching for markdown files in: {path}")
    
    all_md_files = glob.glob(os.path.join(path, "**/*.md"), recursive=True)
    documents = []
    
    for file_path in all_md_files:
        try:
            if FRONTMATTER_AVAILABLE:
                with open(file_path, 'r', encoding='utf-8') as f:
                    post = frontmatter.load(f)
                final_metadata = post.metadata.copy()
                final_metadata['source'] = file_path
                content = post.content
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                final_metadata = {'source': file_path}
            
            doc = Document(page_content=content, metadata=final_metadata)
            documents.append(doc)
            
        except Exception as e:
            print(f"⚠️ Error reading file {file_path}: {e}")
    
    print(f"📄 {len(documents)} documents loaded successfully.")
    return documents

def create_optimized_chunks(documents):
    """Create optimized chunks using two-stage splitting"""
    print("🔧 Starting optimized chunking process...")
    
    headers_to_split_on = [
        ("#", "title"),
        ("##", "section"), 
        ("###", "subsection"),
    ]
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    character_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    chunks = []
    for doc in documents:
        try:
            split_docs = header_splitter.split_text(doc.page_content)
            
            for split_doc in split_docs:
                combined_metadata = doc.metadata.copy()
                combined_metadata.update(split_doc.metadata)
                
                if len(split_doc.page_content) > 1200:
                    char_chunks = character_splitter.split_text(split_doc.page_content)
                    for chunk_content in char_chunks:
                        chunks.append(Document(page_content=chunk_content, metadata=combined_metadata))
                else:
                    chunks.append(Document(page_content=split_doc.page_content, metadata=combined_metadata))
        except Exception as e:
            print(f"⚠️ Error processing document: {e}")
            chunks.append(doc)
    
    print(f"✅ Created {len(chunks)} chunks total.")
    return chunks

def verify_chunks(chunk_list, num_samples=3):
    """Verify chunk quality"""
    if not chunk_list:
        print("❌ No chunks available for verification.")
        return
    
    print(f"\n🕵️‍♂️ Verifying chunk samples...\n" + "="*50)
    
    for i, chunk in enumerate(chunk_list[:num_samples]):
        print(f"\n--- Sample Chunk #{i+1} ---")
        print(f"📊 Metadata: {chunk.metadata}")
        print(f"📏 Content Size: {len(chunk.page_content)} characters")
        print(f"📝 Content Preview: \n'{chunk.page_content[:100]}...'")
        print("."*30)
    
    print(f"\n🔎 Checking for oversized chunks (>1200 chars)...")
    oversized_chunks = []
    
    for i, chunk in enumerate(chunk_list):
        if len(chunk.page_content) > 1200:
            oversized_chunks.append({
                "index": i,
                "size": len(chunk.page_content),
                "source": chunk.metadata.get('source', 'Unknown')
            })
    
    if oversized_chunks:
        print(f"🚨 Warning: {len(oversized_chunks)} oversized chunks found:")
        for item in oversized_chunks[:5]:
            print(f"  - Chunk #{item['index']} | Size: {item['size']} | Source: {os.path.basename(item['source'])}")
    else:
        print("✅ All chunks are within size limits.")

def setup_database():
    """Setup vector database"""
    global embeddings, vectorstore, retriever
    
    try:
        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=BASE_URL,
            api_key=API_KEY
        )
        
        if os.path.exists(DB_DIR):
            print("📦 Loading existing vector database...")
            vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
            print("✅ Database loaded successfully.")
        else:
            print("🆕 Creating new vector database...")
            
            if os.path.exists(KNOWLEDGE_BASE_PATH):
                documents = load_markdown_documents(KNOWLEDGE_BASE_PATH)
            else:
                print(f"⚠️ Knowledge base path not found: {KNOWLEDGE_BASE_PATH}")
                documents = []
            
            if not documents:
                print("⚠️ No documents found. Creating dummy database.")
                documents = [Document(
                    page_content="Sample content for empty database initialization.", 
                    metadata={"source": "dummy", "type": "placeholder"}
                )]
            
            chunks = create_optimized_chunks(documents)
            verify_chunks(chunks)
            
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=DB_DIR
            )
            
            print(f"✅ Vector database created with {len(chunks)} chunks.")
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
        
    except Exception as e:
        print(f"❌ Error setting up database: {e}")
        retriever = None

def get_relevant_context(query):
    """Retrieve relevant documents for a query"""
    if not retriever:
        return "", []
    
    try:
        relevant_docs = retriever.invoke(query)
        
        if not SHOW_SOURCES:
            context_parts = [doc.page_content for doc in relevant_docs]
            return "\n\n".join(context_parts), relevant_docs
        
        context_parts = []
        for i, doc in enumerate(relevant_docs):
            source = doc.metadata.get('source', 'Unknown')
            title = doc.metadata.get('title', '')
            section = doc.metadata.get('section', '')
            
            source_info = f"Source {i+1}: {os.path.basename(source)}"
            if title:
                source_info += f" - {title}"
            if section:
                source_info += f" ({section})"
            
            context_parts.append(f"[{source_info}]\n{doc.page_content}")
        
        context = "\n\n---\n\n".join(context_parts)
        return context, relevant_docs
        
    except Exception as e:
        print(f"❌ Error retrieving context: {e}")
        return "", []



# %%
# ==============================================================================
# AUDIO STREAMING FUNCTIONS
# ==============================================================================

def stream_and_play(text_to_play):
    """Stream and play audio for given text"""
    if not tts_client or not text_to_play:
        return
    
    # Reduced logging
    try:
        with sd.RawOutputStream(
            samplerate=SAMPLE_RATE, 
            channels=CHANNELS, 
            dtype=DTYPE
        ) as audio_stream:
            with tts_client.audio.speech.with_streaming_response.create(
                model=TTS_MODEL,
                voice=TTS_VOICE,
                response_format="pcm",
                input=text_to_play,
            ) as response:
                for chunk in response.iter_bytes(chunk_size=4096):
                    audio_stream.write(chunk)
    except Exception as e:
        print(f"❌ Audio streaming error: {e}")

def audio_playback_worker():
    """Worker thread for audio playback"""
    global is_audio_running
    print("✅ Audio playback worker started.")
    
    while is_audio_running:
        try:
            text_to_play = sentence_queue.get(timeout=1)
            if text_to_play is None:
                break
            stream_and_play(text_to_play)
            sentence_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"❌ Audio worker error: {e}")
    
    print("🛑 Audio playback worker shutting down.")

def start_audio_worker():
    """Start the audio worker thread"""
    global audio_worker_thread, is_audio_running
    
    if not is_audio_running:
        is_audio_running = True
        audio_worker_thread = threading.Thread(target=audio_playback_worker, daemon=True)
        audio_worker_thread.start()

def process_text_stream(text_delta, sentence_buffer, sentence_batch):
    """Process streaming text and batch sentences for audio"""
    sentence_buffer += text_delta
    
    match = re.search(r"([^.?!…\n]+(?:[.?!…\n]|\s$))", sentence_buffer)
    if match:
        sentence = match.group(0).strip()
        sentence_buffer = sentence_buffer[len(sentence):].lstrip()
        sentence_batch.append(sentence)
        
        if len(sentence_batch) >= SENTENCE_BATCH_SIZE:
            text_to_queue = " ".join(sentence_batch)
            # Reduced logging
            sentence_queue.put(text_to_queue)
            sentence_batch.clear()
    
    return sentence_buffer, sentence_batch

def finalize_audio_stream(sentence_buffer, sentence_batch):
    """Process any remaining text at the end of stream"""
    if sentence_batch:
        text_to_queue = " ".join(sentence_batch)
        sentence_queue.put(text_to_queue)
    
    if sentence_buffer.strip():
        sentence_queue.put(sentence_buffer.strip())



# %%
# ==============================================================================
# CHAT FUNCTIONS
# ==============================================================================

def build_messages(user_message, history):
    """Build message list for LLM"""
    messages = [{"role": "system", "content": SYSTEM_RULES}]
    
    for human, ai in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": ai})
    
    messages.append({"role": "user", "content": user_message})
    return messages

def format_response_with_context(user_message, context):
    """Format the final message with context if available"""
    if not context or not SHOW_SOURCES:
        return user_message
    
    return f"""Based on the following documents:

{context}

---

Please answer the user's question: {user_message}

Note: In your response, be sure to cite the sources you used."""

def chat_stream(message, history, audio_enabled):
    """Main chat function with streaming audio"""
    if not llm_client:
        yield "", history + [(message, "خطا: سیستم چت راه‌اندازی نشده است.")]
        return
    
    try:
        # Get relevant context from database
        context, relevant_docs = get_relevant_context(message)
        
        # Format final message
        if context and SHOW_SOURCES:
            final_message = format_response_with_context(message, context)
        else:
            final_message = message
        
        # Build messages for LLM
        messages = build_messages(final_message, history)
        
        # Start streaming
        llm_stream = llm_client.chat.completions.create(
            model=LLM_MODEL, 
            messages=messages, 
            stream=True
        )
        
        response_text = ""
        sentence_buffer = ""
        sentence_batch = []
        ui_history = history + [[message, ""]]
        
        # Process streaming response
        for chunk in llm_stream:
            if (hasattr(chunk, 'choices') and 
                len(chunk.choices) > 0 and 
                hasattr(chunk.choices[0], 'delta') and 
                hasattr(chunk.choices[0].delta, 'content') and 
                chunk.choices[0].delta.content):
                
                text_delta = chunk.choices[0].delta.content
                response_text += text_delta
                ui_history[-1][1] = response_text
                yield "", ui_history
                
                # Process audio if enabled
                if audio_enabled:
                    sentence_buffer, sentence_batch = process_text_stream(
                        text_delta, sentence_buffer, sentence_batch
                    )
        
        # Finalize audio stream
        if audio_enabled:
            finalize_audio_stream(sentence_buffer, sentence_batch)
        
        # Save to history
        conversation_history.append({
            'user': message,
            'assistant': response_text,
            'context_used': bool(context)
        })
        
    except Exception as e:
        print(f"❌ Chat stream error: {e}")
        error_message = f"خطا در پردازش پیام: {str(e)}"
        ui_history = history + [(message, error_message)]
        yield "", ui_history

def clear_chat():
    """Clear chat history and memory"""
    global conversation_history
    conversation_history.clear()
    print("🧹 Conversation history cleared.")
    return None, None

def shutdown_hook():
    """Shutdown hook for audio system"""
    global is_audio_running
    is_audio_running = False
    sentence_queue.put(None)

# ==============================================================================
# INITIALIZE SYSTEMS
# ==============================================================================

# Setup database
setup_database()

# Start audio worker
start_audio_worker()

# Register shutdown hook
atexit.register(shutdown_hook)

# ==============================================================================
# GRADIO INTERFACE - MINIMAL PERSIAN DESIGN
# ==============================================================================

custom_css = """
.main-container {
    max-width: 900px;
    margin: 0 auto;
    direction: rtl;
}

.minimal-header {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

.control-panel {
    background: #f8f9fa;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
    border: 1px solid #e9ecef;
}

.examples-section {
    background: #fff;
    border: 1px solid #dee2e6;
    border-radius: 8px;
}

.rtl-input {
    direction: rtl;
    text-align: right;
}

.settings-info {
    font-size: 12px;
    color: #6c757d;
    background: #f8f9fa;
    padding: 10px;
    border-radius: 5px;
    margin: 5px 0;
}
"""

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="🤖 تحلیلگر مالی هوشمند",
    css=custom_css
) as demo:
    
    # Minimal header - only bot name
    gr.Markdown("""
    <div class="minimal-header">
        <h1>تحلیلگر مالی هوشمند</h1>
    </div>
    """, elem_classes=["main-container"])
    
    # Main chat interface
    chatbot = gr.Chatbot(
        height=500,
        show_label=False,
        show_copy_button=True,
        avatar_images=("👤", "🤖"),
        rtl=True,
        elem_classes=["rtl-input"]
    )
    
    # Controls
    with gr.Row():
        audio_enabled = gr.Checkbox(
            label="🔊 پخش صوتی فعال", 
            value=True,
            info="پخش همزمان صدا با متن"
        )
    
    # Input area
    with gr.Row():
        msg_textbox = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="سوال خود را درباره گزارش‌های مالی بپرسید...",
            container=False,
            lines=2,
            elem_classes=["rtl-input"]
        )
        submit_button = gr.Button("ارسال", variant="primary", scale=1)
    
    with gr.Row():
        clear_button = gr.Button("پاک کردن گفتگو", variant="secondary")
    
    # Persian examples
    with gr.Accordion("💡 نمونه سوالات", open=False,):
        gr.Examples(
            examples=[
                "درآمد عملیاتی شرکت در سال گذشته چقدر بوده است؟",
                "سود خالص شرکت نسبت به سال قبل چه تغییری داشته؟",
                "تحلیل نسبت‌های مالی کلیدی شرکت را ارائه دهید",
                "وضعیت نقدینگی فعلی شرکت چگونه است؟",
                "تحلیل جریان نقدی شرکت را نشان دهید",
                "بدهی‌های شرکت در چه وضعیتی قرار دارد؟",
                "عملکرد مالی شرکت را با سال‌های قبل مقایسه کنید",
                "نرخ بازده سرمایه شرکت چقدر است؟",
                "وضعیت سهام و سرمایه شرکت چگونه است؟",
                "پیش‌بینی عملکرد مالی آینده شرکت چیست؟"
            ],
            inputs=msg_textbox,
            label="انتخاب سوال:"
        )
    
    # Technical specifications (moved to bottom)
    with gr.Accordion("⚙️ مشخصات فنی", open=False):
        gr.Markdown(f"""
        <div class="settings-info">
        <strong>تنظیمات فعلی:</strong><br>
        • مدل زبانی: {LLM_MODEL}<br>
        • مدل صوتی: {TTS_MODEL} با صدای {TTS_VOICE}<br>
        • مدل embedding: {EMBEDDING_MODEL}<br>
        • نمایش منابع: {'فعال' if SHOW_SOURCES else 'غیرفعال'}<br>
        • فرمت پاسخ: {'متن ساده' if PLAIN_TEXT_RESPONSE else 'مارک‌داون'}<br>
        • تعداد جملات در هر دسته صوتی: {SENTENCE_BATCH_SIZE}<br>
        • مسیر پایگاه داده: {DB_DIR}<br>
        • مسیر منابع اطلاعاتی: {KNOWLEDGE_BASE_PATH}
        </div>
        """)
    
    # Event handlers
    chat_inputs = [msg_textbox, chatbot, audio_enabled]
    chat_outputs = [msg_textbox, chatbot]
    
    msg_textbox.submit(chat_stream, chat_inputs, chat_outputs)
    submit_button.click(chat_stream, chat_inputs, chat_outputs)
    clear_button.click(clear_chat, None, [chatbot, msg_textbox], queue=False)

# ==============================================================================
# LAUNCH APPLICATION
# ==============================================================================

def find_free_port(start_port=7860, max_attempts=50):
    """Find a free port for the application"""
    import socket
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    return None

if __name__ == '__main__':
    print("🚀 راه‌اندازی تحلیلگر مالی هوشمند...")
    print(f"🎤 صدا: {TTS_MODEL} با صدای {TTS_VOICE}")
    print(f"🤖 چت: {LLM_MODEL}")
    print(f"📚 منابع: {'فعال' if SHOW_SOURCES else 'غیرفعال'}")
    print(f"📝 فرمت: {'متن ساده' if PLAIN_TEXT_RESPONSE else 'مارک‌داون'}")
    
    free_port = find_free_port()
    
    if free_port:
        print(f"🌐 راه‌اندازی روی پورت {free_port}")
        demo.queue().launch(
            inbrowser=True,
            server_port=free_port,
            share=False,
            quiet=False
        )
    else:
        print("🌐 استفاده از انتخاب خودکار پورت...")
        demo.queue().launch(inbrowser=True, share=False)


