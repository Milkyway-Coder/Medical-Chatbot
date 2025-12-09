import os
import logging
import time
import threading
import asyncio
from flask import Flask, render_template, request, session, jsonify
from werkzeug.utils import secure_filename
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from google import genai
from google.genai import types
from dotenv import load_dotenv
from src.prompt import system_prompt
from collections import defaultdict
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get API keys
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in environment")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY not found in environment")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB

# Store Telegram conversation history per user
telegram_conversations = {}

# Rate limiting
user_requests = defaultdict(list)
MAX_REQUESTS_PER_MINUTE = 10

# Initialize Gemini client
logger.info("Initializing Gemini client...")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Initialize embeddings and Pinecone
logger.info("Loading embeddings and connecting to Pinecone...")
embeddings = download_hugging_face_embeddings()
docsearch = PineconeVectorStore.from_existing_index(
    index_name="medical-chatbot",
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

GEMINI_MODEL = "gemini-2.5-flash"
logger.info("✅ Medical Chatbot initialized successfully!")


# ==================== SHARED FUNCTIONS ====================

def check_rate_limit(user_id):
    """Simple rate limiting"""
    now = datetime.now()
    user_requests[user_id] = [
        req_time for req_time in user_requests[user_id]
        if now - req_time < timedelta(minutes=1)
    ]
    
    if len(user_requests[user_id]) >= MAX_REQUESTS_PER_MINUTE:
        return False
    
    user_requests[user_id].append(now)
    return True


def detect_emergency(message: str) -> dict:
    """Detect if message contains emergency keywords"""
    emergency_keywords = {
        'critical': [
            'chest pain', 'heart attack', 'stroke', 'can\'t breathe', 'cannot breathe',
            'severe bleeding', 'unconscious', 'seizure', 'overdose', 'poisoning', 
            'severe burn', 'choking', 'not breathing', 'cardiac arrest', 'anaphylaxis'
        ],
        'urgent': [
            'difficulty breathing', 'severe pain', 'high fever', 'vomiting blood', 
            'severe headache', 'blurred vision', 'confusion', 'severe abdominal pain',
            'broken bone', 'deep cut', 'blood loss', 'allergic reaction'
        ],
        'mental_health': [
            'suicide', 'kill myself', 'want to die', 'self harm', 'end my life',
            'hurt myself', 'no reason to live', 'better off dead', 'suicidal'
        ]
    }
    
    message_lower = message.lower()
    
    for severity, keywords in emergency_keywords.items():
        for keyword in keywords:
            if keyword in message_lower:
                return {
                    'is_emergency': True,
                    'severity': severity,
                    'keyword': keyword
                }
    
    return {'is_emergency': False}


def is_meta_question(message: str) -> bool:
    """Check if the question is about the conversation itself"""
    meta_keywords = [
        'previous question', 'what did i ask', 'my last question',
        'earlier', 'before', 'previously asked', 'conversation',
        'what were we talking', 'remind me', 'we discussed',
        'you said', 'you mentioned', 'you told me'
    ]
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in meta_keywords)


def retrieve_context(query):
    """Retrieve relevant context from Pinecone"""
    try:
        docs = retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        logger.warning(f"⚠️ Retrieval error: {e}")
        return ""


def generate_response(user_message, chat_history, max_retries=2):
    """Generate response using Gemini API with RAG"""
    for attempt in range(max_retries):
        try:
            logger.info(f"🔄 Attempt {attempt + 1}/{max_retries}")
            
            # Check if this is a meta question about the conversation
            is_meta = is_meta_question(user_message)
            
            if is_meta and len(chat_history) > 0:
                logger.info("Detected meta-question about conversation history")
            
            # Retrieve context
            context = retrieve_context(user_message)
            
            # Build history text
            history_text = ""
            if chat_history:
                history_text = "\n\nCONVERSATION HISTORY:\n"
                for msg in chat_history[-6:]:
                    role = "User" if msg['role'] == 'user' else "Assistant"
                    history_text += f"{role}: {msg['content']}\n"
            
            # Enhanced system prompt with conversation recall
            enhanced_system_prompt = system_prompt + """

IMPORTANT INSTRUCTIONS:

1. CONVERSATION MEMORY:
You have access to the full conversation history. When users ask about:
- "What did I ask before?" or "My previous questions"
- "What were we talking about?"
- "Can you remind me what you said about X?"
- Follow-up questions like "Tell me more about that" or "What about the previous topic?"

You MUST:
- Review the chat_history to recall what was discussed
- Provide specific details about previous questions and answers
- Reference earlier parts of the conversation naturally
- Maintain context across the entire conversation

2. CONTEXTUAL UNDERSTANDING:
- When a question seems incomplete or vague (like "what's the dosage?" or "tell me more"), ALWAYS check the chat history
- Use previous messages to understand what the user is referring to
- If they asked about a medication before, and now ask "what's the dosage?", understand they mean the dosage of that medication
- Connect follow-up questions to the previous context automatically

3. UNCLEAR QUESTIONS:
If a question is genuinely unclear and you cannot understand it even with full chat history:
- Politely ask for clarification
- Do NOT generate errors or technical messages
- Say something like: "I'm not sure I understood your question correctly. Could you please provide more details or rephrase it?"
- Suggest what additional information would help

4. ERROR HANDLING:
- NEVER show technical errors to users
- If you cannot find relevant information, say: "I don't have enough information to answer that question accurately. Could you please provide more context?"
- Always maintain a helpful, friendly tone

Remember: Your goal is to be helpful and understand context. Use the conversation history to make sense of follow-up questions."""
            
            # Create prompt
            full_prompt = f"""You are a helpful medical assistant with access to a medical knowledge base and Google Search.

RELEVANT MEDICAL CONTEXT:
{context if context else "No specific context retrieved."}

{enhanced_system_prompt}
{history_text}

User Question: {user_message}

Instructions: 
1. Consider the medical context provided
2. Use Google Search for current information
3. Provide detailed, accurate answers
4. Include relevant details and treatment options
5. If this is a follow-up question, reference previous context naturally
6. ALWAYS include appropriate medical disclaimers and safety warnings"""
            
            # Configure Gemini
            contents = [types.Content(role="user", parts=[types.Part.from_text(text=full_prompt)])]
            tools = [types.Tool(googleSearch=types.GoogleSearch())]
            config = types.GenerateContentConfig(
                tools=tools,
                temperature=0.7,
                max_output_tokens=4096,
            )
            
            logger.info("🔍 Generating response with Gemini...")
            
            # Generate response
            response_text = ""
            chunk_count = 0
            for chunk in gemini_client.models.generate_content_stream(
                model=GEMINI_MODEL,
                contents=contents,
                config=config,
            ):
                if chunk.text:
                    response_text += chunk.text
                    chunk_count += 1
            
            logger.info(f"📦 Received {chunk_count} chunks")
            
            if response_text.strip():
                logger.info(f"✅ Response generated successfully ({len(response_text.strip())} chars)")
                return response_text.strip()
            
            logger.warning("⚠️ Empty response received")
            if attempt < max_retries - 1:
                logger.info("🔄 Retrying...")
                time.sleep(2)
                continue
            return "I couldn't generate a response. Please try again."
                    
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Error (Attempt {attempt + 1}): {error_msg}")
            
            # Check if it's a connection error
            if "disconnect" in error_msg.lower() or "connection" in error_msg.lower() or "timeout" in error_msg.lower():
                if attempt < max_retries - 1:
                    logger.info(f"🔄 Connection error, retrying in 3 seconds...")
                    time.sleep(3)
                    continue
                else:
                    return "⏱️ I'm experiencing connection issues with the AI service. Please try again in a moment."
            
            if attempt < max_retries - 1:
                logger.info(f"🔄 Retrying after error...")
                time.sleep(2)
                continue
            
            import traceback
            traceback.print_exc()
            return "I encountered an error. Please try again."
    
    return "⏱️ Service temporarily unavailable. Please try again in a moment."


# ==================== FLASK ROUTES ====================

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["POST"])
def chat():
    try:
        msg = request.form.get("msg", "").strip()
        if not msg:
            return "Please provide a message.", 400
        
        # Rate limiting
        user_id = request.remote_addr
        if not check_rate_limit(user_id):
            return "⚠️ Too many requests. Please wait a moment before trying again.", 429
        
        logger.info(f"\n{'='*60}")
        logger.info(f"💤 User: {msg}")
        
        # Emergency detection
        emergency_check = detect_emergency(msg)
        if emergency_check['is_emergency']:
            severity = emergency_check['severity']
            logger.warning(f"🚨 Emergency detected: {severity} - {emergency_check['keyword']}")
            
            if severity == 'critical':
                emergency_response = (
                    "🚨 **MEDICAL EMERGENCY DETECTED** 🚨\n\n"
                    "⚠️ **CALL EMERGENCY SERVICES IMMEDIATELY:**\n"
                    "• USA: 911\n"
                    "• UK: 999\n"
                    "• India: 112 or 102\n"
                    "• EU: 112\n"
                    "• Australia: 000\n\n"
                    "**While waiting for help:**\n"
                    "• Stay calm and try to breathe slowly\n"
                    "• Don't move if injured (unless in immediate danger)\n"
                    "• Keep your phone nearby\n"
                    "• If possible, have someone stay with you\n"
                    "• Unlock your door if safe to do so\n\n"
                    "⚠️ This is a life-threatening situation. Please seek immediate medical attention. "
                    "I can provide general information after emergency services are contacted."
                )
                return emergency_response
            
            elif severity == 'mental_health':
                crisis_response = (
                    "🆘 **Crisis Support Resources** 🆘\n\n"
                    "**Please reach out for immediate help:**\n\n"
                    "📞 **Crisis Hotlines (24/7):**\n"
                    "• USA - National Suicide Prevention: 988\n"
                    "• USA - Crisis Text Line: Text HOME to 741741\n"
                    "• UK - Samaritans: 116 123\n"
                    "• International: findahelpline.com\n\n"
                    "**You are not alone.** Trained counselors are available right now to listen and help. "
                    "These services are:\n"
                    "✅ Free and confidential\n"
                    "✅ Available 24/7\n"
                    "✅ Staffed by caring professionals\n\n"
                    "Your life matters, and help is available. Please reach out to someone who can provide immediate support."
                )
                return crisis_response
            
            elif severity == 'urgent':
                urgent_response = (
                    "⚠️ **Urgent Medical Attention Recommended** ⚠️\n\n"
                    "Based on your symptoms, you should seek medical attention soon:\n\n"
                    "**Options:**\n"
                    "• Visit an urgent care center\n"
                    "• Contact your doctor immediately\n"
                    "• Go to an emergency room if symptoms worsen\n\n"
                    "If symptoms become severe, call emergency services.\n\n"
                    "I can provide general information, but professional medical evaluation is important for your situation."
                )
                # Still generate response but prepend urgent message
                if 'chat_history' not in session:
                    session['chat_history'] = []
                chat_history = session['chat_history']
                
                answer = generate_response(msg, chat_history)
                
                chat_history.extend([
                    {'role': 'user', 'content': msg},
                    {'role': 'assistant', 'content': urgent_response + "\n\n" + answer}
                ])
                
                if len(chat_history) > 20:
                    chat_history = chat_history[-20:]
                
                session['chat_history'] = chat_history
                session.modified = True
                
                return urgent_response + "\n\n" + answer
        
        # Normal processing
        if 'chat_history' not in session:
            session['chat_history'] = []
        chat_history = session['chat_history']
        
        logger.info(f"💬 History: {len(chat_history)} messages")
        
        # Generate response
        answer = generate_response(msg, chat_history)
        
        logger.info(f"🤖 Bot: {answer[:100]}...")
        logger.info(f"{'='*60}\n")
        
        # Store conversation
        chat_history.extend([
            {'role': 'user', 'content': msg},
            {'role': 'assistant', 'content': answer}
        ])
        
        # Keep last 20 messages
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]
        
        session['chat_history'] = chat_history
        session.modified = True
        
        return str(answer)
    except Exception as e:
        logger.error(f"❌ Chat error: {e}")
        import traceback
        traceback.print_exc()
        return "I apologize, but I encountered an error processing your request. Please try again.", 500


@app.route("/clear_history", methods=["POST"])
def clear_history():
    session['chat_history'] = []
    session.modified = True
    logger.info("✅ Chat history cleared")
    return jsonify({'status': 'success', 'message': 'Chat history cleared'})


@app.route("/get_history", methods=["GET"])
def get_history():
    """Get current conversation history (for debugging)"""
    try:
        if 'chat_history' not in session:
            session['chat_history'] = []
        history = session['chat_history']
        return jsonify({
            'status': 'success',
            'history': history,
            'count': len(history)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route("/feedback", methods=["POST"])
def submit_feedback():
    """Collect user feedback on responses"""
    try:
        data = request.json
        feedback_type = data.get('type')  # 'helpful', 'not_helpful'
        message_id = data.get('message_id')
        comment = data.get('comment', '')
        
        # Log feedback (you could save to database later)
        logger.info(f"📝 Feedback received: {feedback_type} for message {message_id}")
        if comment:
            logger.info(f"   Comment: {comment}")
        
        return jsonify({'status': 'success', 'message': 'Thank you for your feedback!'})
    except Exception as e:
        logger.error(f"❌ Feedback error: {e}")
        return jsonify({'status': 'error', 'message': 'Failed to submit feedback'}), 500


@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"✅ File uploaded: {filename}")
        
        return jsonify({
            'status': 'success',
            'filename': filename,
            'message': 'File uploaded successfully'
        })
    except Exception as e:
        logger.error(f"❌ Upload error: {e}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large (max 5MB)'}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(429)
def rate_limit_exceeded(e):
    return jsonify({'error': 'Too many requests. Please slow down.'}), 429


@app.errorhandler(500)
def internal_error(e):
    logger.error(f"❌ Internal error: {e}")
    return jsonify({'error': 'Internal server error'}), 500


# ==================== TELEGRAM BOT ====================

def get_telegram_history(user_id):
    if user_id not in telegram_conversations:
        telegram_conversations[user_id] = []
    return telegram_conversations[user_id]


def clear_telegram_history(user_id):
    telegram_conversations[user_id] = []
    logger.info(f"Cleared history for user {user_id}")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id in telegram_conversations and len(telegram_conversations[user_id]) > 0:
        clear_telegram_history(user_id)
        history_cleared = True
    else:
        history_cleared = False
    
    welcome_message = """🩺 **Welcome to Medical Chatbot!**

I'm an AI-powered medical assistant designed to help with health-related questions.

📋 **What I can help with:**
- Symptoms and medical conditions
- General health information
- Medication queries
- Medical terminology
- Health tips and advice

⚠️ **Important:** I'm for educational purposes only and cannot replace professional medical advice.

🚨 **For Emergencies:** Call your local emergency number immediately!

💬 **How to use:**
Just type your medical question and I'll help you!

**Commands:**
/help - Get help information
/about - Learn about this bot
/clear - Clear your conversation history
/history - View your conversation stats

Ask me anything about health and medicine! 🥼

**Note:** I remember our entire conversation! You can:
✅ Ask follow-up questions
✅ Request clarification on previous topics
✅ Ask "What did I ask before?"
✅ Say "Tell me more about that"
"""
    
    if history_cleared:
        welcome_message += "\n\n🗑️ _Your previous conversation history has been cleared._"
    
    await update.message.reply_text(welcome_message, parse_mode='Markdown')


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = """🆘 **How to use Medical Chatbot:**

**Sample questions you can ask:**
- "What are the symptoms of diabetes?"
- "How does aspirin work?"
- "What is high blood pressure?"
- "Treatment for knee pain"
- "What causes headaches?"

**Follow-up questions:**
- "Tell me more about that"
- "What are the treatment options?"
- "Can you explain the previous answer in simpler terms?"
- "What did I ask about earlier?"
- "Remind me what you said about [topic]"

**Memory Features:**
- I remember our entire conversation
- Ask about previous questions anytime
- Reference earlier topics naturally
- Get contextual answers to follow-ups

**Commands:**
/help - This help message
/about - About this bot
/clear - Clear your conversation history
/history - View your conversation stats

**Tips:**
✅ Be specific with your questions
✅ Ask about symptoms, conditions, or treatments
✅ Use clear, simple language
✅ Ask follow-up questions - I remember everything!
✅ Reference previous topics naturally

❌ Don't ask for personal medical diagnosis
❌ Don't use for emergency situations

**Emergency:** If you have a medical emergency, contact your local emergency services immediately! 🚨"""
    
    await update.message.reply_text(help_text, parse_mode='Markdown')


async def about(update: Update, context: ContextTypes.DEFAULT_TYPE):
    about_text = """🤖 **About Medical Chatbot**

This is an AI-powered medical information bot created for educational purposes.

**Technology:**
- Powered by Google Gemini 2.5 Flash (Native API)
- Uses RAG (Retrieval Augmented Generation)
- Google Search integration for current information
- Medical knowledge from Pinecone vector database
- Advanced conversation memory system
- Emergency detection system
- Built with Python and Telegram Bot API

**Features:**
- Full conversation memory (30 messages)
- Context-aware responses
- Follow-up question handling
- Previous conversation recall
- Emergency keyword detection
- Crisis resource provision
- Comprehensive, detailed answers
- Google Search for latest information

**Purpose:**
- Student project demonstration
- Educational tool for medical information
- Accessible health information platform

**Limitations:**
- For educational use only
- Cannot provide personal medical diagnosis
- Not a replacement for healthcare professionals
- Always consult doctors for medical concerns

**Safety Features:**
- Emergency detection
- Crisis hotline information
- Medical disclaimers
- Rate limiting protection

**Developer:** Monish (Team Lead) & Amruth (Team Member)
**Version:** 4.0 (Enhanced with Emergency Detection)
**Status:** Educational/Non-commercial use

For medical emergencies, always contact professional healthcare services! 🥼"""
    
    await update.message.reply_text(about_text, parse_mode='Markdown')


async def clear_history_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    clear_telegram_history(user_id)
    
    await update.message.reply_text(
        "🗑️ Your conversation history has been cleared!\n\n"
        "You can start a fresh conversation now. All previous context has been forgotten.",
        parse_mode='Markdown'
    )


async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    history = get_telegram_history(user_id)
    
    num_exchanges = len(history) // 2
    
    if num_exchanges == 0:
        await update.message.reply_text(
            "📊 **Conversation Statistics**\n\n"
            "No conversation history yet. Start by asking a medical question!",
            parse_mode='Markdown'
        )
        return
    
    # Get recent user questions (last 3)
    user_questions = [msg['content'] for msg in history if msg['role'] == 'user']
    recent_questions = user_questions[-3:] if len(user_questions) >= 3 else user_questions
    
    recent_q_text = "\n".join([f"• {q[:80]}..." if len(q) > 80 else f"• {q}" for q in recent_questions])
    
    history_text = f"""📊 **Conversation Statistics**

**Total messages:** {len(history)}
**Exchanges:** {num_exchanges}
**Memory limit:** 15 exchanges (30 messages)

{"**Status:** Within memory limit ✅" if len(history) <= 30 else "**Status:** At memory limit (oldest messages removed) ⚠️"}

**Recent Questions:**
{recent_q_text}

💡 **Tip:** You can ask me "What did I ask before?" or reference any previous topic in our conversation!

Use /clear to start a fresh conversation."""
    
    await update.message.reply_text(history_text, parse_mode='Markdown')


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    user_id = update.effective_user.id
    user_name = update.effective_user.first_name or "User"
    
    logger.info(f"Received message from {user_name} (ID: {user_id}): {user_message}")
    
    processing_msg = None
    try:
        # Show typing indicator
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        
        # Send a "processing" message for user feedback
        processing_msg = await update.message.reply_text(
            "🔍 Processing your question...\n_This may take a moment._",
            parse_mode='Markdown'
        )
        
        # Check for emergencies
        emergency_check = detect_emergency(user_message)
        if emergency_check['is_emergency']:
            severity = emergency_check['severity']
            
            try:
                await processing_msg.delete()
            except:
                pass
            
            if severity == 'critical':
                await update.message.reply_text(
                    "🚨 **MEDICAL EMERGENCY DETECTED** 🚨\n\n"
                    "⚠️ **CALL EMERGENCY SERVICES IMMEDIATELY:**\n"
                    "• USA: 911\n"
                    "• UK: 999\n"
                    "• India: 112 or 102\n"
                    "• EU: 112\n"
                    "• Australia: 000\n\n"
                    "**While waiting for help:**\n"
                    "• Stay calm and try to breathe slowly\n"
                    "• Don't move if injured (unless in immediate danger)\n"
                    "• Keep your phone nearby\n"
                    "• If possible, have someone stay with you\n"
                    "• Unlock your door if safe to do so\n\n"
                    "⚠️ This is a life-threatening situation. Please seek immediate medical attention. "
                    "I can provide general information after emergency services are contacted.",
                    parse_mode='Markdown'
                )
                return
            
            elif severity == 'mental_health':
                await update.message.reply_text(
                    "🆘 **Crisis Support Resources** 🆘\n\n"
                    "📞 **24/7 Crisis Hotlines:**\n"
                    "• India: 112 or 102 or 100\n"
                    "• USA: 988 (Suicide Prevention)\n"
                    "• UK: 116 123 (Samaritans)\n"
                    "• Text HOME to 741741 (Crisis Text Line)\n\n"
                    "You are not alone. Please reach out for help.",
                    parse_mode='Markdown'
                )
                return
        
        # Get history and response
        history = get_telegram_history(user_id)
        response = generate_response(user_message, history)
        
        # Store conversation
        history.extend([
            {'role': 'user', 'content': user_message},
            {'role': 'assistant', 'content': response}
        ])
        
        # Keep only last 30 messages
        if len(history) > 30:
            telegram_conversations[user_id] = history[-30:]
        
        # Delete processing message
        try:
            await processing_msg.delete()
        except:
            pass
        
        # Telegram has a 4096 character limit per message
        max_length = 4000
        
        # Send response (split if too long)
        if len(response) > max_length:
            chunks = [response[i:i+max_length] for i in range(0, len(response), max_length)]
            for i, chunk in enumerate(chunks):
                try:
                    if i == 0:
                        await update.message.reply_text(
                            f"📋 **Medical Information (Part {i+1}/{len(chunks)}):**\n\n{chunk}",
                            parse_mode='Markdown'
                        )
                    else:
                        await context.bot.send_message(
                            chat_id=update.effective_chat.id,
                            text=f"📋 **Continued (Part {i+1}/{len(chunks)}):**\n\n{chunk}",
                            parse_mode='Markdown'
                        )
                    time.sleep(0.5)
                except Exception as send_error:
                    logger.warning(f"Markdown error, sending as plain text: {send_error}")
                    if i == 0:
                        await update.message.reply_text(f"Medical Information (Part {i+1}/{len(chunks)}):\n\n{chunk}")
                    else:
                        await context.bot.send_message(
                            chat_id=update.effective_chat.id,
                            text=f"Continued (Part {i+1}/{len(chunks)}):\n\n{chunk}"
                        )
                        time.sleep(0.5)
        else:
            try:
                await update.message.reply_text(
                    f"🩺 **Medical Information:**\n\n{response}",
                    parse_mode='Markdown'
                )
            except Exception as send_error:
                logger.warning(f"Markdown error, sending as plain text: {send_error}")
                await update.message.reply_text(f"🩺 Medical Information:\n\n{response}")
    
    except Exception as e:
        logger.error(f"Error in handle_message: {str(e)}")
        import traceback
        traceback.print_exc()
        
        if processing_msg:
            try:
                await processing_msg.delete()
            except:
                pass
        
        try:
            await update.message.reply_text(
                "🩺 I apologize, but I'm having trouble processing your message right now.\n\n"
                "**You can try:**\n"
                "• Rephrasing your question\n"
                "• Being more specific\n"
                "• Using /clear to start a fresh conversation\n\n"
                "I'm here to help with your medical questions! 😊"
            )
        except:
            # If even error message fails, try without markdown
            await update.message.reply_text(
                "I apologize, but I'm having trouble right now. Please try again or use /clear to reset."
            )


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.warning(f'Update {update} caused error {context.error}')
    
    if update and update.message:
        await update.message.reply_text(
            "🚫 Sorry, I encountered an error while processing your request. Please try again or contact support."
        )


# ==================== TELEGRAM BOT THREAD ====================

def run_telegram_bot():
    """Run Telegram bot in separate thread with proper event loop"""
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("⚠️ TELEGRAM_BOT_TOKEN not found. Telegram bot disabled.")
        return
    
    try:
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Create application
        application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
        
        # Register handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(CommandHandler("about", about))
        application.add_handler(CommandHandler("clear", clear_history_command))
        application.add_handler(CommandHandler("history", history_command))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        application.add_error_handler(error_handler)
        
        logger.info("🚀 Telegram bot started!")
        logger.info("🤖 Model: Google Gemini 2.5 Flash (Native API)")
        logger.info("💬 Conversation memory: ENABLED (30 messages)")
        logger.info("🧠 Context-aware responses: ACTIVE")
        logger.info("🚨 Emergency detection: ACTIVE")
        logger.info("🔍 Google Search: ENABLED")
        logger.info("🔄 Auto-retry: 2 attempts with smart error handling")
        logger.info("🛡️ Rate limiting: ACTIVE")
        logger.info("Bot is ready to receive messages!")
        
        # Run bot in the event loop
        loop.run_until_complete(application.run_polling(allowed_updates=Update.ALL_TYPES))
    except Exception as e:
        logger.error(f"❌ Telegram bot error: {e}")
        import traceback
        traceback.print_exc()


# ==================== MAIN ====================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🥼 Medical Chatbot - Enhanced Server v4.0")
    print("="*70)
    print("🌐 Flask Web: http://localhost:8080")
    print("🎤 Voice input: Enabled")
    print("🔊 Text-to-speech: Enabled (browser-based)")
    print("🤖 Telegram: " + ("✅ Enabled" if TELEGRAM_BOT_TOKEN else "❌ Disabled"))
    print("🧠 Model: Google Gemini 2.5 Flash")
    print("🔍 Google Search: Enabled")
    print("💬 Memory: Active (20 messages)")
    print("📎 File upload: Ready")
    print("🔄 Auto-retry: 2 attempts")
    print("🚨 Emergency Detection: ACTIVE")
    print("🛡️ Rate Limiting: 10 requests/minute")
    print("📝 Feedback System: Ready")
    print("="*70 + "\n")
    
    # Start Telegram bot in separate thread if token exists
    if TELEGRAM_BOT_TOKEN:
        telegram_thread = threading.Thread(target=run_telegram_bot, daemon=True)
        telegram_thread.start()
        logger.info("✅ Telegram bot thread started")
    
    # Run Flask app
    app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False)
