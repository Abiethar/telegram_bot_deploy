from typing import Final
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import pickle
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import uuid

TOKEN: Final = "8047499963:AAEi229MlfV1xhqgAbPFOGhXTM_Rap3Ysi4"
BOT_USERNAME: Final = "@testingtriplebot"

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello!")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Please Type Something")
    
async def custom_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("This is a custom command!")
    
async def generate_answer(user_input: str) -> str:
    with open("rag_components.pkl", "rb") as f:
        loaded_data = pickle.load(f)

    retriever = loaded_data["retriever"]
    contextualize_q_prompt = loaded_data["contextualize_q_prompt"]
    qa_prompt = loaded_data["qa_prompt"]

    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key="AIzaSyCTWU4EWjxo3LjnPZZvC0dPMX098tkSOP0",
        temperature=0.2,
        max_tokens=None
    )

    history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_keys="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    session_id = str(uuid.uuid4())

    if user_input.lower() == "restart":
        store.clear()
    else:
        answer = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )["answer"]
        return answer

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    print(f"Received message: {text}", flush=True)  # Debugging output
    await update.message.reply_text("Hello! Your message was received.")

async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"Update {update} caused error {context.error}")

if __name__ == "__main__":
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("custom", custom_command))
    
    app.add_handler(MessageHandler(filters.TEXT, handle_message))
    
    app.add_error_handler(error)
    print("Telegram bot has worked")
    app.run_polling(poll_interval=3)
