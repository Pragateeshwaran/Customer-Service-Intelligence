from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import re
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from dotenv import load_dotenv
import os

load_dotenv()

def generate_response(user_input, system_input):
    llm = ChatGroq(model_name="llama3-70b-8192", api_key=os.getenv("GROQ_API_KEY"))
    prompt_template = PromptTemplate(
        input_variables=["system_input", "user_input"],
        template="### System:\n{system_input}\n### User:\n{user_input}\n### Assistant:\n"
    )
    chain = prompt_template | llm
    response = chain.invoke({"system_input": system_input, "user_input": user_input})
    return response.content

def prompt_remover(string):
    return re.findall(r'\*{3}(.*?)\*{3}', string)

def extract_content(string):
    match = re.match(r'(\w+)\(\"(.*?)\"\)', string)
    return match.groups() if match else None

def sendmail(recipient, subject, content):
    try:
        message = Mail(
            from_email='no-reply@yourdomain.com',
            to_emails=recipient,
            subject=subject,
            plain_text_content=content
        )
        sg = SendGridAPIClient(os.getenv('SENDGRID_API_KEY'))
        sg.send(message)
    except Exception as ex:
        print(f"Error sending email: {ex}")

def assess(user_input, model, tokenizer):
    template = '***official("Johnny showed unprofessional behavior at the beginning of the call by using inappropriate language and taking a long time, later he gave a contact information.")***'
    system_input = f"You are the customer service supervisor... {template}"
    response = generate_response(user_input, system_input)
    queries = prompt_remover(response)
    for i in queries:
        task, content = extract_content(i)
        if task == 'official':
            sendmail('pragateeshgamesid@gmail.com', 'Issue regarding customer service', content)
