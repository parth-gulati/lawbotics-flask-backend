from flask import Flask, request, jsonify
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import base64
from haystack.telemetry import tutorial_running
from haystack.nodes import FileTypeClassifier, PreProcessor, PromptNode, OpenAIAnswerGenerator, BM25Retriever, FARMReader, DocxToTextConverter, PDFToTextConverter
from haystack.document_stores import SQLDocumentStore, ElasticsearchDocumentStore, BaseDocumentStore
from haystack.pipelines import ExtractiveQAPipeline, DocumentSearchPipeline, GenerativeQAPipeline
from haystack import Pipeline
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever
from haystack import Document
from bs4 import BeautifulSoup
import os
import json
from operator import itemgetter
import datetime

app = Flask(__name__)

##GLOBAL STATE AND ENV VARIABLES TO BE USED, WILL GO INTO .ENV OR GLOBAL STATE

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

app_state = {
    "user": None
}

document_store = InMemoryDocumentStore(use_bm25=True)

dicts = []

service = None

user_id = "me"

directory_path = "files-attachments"

##HELPER FUNCTIONS

def get_doc_files(directory_path):
    doc_files = []

    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        if os.path.isfile(file_path) and file_name.lower().endswith((".doc", ".docx")):
            doc_files.append(file_name)

    return doc_files


def get_attachments(service, user_id, msg_id):
    try:
        message = service.users().messages().get(userId=user_id, id=msg_id).execute()
        parts = message['payload']['parts']
        attachments = []
        for part in parts:
            if part['filename']:
                attachment = service.users().messages().attachments().get(userId=user_id, messageId=msg_id,
                                                                          id=part['body'][
                                                                              'attachmentId']).execute()
                file_data = base64.urlsafe_b64decode(attachment['data'].encode('UTF-8'))
                attachments.append({'filename': part['filename'], 'data': file_data})
        return attachments
    except Exception as e:
        print('An error occurred:', e)
        return []


def search_emails(service, user_id, query):
    try:
        response = service.users().messages().list(userId=user_id, q=query).execute()
        messages = []
        if 'messages' in response:
            messages.extend(response['messages'])
        while 'nextPageToken' in response:
            page_token = response['nextPageToken']
            response = service.users().messages().list(userId=user_id, q=query, pageToken=page_token).execute()
            if 'messages' in response:
                messages.extend(response['messages'])
        return messages
    except Exception as e:
        print('An error occurred:', e)
        return []


def download_attachments(emails):
    # Create the 'files-attachments' folder if it doesn't exist
    if not os.path.exists('files-attachments'):
        os.makedirs('files-attachments')

    # Download attachments from relevant emails
    for email in emails:
        msg_id = email['id']
        attachments = get_attachments(service, user_id, msg_id)
        for attachment in attachments:
            filename, file_extension = os.path.splitext(attachment['filename'])
            if file_extension.lower() in ['.pdf', '.docx', '.doc']:
                file_path = os.path.join('files-attachments', attachment['filename'])
                with open(file_path, 'wb') as f:
                    f.write(attachment['data'])
                print(f"Downloaded attachment: {attachment['filename']}")

##MAIN APIS
@app.route('/login')
def login():
    """Shows basic usage of the Gmail API.
        Lists the user's Gmail labels.
        """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        # Call the Gmail API
        global service
        service = build('gmail', 'v1', credentials=creds)
        results = service.users().labels().list(userId='me').execute() ##getting labels of email, sample API to check if logged in User is able to access email

        return "Logged in successfully"

    except HttpError as error:
        # TODO(developer) - Handle errors from gmail API.
        print(f'An error occurred: {error}')
        return "Unable to login"
    return "Logged in successfully"


@app.route('/retrieve-documents', methods=['POST'])
def fetch_data():
    document_store = InMemoryDocumentStore(use_bm25=True)
    dicts = []
    global service
    if service is None:
        return "Log in first, no results fetched"
    raw_data = request.get_data()

    # Decode the bytes data assuming it's in utf-8 encoding
    decoded_data = raw_data.decode('utf-8')

    # Parse the decoded data as JSON
    json_data = json.loads(decoded_data)

    # Now you can access the attributes using dictionary-style access
    start_yy = json_data['start_yy']
    start_mm = json_data['start_mm']
    start_dd = json_data['start_dd']
    end_yy = json_data['end_yy']  # Note the correct assignment operator
    end_mm = json_data['end_mm']
    end_dd = json_data['end_dd']
    search_text = json_data['query']


    ##Define the start and end dates for the date range
    start_date = datetime.datetime(start_yy, start_mm, start_dd)  # Replace with your desired start date
    end_date = datetime.datetime(end_yy, end_mm, end_dd)  # Replace with your desired end date

    start_date_disp = start_date.strftime("%m/%d/%Y")
    end_date_disp = end_date.strftime("%m/%d/%Y")

    # Create the 'files-attachments' folder if it doesn't exist
    if not os.path.exists('files-attachments'):
        os.makedirs('files-attachments')

    # Construct the query
    query = f'"${search_text}" after:{start_date_disp} before:{end_date_disp}'
    result = service.users().messages().list(userId='me', q=query).execute()
    messages = result.get('messages')

    # iterate through all the messages
    for msg in messages:
        txt = service.users().messages().get(userId='me', id=msg['id']).execute()
        try:
            payload = txt['payload']
            headers = payload['headers']
            # Look for Subject and Sender Email in the headers
            for d in headers:
                if d['name'] == 'Subject':
                    subject = d['value']
                if d['name'] == 'From':
                    sender = d['value']
            parts = payload.get('parts')[0]
            data = parts['body']['data']
            data = data.replace("-", "+").replace("_", "/")
            decoded_data = base64.b64decode(data)
            soup = BeautifulSoup(decoded_data, "lxml")
            body = soup.body.get_text()
            print(type(body))
            document = {
                "content": body,  # Email body
                "meta": {
                    'sender': sender,  # Sender email address
                    'subject': subject  # Email subject
                }
            }
            dicts.append(document)
        except Exception as e:
            print(e)
        document_store.write_documents(dicts)
        creds = Credentials.from_authorized_user_file('token.json')

        # Set up the Gmail API service
        service = build('gmail', 'v1', credentials=creds)

        # Replace 'your_email@gmail.com' with your Gmail address
        user_id = 'me'

        # Replace 'Your search query' with the relevant search query
        search_query = search_text

        # Search for emails with the given query
        relevant_emails = search_emails(service, user_id, search_query)

        # Download attachments from relevant emails and save them to the 'files-attachments' folder
        download_attachments(relevant_emails)

        # stores file text in elasticsearch
        store_file_text_pipeline = Pipeline()
        store_file_text_pipeline.add_node(FileTypeClassifier(), name="FTClassifier", inputs=["File"])

        # supported file types
        store_file_text_pipeline.add_node(DocxToTextConverter(), name="DocxConverter", inputs=["FTClassifier.output_4"])

        # Text preprocessor
        store_file_text_pipeline.add_node(PreProcessor(), name="Preprocessor", inputs=[
            "DocxConverter",
        ])
        store_file_text_pipeline.add_node(document_store, name="DocStore", inputs=["Preprocessor"])

        directory_path = "files-attachments"
        doc_files = get_doc_files(directory_path)

        for file_name in doc_files:
            res = store_file_text_pipeline.run(
                file_paths=[f"{directory_path}/{file_name}"],
                meta={
                    "filename": file_name
                },
            )
            # Add filename to document id
            for document in res['documents']:
                document.id = f"{document.id}_{file_name}"

            documents = res["documents"]
            document_store.write_documents(documents)

    return "Successfully retrieved all the documents"

@app.route('/run-query', methods=['POST'])
def run_query():
    raw_data = request.get_data()

    # Decode the bytes data assuming it's in utf-8 encoding
    decoded_data = raw_data.decode('utf-8')

    # Parse the decoded data as JSON
    json_data = json.loads(decoded_data)

    question = json_data['question']

    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
    retriever = BM25Retriever(document_store=document_store)

    generator = OpenAIAnswerGenerator(
        api_key="**insert API key",
        model="text-davinci-003",
        max_tokens=1000,
        temperature=0.75,
    )

    gen_qa_premade = GenerativeQAPipeline(generator, retriever)

    prediction = gen_qa_premade.run(
        query=question,
        params={
            "Retriever": {"top_k": 5},
            "Generator": {
                "top_k": 5,
            },
        }
    )

    return prediction

if __name__ == '__main__':
    app.run()
