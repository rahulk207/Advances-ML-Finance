import ccxt
import pandas as pd
import io
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseUpload
import pickle

from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("API_KEY")
api_secret = os.getenv("API_SECRET")
# Initialize binance object
binance = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret
})


def write_to_drive(trades_df, timestamp, service):
    file_name = "{0}.pickle".format(str(timestamp))
    pickle_obj = pickle.dumps(trades_df)
    parent_folder_id = "1Vnj8fuXnxBPP0hEW5QN1eW_hlaZthxhy"
    try:
        # Create a MediaIoBaseUpload object for the local file
        file_metadata = {'name': file_name, 'parents': [
            parent_folder_id]} if parent_folder_id else {'name': file_name}
        media = MediaIoBaseUpload(io.BytesIO(
            pickle_obj), mimetype='application/octet-stream', resumable=True)

        # Upload the file to Google Drive
        file = service.files().create(body=file_metadata,
                                      media_body=media, fields='id').execute()
        print(f'File ID: {file["id"]}')
    except HttpError as error:
        print(f'An error occurred: {error}')

    print("Done till", timestamp)


def fetch_trades(symbol, since, end, drive_service):
    start = since
    while start < end:
        trades = binance.fetch_trades(
            symbol=symbol, since=start)
        if len(trades) == 0:
            # If stuck somewhere (means data not available), we add 1 hour and try again
            print("Stuck at", start, "Trying", start + int(3.6e+6))
            start += int(3.6e+6)
            continue
        trades_df = pd.DataFrame(trades)
        start = int(trades_df.iloc[-1]['timestamp']) + 1

        write_to_drive(trades_df, start - 1, drive_service)

    return


if __name__ == '__main__':
    symbol = 'ETH/USDT'  # Choosing USDT over USDC to get more historic data
    # since = 1504223940000  # August 31, 2017 11:59:00 GMT
    # The code stopped at 1504713588194 (6th Sep 2017). Started again with the below ts.
    # since = 1505087940000  # Sep 10, 2017 11:59:00 GMT
    # The code stopped at 1513600152596 (18th Dec 2017). Started again with the below ts.
    # since = 1513814340000  # Dec 20, 2017 11:59:00 GMT
    since = 1625135106073  # Restarting because of internet issue
    end = 1678492740000  # March 10, 2023 11:59:00 GMT
    # fetch_trades(symbol, since, end)

    # Code to prepare google-drive-oauth
    SCOPES = ['https://www.googleapis.com/auth/drive']
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

    # Create a Drive API client
    service = build('drive', 'v3', credentials=creds)
    fetch_trades(symbol, since, end, service)
