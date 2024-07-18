import os
from binanceHistoricData import *
import pandas as pd
import math
from tqdm import tqdm
import pickle
import time
from googleapiclient.errors import HttpError


def preprocess_df(df):
    return df.drop(columns=["info", "id", "order", "type", "side", "takerOrMaker", "fee", "fees"], axis=1)


def loadDriveService():
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
    return build('drive', 'v3', credentials=creds)


def getFileNamesFromDrive(start_timestamp, end_timestamp, drive_service, folder_id, file_names_path):
    # Get ID of starting file
    start_file_name = "{}.pickle".format(str(start_timestamp))
    end_file_name = "{}.pickle".format(str(end_timestamp))

    # Get creation time of start file
    query = "name = '{}' and trashed=false and parents in '{}'".format(
        start_file_name, folder_id)
    response = drive_service.files().list(
        q=query, fields='files(createdTime)').execute()
    start_file_created_time = response['files'][0]['createdTime']

    # Get creation time of end file
    query = "name = '{}' and trashed=false and parents in '{}'".format(
        end_file_name, folder_id)
    response = drive_service.files().list(
        q=query, fields='files(createdTime)').execute()
    end_file_created_time = response['files'][0]['createdTime']

    # Getting all files between the start and end file
    query = "'{}' in parents and trashed=false and createdTime >= '{}' and createdTime <= '{}'".format(
        folder_id, start_file_created_time, end_file_created_time)
    response = drive_service.files().list(
        q=query, pageSize=1000, fields='nextPageToken, files(name)').execute()
    files = response.get('files', [])
    next_page_token = response.get('nextPageToken', None)

    num_page = 1
    while next_page_token:
        response = drive_service.files().list(q=query, pageSize=1000, pageToken=next_page_token,
                                              fields='nextPageToken, files(name)').execute()
        files.extend(response.get('files', []))
        next_page_token = response.get('nextPageToken', None)
        num_page += 1

        print(num_page)
    print(len(files))

    # Reverse the files list to convert in increasing order of creation time
    files.reverse()

    # Saving names for future use
    f = open(file_names_path, "wb")
    pickle.dump(files, f)


def constructDollarBar(trades, dollar_threshold):
    timestamp_start = trades[0][1]
    open_price = trades[0][-3]
    high = 0
    low = math.inf
    running_dollar_sum, total_volume = 0, 0
    for i in range(len(trades)):
        high = max(high, trades[i][-3])
        low = min(low, trades[i][-3])
        running_dollar_sum += trades[i][-1]
        total_volume += trades[i][-2]
        if running_dollar_sum >= dollar_threshold:
            close = trades[i][-3]
            timestamp_end = trades[i][1]
            symbol = trades[i][3]
            # Give ohlc, amount and cost values
            dollar_bar = [timestamp_start, timestamp_end, symbol,
                          open_price, high, low, close, total_volume, running_dollar_sum]
            trades = trades[(i+1):]
            break

    return trades, dollar_bar


def createDollarBarsFromDrive(start_timestamp, end_timestamp, dollar_threshold, checkpoint_file_name):
    drive_service = loadDriveService()
    folder_id = "1Vnj8fuXnxBPP0hEW5QN1eW_hlaZthxhy"

    file_names_path = "drive_data/files_list/{}-{}.pickle".format(
        start_timestamp, end_timestamp)
    if not os.path.exists(file_names_path):
        getFileNamesFromDrive(start_timestamp, end_timestamp,
                              drive_service, folder_id, file_names_path)

    f = open(file_names_path, "rb")
    files = pickle.load(f)
    print("Loaded files")

    if os.path.exists(checkpoint_file_name):
        f = open(checkpoint_file_name, "rb")
        checkpoint_dict = pickle.load(f)
    else:
        checkpoint_dict = {"num_files": 0, "dollar_bars": [], "trades": []}

    # Create dollar-bars based on threshold
    dollar_bars = checkpoint_dict["dollar_bars"]
    trades = checkpoint_dict["trades"]
    num_files = checkpoint_dict["num_files"]
    for i in tqdm(range(num_files, len(files))):
        file = files[i]

        # Error handling to cater network error
        while True:
            try:
                query = "name = '{}' and trashed=false and parents in '{}'".format(
                    file['name'], folder_id)
                response = drive_service.files().list(
                    q=query).execute()
                df_file = response.get("files", [])
            except HttpError:
                print("Network error occurred")
                time.sleep(3)
            else:
                break

        if not df_file:
            print("File not found")
            break

        # Download the file contents using the retrieved file ID
        df_file_id = df_file[0]["id"]
        df_file_content = drive_service.files().get_media(fileId=df_file_id).execute()

        df = pickle.loads(df_file_content)
        trades.extend(preprocess_df(df).values.tolist())

        total_dollars = 0
        for j in range(len(trades)):
            total_dollars += trades[j][-1]

        if total_dollars >= dollar_threshold:
            trades, dollar_bar = constructDollarBar(trades, dollar_threshold)
            dollar_bars.append(dollar_bar)

        num_files += 1
        if num_files % 500 == 0:
            f = open(checkpoint_file_name, "wb")
            checkpoint_dict = {"num_files": num_files,
                               "dollar_bars": dollar_bars, "trades": trades}
            pickle.dump(checkpoint_dict, f)

    while True:
        total_dollars = 0
        for j in range(len(trades)):
            total_dollars += trades[j][-1]

        if total_dollars < dollar_threshold:
            break
        elif total_dollars >= dollar_threshold:
            trades, dollar_bar = constructDollarBar(
                trades, dollar_threshold)
            dollar_bars.append(dollar_bar)

    return pd.DataFrame(dollar_bars, columns=["timestamp_start", "timestamp_end", "symbol", "open", "high", "low", "close", "volume", "total_dollars"])


def createDollarBarsScalable(trades_dir, dollar_threshold):
    dollar_bars = []
    file_names = sorted(os.listdir(trades_dir))
    trades = []
    for name in tqdm(file_names):
        data_path = trades_dir + "/" + name

        f = open(data_path, "rb")
        df = pickle.load(f)
        # df = pd.read_csv(data_path)

        trades.extend(preprocess_df(df).values.tolist())
        total_dollars = 0
        for j in range(len(trades)):
            total_dollars += trades[j][-1]

        if total_dollars >= dollar_threshold:
            trades, dollar_bar = constructDollarBar(trades, dollar_threshold)
            dollar_bars.append(dollar_bar)

    while True:
        total_dollars = 0
        for j in range(len(trades)):
            total_dollars += trades[j][-1]

        if total_dollars < dollar_threshold:
            break
        elif total_dollars >= dollar_threshold:
            trades, dollar_bar = constructDollarBar(trades, dollar_threshold)
            dollar_bars.append(dollar_bar)

    return pd.DataFrame(dollar_bars, columns=["timestamp_start", "timestamp_end", "symbol", "open", "high", "low", "close", "volume", "total_dollars"])


def createDollarBars(df, dollar_threshold):
    df.drop(['info'], axis=1, inplace=True)

    # Convert timestamp to a datetime and set as index
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    # Calculate the total value traded (in USDT) for each trade
    df['value'] = df['cost']
    # Define a target dollar amount for each bar
    bar_size = dollar_threshold

    # Calculate the running total value traded
    df['cum_value'] = df['value'].cumsum()
    print(df)
    print((df['cum_value'] // bar_size).nunique())

    # Calculate the number of bars
    num_bars = (df['cum_value'] // bar_size).nunique()
    print(num_bars)

    df['bins'] = pd.cut(df['cum_value'], num_bars)
    print(df)
    # GroupBY the DataFrame based on bins to create dollar bars
    dollar_bars = df.groupby(df['bins']).agg({
        'side': 'last',
        'price': 'ohlc',
        'amount': 'sum',
        'value': 'sum'
    })

    # Drop any rows with missing data
    dollar_bars.dropna(inplace=True)
    print(dollar_bars)

    # Flatten the column MultiIndex
    dollar_bars.columns = ["_".join(x) for x in dollar_bars.columns.ravel()]
    # Rename columns for clarity
    dollar_bars.rename(columns={
        'price_open': 'open',
        'price_high': 'high',
        'price_low': 'low',
        'price_close': 'close',
        'amount_amount': 'volume',
        'value_value': 'dollar_value'
    }, inplace=True)
    print(dollar_bars)


if __name__ == '__main__':
    # dollar_threshold = 100000  # In USD
    # dollar_bars = createDollarBarsScalable(
    #     "data_large/raw_data", dollar_threshold)
    # dollar_bars.to_csv("data_large/dollar_bars.csv")

    # start_timestamp = 1547227532019
    # end_timestamp = 1550474521640
    # dollar_threshold = 100000

    start_timestamp = 1615900014959
    end_timestamp = 1618399971101
    dollar_threshold = 1000000

    checkpoint_file_name = "drive_data/checkpoints/{}-{}-{}.checkpoint".format(
        start_timestamp, end_timestamp, dollar_threshold)
    dollar_bars = createDollarBarsFromDrive(
        start_timestamp, end_timestamp, dollar_threshold, checkpoint_file_name)

    dollar_bars.to_csv(
        "drive_data/dollar_bars/{}-{}-{}.csv".format(start_timestamp, end_timestamp, dollar_threshold))
