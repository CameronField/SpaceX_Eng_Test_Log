import os
import io
import boto3
import logging
import subprocess
import shutil
from datetime import timedelta, datetime
import pandas as pd
import matplotlib.pyplot as plt
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# --- Configuration ---
SES_SENDER = "#INPUT_SES_SENDER"               # e.g., "your_ses_sender@example.com"
# Visible recipient appears in the email header
VISIBLE_RECIPIENT = "#INPUT_VISIBLE_RECIPIENT" # e.g., "visible_recipient@example.com"
# BCC recipient does not appear in headers
BCC_RECIPIENT = "#INPUT_BCC_RECIPIENT"         # e.g., "bcc_recipient@example.com"
SES_REGION = "#INPUT_SES_REGION"               # e.g., "us-east-1"
REPO_URL = "#INPUT_REPO_URL"                   # e.g., "https://github.com/YourUsername/YourRepo.git"
CSV_FILENAME = "#INPUT_CSV_FILENAME"           # e.g., "test_logs.csv"
CSV_HYPERLINK = "#INPUT_CSV_HYPERLINK"         # e.g., "https://github.com/YourUsername/YourRepo/blob/main/test_logs.csv"

# --- Repository Functions ---

def clone_repo(repo_url, clone_path="/tmp/SpaceX_Eng_Test_Log"):
    """
    Clone the GitHub repository into /tmp.
    """
    try:
        if os.path.exists(clone_path):
            shutil.rmtree(clone_path)
        logger.info(f"Cloning repo {repo_url} to {clone_path}")
        subprocess.run(["git", "clone", repo_url, clone_path], check=True)
        logger.info("Repository cloned successfully.")
    except Exception as e:
        logger.error(f"Error cloning repository: {e}")
        raise

def fetch_csv_from_repo(clone_path="/tmp/SpaceX_Eng_Test_Log", csv_filename=CSV_FILENAME):
    """
    Read the CSV file from the cloned repository.
    Let pandas auto-detect the delimiter.
    """
    csv_path = os.path.join(clone_path, csv_filename)
    logger.info(f"Reading CSV from {csv_path}")
    try:
        df = pd.read_csv(csv_path, sep=None, engine='python')
        logger.info(f"CSV data read successfully. Columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        raise

# --- Analysis Functions ---

def compute_analysis_week():
    """
    Since this function is scheduled to run every Wednesday morning at 7:30 AM,
    define the analysis week as the previous Wednesday through Tuesday.
    """
    analysis_time = pd.Timestamp.now()  # assumed to be Wednesday morning
    today = analysis_time.normalize()
    week_start = today - pd.Timedelta(days=7)  # last Wednesday
    week_end = today - pd.Timedelta(days=1)      # last Tuesday
    logger.info(f"Analysis week from {week_start.date()} to {week_end.date()}")
    return week_start, week_end

def filter_week_data(df, week_start, week_end):
    """
    Convert the 'date' column and filter for the analysis week.
    """
    df['date'] = pd.to_datetime(df['date'], format='%b %d %Y', errors='coerce')
    week_df = df[(df['date'] >= week_start) & (df['date'] <= week_end)]
    return week_df

def compute_overall_summary(week_df):
    """
    Compute overall statistics for the week.
    """
    total_tests = len(week_df)
    total_duration = week_df['duration (s)'].sum()
    avg_duration = week_df['duration (s)'].mean() if total_tests > 0 else 0
    return {
        'total_tests': total_tests,
        'total_duration': total_duration,
        'avg_duration': avg_duration
    }

def compute_current_engine_stats(week_df):
    """
    Compute, for each engine/test type, the current week's totals.
    """
    current = week_df.groupby('test').agg(
        current_tests=('test', 'size'),
        current_duration=('duration (s)', 'sum')
    ).reset_index()
    return current

def compute_historical_averages(df, week_end):
    """
    Compute historical weekly averages for each engine type.
    Consider all complete weeks (weeks ending on Tuesday).
    """
    hist_df = df[df['date'] <= week_end].copy()
    weekly = hist_df.groupby([pd.Grouper(key='date', freq='W-TUE'), 'test']).agg(
        tests_count=('test', 'size'),
        total_duration=('duration (s)', 'sum')
    ).reset_index()
    historical_avg = weekly.groupby('test').agg(
        avg_tests=('tests_count', 'mean'),
        avg_duration=('total_duration', 'mean')
    ).reset_index()
    return historical_avg

def merge_current_with_history(current, historical):
    """
    Merge current week stats with historical averages.
    Compute percentage differences.
    """
    merged = pd.merge(current, historical, on='test', how='left')
    merged['pct_tests'] = ((merged['current_tests'] - merged['avg_tests']) / merged['avg_tests'] * 100).round(1)
    merged['pct_duration'] = ((merged['current_duration'] - merged['avg_duration']) / merged['avg_duration'] * 100).round(1)
    return merged

# --- Plotting Functions ---

def create_engine_plots(merged_stats):
    """
    Create two grouped bar charts:
      - Number of tests per engine type.
      - Total hotfire duration per engine type.
    Each chart shows this week’s value alongside the historical average.
    Save images in /tmp and return a dict of content IDs to file paths.
    """
    image_paths = {}
    engines = merged_stats['test']
    indices = range(len(engines))
    width = 0.35

    # Plot: Number of tests
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - width/2 for i in indices],
           merged_stats['current_tests'], width, label='This Week')
    ax.bar([i + width/2 for i in indices],
           merged_stats['avg_tests'], width, label='Historical Avg')
    ax.set_xticks(indices)
    ax.set_xticklabels(engines, rotation=45, ha='right')
    ax.set_ylabel("Number of Tests")
    ax.set_title("Engine Test Counts: This Week vs. Historical Average")
    ax.legend()
    plt.tight_layout()
    tests_plot_path = '/tmp/engine_tests.png'
    plt.savefig(tests_plot_path)
    plt.close()
    image_paths['engine_tests'] = tests_plot_path

    # Plot: Hotfire duration
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - width/2 for i in indices],
           merged_stats['current_duration'], width, label='This Week')
    ax.bar([i + width/2 for i in indices],
           merged_stats['avg_duration'], width, label='Historical Avg')
    ax.set_xticks(indices)
    ax.set_xticklabels(engines, rotation=45, ha='right')
    ax.set_ylabel("Hotfire Duration (s)")
    ax.set_title("Engine Hotfire Duration: This Week vs. Historical Average")
    ax.legend()
    plt.tight_layout()
    duration_plot_path = '/tmp/engine_duration.png'
    plt.savefig(duration_plot_path)
    plt.close()
    image_paths['engine_duration'] = duration_plot_path

    logger.info("Engine plots created.")
    return image_paths

def create_daily_trend_plot(week_df):
    """
    Optionally, create a daily trend plot for the current week.
    """
    if week_df.empty:
        return None
    daily_counts = week_df.groupby(week_df['date'].dt.date).size()
    plt.figure(figsize=(8,4))
    daily_counts.sort_index().plot(kind='line', marker='o')
    plt.title('Daily Test Counts (Current Week)')
    plt.xlabel('Date')
    plt.ylabel('Number of Tests')
    plt.tight_layout()
    daily_plot_path = '/tmp/daily_trend.png'
    plt.savefig(daily_plot_path)
    plt.close()
    return daily_plot_path

# --- Email Content Functions ---

def build_email_content(overall, week_start, week_end, merged_stats, image_paths):
    """
    Build an HTML email that:
      - Summarizes overall week totals.
      - Provides a per-engine breakdown with percentage differences.
      - Includes the engine tests and duration plots.
      - Ends with a hyperlink to the CSV file.
    """
    # Build plain text summary
    text_content = f"Weekly SpaceX Engine Test Analysis\n"
    text_content += f"Analysis Week: {week_start.date()} to {week_end.date()}\n\n"
    text_content += f"Overall: {overall['total_tests']} tests, {overall['total_duration']} seconds of hotfire time.\n\n"
    text_content += "Engine Breakdown:\n"
    for _, row in merged_stats.iterrows():
        text_content += (f" - {row['test']}: {row['current_tests']} tests "
                         f"({row['pct_tests']}% vs. avg of {row['avg_tests']:.1f}), "
                         f"{row['current_duration']} sec hotfire ({row['pct_duration']}% vs. avg of {row['avg_duration']:.1f}).\n")
    text_content += f"\nFull CSV log: {CSV_HYPERLINK}\n"

    # Build HTML content
    html_content = f"""
    <html>
      <head>
        <style>
          table, th, td {{
            border: 1px solid black;
            border-collapse: collapse;
            padding: 5px;
          }}
        </style>
      </head>
      <body>
        <h2>Weekly SpaceX Engine Test Analysis</h2>
        <p><strong>Analysis Week:</strong> {week_start.date()} to {week_end.date()}</p>
        <p>
          <strong>Overall:</strong> {overall['total_tests']} tests, {overall['total_duration']} seconds of hotfire time.
        </p>
        <h3>Engine Breakdown</h3>
        <table>
          <tr>
            <th>Engine Type</th>
            <th>This Week (Tests)</th>
            <th>Historical Avg (Tests)</th>
            <th>% Diff (Tests)</th>
            <th>This Week (Duration s)</th>
            <th>Historical Avg (Duration s)</th>
            <th>% Diff (Duration)</th>
          </tr>
    """
    for _, row in merged_stats.iterrows():
        html_content += f"""
          <tr>
            <td>{row['test']}</td>
            <td>{row['current_tests']}</td>
            <td>{row['avg_tests']:.1f}</td>
            <td>{row['pct_tests']}%</td>
            <td>{row['current_duration']}</td>
            <td>{row['avg_duration']:.1f}</td>
            <td>{row['pct_duration']}%</td>
          </tr>
        """
    html_content += """
        </table>
        <h3>Engine Test Counts</h3>
        <img src="cid:engine_tests" alt="Engine Tests Plot">
        <h3>Engine Hotfire Duration</h3>
        <img src="cid:engine_duration" alt="Engine Duration Plot">
    """
    if 'daily_trend' in image_paths:
        html_content += """
        <h3>Daily Trend (Current Week)</h3>
        <img src="cid:daily_trend" alt="Daily Trend Plot">
        """
    html_content += f"""
        <p>For the full test log, please visit: <a href="{CSV_HYPERLINK}">CSV Log on GitHub</a></p>
      </body>
    </html>
    """
    return text_content, html_content

def send_email(subject, text_content, html_content, image_paths):
    """
    Compose and send the email via AWS SES.
    Uses VISIBLE_RECIPIENT for the header and includes BCC_RECIPIENT in the SES Destinations.
    """
    ses_client = boto3.client('ses', region_name=SES_REGION)
    msg = MIMEMultipart('related')
    msg['Subject'] = subject
    # Only the visible recipient appears in the header
    msg['To'] = VISIBLE_RECIPIENT

    # Combine visible and BCC recipients in the SES send call
    all_recipients = [VISIBLE_RECIPIENT, BCC_RECIPIENT]

    msg_alternative = MIMEMultipart('alternative')
    msg.attach(msg_alternative)
    msg_alternative.attach(MIMEText(text_content, 'plain'))
    msg_alternative.attach(MIMEText(html_content, 'html'))
    
    for cid, path in image_paths.items():
        try:
            with open(path, 'rb') as f:
                img_data = f.read()
            image = MIMEImage(img_data)
            image.add_header('Content-ID', f'<{cid}>')
            image.add_header('Content-Disposition', 'inline', filename=os.path.basename(path))
            msg.attach(image)
        except Exception as e:
            logger.error(f"Error attaching image {path}: {e}")
    
    try:
        response = ses_client.send_raw_email(
            Source=SES_SENDER,
            Destinations=all_recipients,
            RawMessage={'Data': msg.as_string()}
        )
        logger.info("Email sent! Message ID: " + response['MessageId'])
    except Exception as e:
        logger.error("Failed to send email: " + str(e))
        raise

# --- Lambda Handler ---

def lambda_handler(event, context):
    logger.info("Starting weekly analysis email process.")
    try:
        clone_path = "/tmp/SpaceX_Eng_Test_Log"
        clone_repo(REPO_URL, clone_path)
        df = fetch_csv_from_repo(clone_path)
        
        week_start, week_end = compute_analysis_week()
        week_df = filter_week_data(df, week_start, week_end)
        
        overall = compute_overall_summary(week_df)
        current_stats = compute_current_engine_stats(week_df)
        historical_avg = compute_historical_averages(df, week_end)
        merged_stats = merge_current_with_history(current_stats, historical_avg)
        
        engine_images = create_engine_plots(merged_stats)
        daily_trend_path = create_daily_trend_plot(week_df)
        if daily_trend_path:
            engine_images['daily_trend'] = daily_trend_path

        text_body, html_body = build_email_content(overall, week_start, week_end, merged_stats, engine_images)
        subject = f"Weekly SpaceX Engine Test Analysis: {week_start.date()} to {week_end.date()}"
        send_email(subject, text_body, html_body, engine_images)
        return {"status": "success", "message": "Email sent successfully."}
    except Exception as e:
        logger.error("Error in lambda_handler: " + str(e))
        return {"status": "error", "message": str(e)}

# For local testing:
if __name__ == '__main__':
    lambda_handler({}, None)
