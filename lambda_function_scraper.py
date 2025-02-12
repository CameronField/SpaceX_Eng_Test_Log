import os
import subprocess
from paddleocr import PaddleOCR
import re
import csv
from datetime import datetime, timedelta
import shutil
import boto3
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Hardcoded SES and GitHub credentials (replace placeholders with your actual values)
SES_SENDER_EMAIL = "#INPUT_SES_SENDER_EMAIL"
SES_RECIPIENT_EMAILS = ["#INPUT_RECIPIENT_EMAIL"]    # List of recipient emails

GITHUB_USERNAME = "#INPUT_GITHUB_USERNAME"           # Your GitHub username
GITHUB_TOKEN = "#INPUT_GITHUB_TOKEN"                   # Your GitHub token

# Initialize global variables for PaddleOCR and SES client
ocr_model = None
ses_client = None

def initialize_services():
    global ocr_model, ses_client
    if ocr_model is None:
        try:
            logger.info("Initializing PaddleOCR.")
            # Specify the directories where models are pre-downloaded
            ocr_model = PaddleOCR(
                det_model_dir='/var/task/paddleocr_models/det/en_PP-OCRv3_det_infer/',
                rec_model_dir='/var/task/paddleocr_models/rec/en_PP-OCRv3_rec_infer/',
                cls_model_dir='/var/task/paddleocr_models/cls/ch_ppocr_mobile_v2.0_cls_infer/',
                use_angle_cls=True,
                lang='en'
            )
            logger.info("PaddleOCR initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing PaddleOCR: {e}")
            raise e
    
    if ses_client is None:
        try:
            logger.info("Initializing SES client.")
            ses_client = boto3.client('ses', region_name='us-east-1')  # Replace with your region if needed
            logger.info("SES client initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing SES client: {e}")
            raise e

def pipe_yt_dlp_to_ffmpeg(youtube_url, frame_path):
    """
    Uses yt-dlp to fetch the stream and pipes its stdout directly into ffmpeg.
    ffmpeg then extracts a single frame and saves it to `frame_path`.
    Returns a tuple of (frame_path, error_message). If successful, error_message is None.
    """
    ytdlp_command = [
        "yt-dlp",
        "--cookies", "/var/task/cookies.txt",
        "--add-header", "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "--add-header", "Accept-Language: en-US,en;q=0.9",
        "-f", "bestvideo[height<=1080]+bestaudio/best",
        "-o", "-",  # Output to stdout
        youtube_url
    ]

    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",  # Change to 'error' to reduce verbosity
        "-i", "pipe:0",         # Read from stdin
        "-vf", "scale=1920:1080",
        "-frames:v", "1",
        frame_path
    ]

    logger.info(f"Running pipeline: {' '.join(ytdlp_command)} | {' '.join(ffmpeg_command)}")
    try:
        # Run yt-dlp and pipe its stdout to ffmpeg's stdin
        p1 = subprocess.Popen(ytdlp_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p2 = subprocess.run(ffmpeg_command, stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        p1.stdout.close()
        stderr = p2.stderr.decode()
        if p2.returncode != 0:
            logger.error(f"ffmpeg error: {stderr}")
            return None, f"ffmpeg error: {stderr}"
        return frame_path, None
    except subprocess.CalledProcessError as e:
        error_output = e.stderr.decode() if e.stderr else "Unknown error"
        logger.error(f"Pipeline command failed (exit code {e.returncode}). Error: {error_output}")
        return None, f"Pipeline command failed: {error_output}"
    except Exception as e:
        logger.error(f"Error during yt-dlp->ffmpeg pipeline: {e}")
        return None, str(e)

def extract_text(image_path):
    """
    Uses PaddleOCR to extract text directly from the original image.
    """
    try:
        # Perform OCR
        result = ocr_model.ocr(image_path, rec=True, cls=True)

        logger.debug(f"Full OCR Result:\n{result}\n")

        extracted_text = []

        # Process each image in the OCR result.
        for image_idx, image_result in enumerate(result):
            logger.info(f"Processing image {image_idx + 1}")
            
            for detection_idx, detection in enumerate(image_result):
                logger.info(f"  Processing detection {detection_idx + 1}: {detection}")
                bounding_box, (text, confidence) = detection

                MIN_CONFIDENCE = 0.65  # Adjust if needed.
                if confidence < MIN_CONFIDENCE:
                    logger.info(f"    Skipping low-confidence text: '{text}' with confidence {confidence}")
                    continue

                if isinstance(text, str):
                    cleaned_piece = text.strip()
                    extracted_text.append(cleaned_piece)
                    logger.info(f"    Detected text: '{cleaned_piece}' with confidence {confidence}")
                elif isinstance(text, list):
                    flat_text = ' '.join(str(item) for item in text)
                    extracted_text.append(flat_text)
                    logger.info(f"    Detected text (flattened): '{flat_text}' with confidence {confidence}")
                else:
                    logger.info(f"    Unexpected type for text: {type(text)}")

        final_text = '\n'.join(extracted_text)
        logger.info(f"Extracted Text:\n{final_text}")
        return final_text
    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        return ""

def clean_text(text):
    try:
        # Remove unwanted characters.
        cleaned_text = re.sub(r"[^\w\s:\-]", "", text)
        
        # Ensure proper spacing in time fields:
        cleaned_text = re.sub(r"(\d{1,2}:\d{2}:\d{2})([AP]M)", r"\1 \2", cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r"(\d{1,2}:\d{2}:\d{2}\s+[AP]M)[A-Z]+", r"\1", cleaned_text, flags=re.IGNORECASE)
        
        # Fix F9 Stage 2 issues:
        cleaned_text = re.sub(r"\b(F9Stage2)(?=\d)", r"F9 Stage 2 ", cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r"\bF9Stage2\b", "F9 Stage 2", cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r"\b(F9\s+Stage)(\d)", r"\1 \2", cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r"\b(F9\s+Stage\s+2)(?=\d)", r"\1 ", cleaned_text, flags=re.IGNORECASE)

        # Fix F9 Stage 1 issues:
        cleaned_text = re.sub(r"\b(F9Stage1)(?=\d)", r"F9 Stage 1 ", cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r"\bF9Stage1\b", "F9 Stage 1", cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r"\b(F9\s+Stage)(\d)", r"\1 \2", cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r"\b(F9\s+Stage\s+1)(?=\d)", r"\1 ", cleaned_text, flags=re.IGNORECASE)
        
        # Insert missing spaces between test names and time:
        cleaned_text = re.sub(r"\b(Merlin)(?=\d)", r"\1 ", cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r"(Raptor)(Horizontal|Vertical|Tripod)", r"\1 \2", cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r"\b(Raptor\s+(?:Horizontal|Vertical|Tripod))(?=\d)", r"\1 ", cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r"\b(MVac|RVac)(?=\d)", r"\1 ", cleaned_text, flags=re.IGNORECASE)
        
        # Insert a space between AM/PM and a following digit.
        cleaned_text = re.sub(r"([AP]M)(\d)", r"\1 \2", cleaned_text, flags=re.IGNORECASE)
        
        # Remove extraneous negative offset times.
        cleaned_text = re.sub(r"\-\d{1,2}:\d{2}", "", cleaned_text)
        
        # Compress multiple spaces.
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
        
        logger.info(f"Cleaned Text:\n{cleaned_text}")
        return cleaned_text
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        return text

def parse_to_rows(text):
    try:
        text = clean_text(text)
        # The regex now makes the duration group optional.
        pattern = r"""
            \b
            (
                Merlin(?:\s+Tripod)?|
                Raptor\s+(?:Horizontal|Vertical|Tripod)|
                MVac(?:\s+Tripod)?|
                F9\s+Stage\s+2|
                F9\s+Stage\s+1|
                RVac(?:\s+Tripod)?
            )\s+
            (\d{1,2}:\d{2}:\d{2}\s+(?:AM|PM))
            (?:\s+((?:\d+s|\d+|PB)))?
            \b
        """
        matches = re.findall(pattern, text, re.IGNORECASE | re.VERBOSE)
        logger.debug(f"Regex matches: {matches}")

        rows = []
        date_yesterday = (datetime.now() - timedelta(days=1)).strftime("%b %d %Y")

        for test, time_str, duration in matches:
            duration_val = duration.rstrip("s") if duration and duration.endswith("s") else (duration if duration else "")
            rows.append({
                "date": date_yesterday,
                "test": test.strip(),
                "time": time_str.strip(),
                "duration (s)": duration_val
            })

        logger.info(f"Parsed rows: {rows}")
        return rows
    except Exception as e:
        logger.error(f"Error parsing text: {e}")
        return []

def save_to_csv(new_rows, file_path):
    """
    Uses the Python csv module to append new rows, removing duplicates.
    """
    fieldnames = ["date", "test", "time", "duration (s)"]
    existing_records = []

    if os.path.exists(file_path):
        with open(file_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_records = list(reader)

    existing_set = set((r["date"], r["test"], r["time"], r["duration (s)"]) for r in existing_records)

    for row in new_rows:
        row_key = (row["date"], row["test"], row["time"], str(row["duration (s)"]))
        if row_key not in existing_set:
            existing_records.append(row)
            existing_set.add(row_key)

    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_records)
    logger.info(f"Data saved to {file_path}")
    return file_path

def set_git_config(repo_path):
    """
    Set Git user email/name for commits.
    """
    try:
        os.chdir(repo_path)
        subprocess.run(["git", "config", "user.email", "#INPUT_EMAIL"], check=True)
        subprocess.run(["git", "config", "user.name", "#INPUT_NAME"], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error setting git config: {e}")

def commit_and_push(file_path, repo_path, commit_message="Update test log"):
    """
    Adds, commits, and pushes an updated CSV to GitHub.
    """
    try:
        if not os.path.exists(repo_path):
            logger.error(f"Repository path {repo_path} does not exist.")
            return False
        os.chdir(repo_path)
        if not os.path.exists(file_path):
            logger.error(f"File {file_path} does not exist.")
            return False

        subprocess.run(["git", "add", file_path], check=True)
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        subprocess.run(["git", "push"], check=True)
        logger.info("Changes successfully pushed to GitHub.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error committing/pushing to GitHub: {e}")
        return False
    except Exception as e:
        logger.error(f"General error during commit/push: {e}")
        return False

def send_email(subject, body_text, body_html, image_path=None, error=False):
    """
    Sends an email with the specified subject and body.
    Optionally attaches an image.
    """
    try:
        msg = MIMEMultipart('mixed')
        msg['Subject'] = subject
        msg['From'] = SES_SENDER_EMAIL
        msg['To'] = ", ".join(SES_RECIPIENT_EMAILS)

        msg_body = MIMEMultipart('alternative')
        part1 = MIMEText(body_text, 'plain')
        part2 = MIMEText(body_html, 'html')
        msg_body.attach(part1)
        msg_body.attach(part2)
        msg.attach(msg_body)

        if image_path and os.path.exists(image_path):
            with open(image_path, 'rb') as img:
                img_data = img.read()
                image_part = MIMEApplication(img_data, _subtype="jpeg")
                image_part.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image_path))
                image_part.add_header('Content-ID', '<frame_image>')
                msg.attach(image_part)
        elif image_path:
            logger.warning(f"Image path {image_path} does not exist. Skipping attachment.")

        response = ses_client.send_raw_email(
            Source=SES_SENDER_EMAIL,
            Destinations=SES_RECIPIENT_EMAILS,
            RawMessage={'Data': msg.as_string()}
        )

        logger.info("Email sent! Message ID: {}".format(response['MessageId']))
        return True
    except Exception as e:
        logger.error(f"Error sending email: {e}")
        return False

def lambda_handler(event, context):
    """
    Main function for AWS Lambda.
    """
    initialize_services()

    youtube_url = "#INPUT_YOUTUBE_URL"
    frame_path = "/tmp/frame.jpg"
    repo_path = "/tmp/Your_Repo_Folder"  # Adjust if necessary
    file_name = "test_logs.csv"

    initial_working_directory = os.getcwd()

    yt_dlp_ffmpeg_error = False
    ocr_no_data = False
    error_message = ""

    date_today = datetime.now().strftime("%b %d %Y")
    date_yesterday = (datetime.now() - timedelta(days=1)).strftime("%b %d %Y")

    try:
        if not GITHUB_USERNAME or not GITHUB_TOKEN:
            raise ValueError("GitHub credentials missing.")

        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)

        repo_url = f"https://{GITHUB_USERNAME}:{GITHUB_TOKEN}@github.com/#INPUT_GITHUB_REPO.git"
        logger.info(f"Cloning repository: {repo_url}")
        result = subprocess.run(["git", "clone", repo_url, repo_path], capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Error cloning: {result.stderr}")
            yt_dlp_ffmpeg_error = True
            error_message = f"Error cloning GitHub repo: {result.stderr}"
    except Exception as e:
        logger.error(f"Error cloning GitHub repo: {e}")
        yt_dlp_ffmpeg_error = True
        error_message = str(e)

    if not yt_dlp_ffmpeg_error:
        frame_captured, pipeline_error = pipe_yt_dlp_to_ffmpeg(youtube_url, frame_path)
        if not frame_captured:
            yt_dlp_ffmpeg_error = True
            error_message = f"Failed to capture frame: {pipeline_error}"

    if yt_dlp_ffmpeg_error:
        subject = f"Lambda Function Error: Test Log - {date_yesterday}"
        body_text = (
            f"An error occurred during the execution of the Test Log Lambda function on {date_yesterday}.\n\n"
            f"Error Details:\n{error_message}\n\n"
            f"View the test log data here: #INPUT_CSV_HYPERLINK"
        )
        body_html = f"""<html>
        <head></head>
        <body>
            <p>An error occurred during the execution of the Test Log Lambda function on {date_yesterday}.</p>
            <p><strong>Error Details:</strong></p>
            <pre>{error_message}</pre>
            <p>View the test log data here: <a href="#INPUT_CSV_HYPERLINK">Test Logs CSV</a></p>
        </body>
        </html>"""

        email_sent = send_email(subject, body_text, body_html, image_path=frame_path)
        if email_sent:
            return {"status": "error", "message": "An error occurred. Notification email sent."}
        else:
            return {"status": "error", "message": "An error occurred. Failed to send notification email."}

    text = extract_text(frame_path)
    rows = parse_to_rows(text)
    if not rows:
        ocr_no_data = True

    if not ocr_no_data:
        csv_path = os.path.join(repo_path, file_name)
        save_to_csv(rows, csv_path)

    if not ocr_no_data:
        try:
            set_git_config(repo_path)
            os.chdir(initial_working_directory)
            push_success = commit_and_push(file_name, repo_path)
            os.chdir(initial_working_directory)
            if not push_success:
                yt_dlp_ffmpeg_error = True
                error_message = "Failed to push data to GitHub."
            else:
                logger.info("Data updated & pushed to GitHub")
        except Exception as e:
            logger.error(f"Error pushing data: {e}")
            yt_dlp_ffmpeg_error = True
            error_message = f"Error pushing data to GitHub: {e}"

    if yt_dlp_ffmpeg_error:
        subject = f"Lambda Function Error: Test Log - {date_yesterday}"
        body_text = (
            f"An error occurred during the execution of the Test Log Lambda function on {date_yesterday}.\n\n"
            f"Error Details:\n{error_message}\n\n"
            f"View the test log data here: #INPUT_CSV_HYPERLINK"
        )
        body_html = f"""<html>
        <head></head>
        <body>
            <p>An error occurred during the execution of the Test Log Lambda function on {date_yesterday}.</p>
            <p><strong>Error Details:</strong></p>
            <pre>{error_message}</pre>
            <p>View the test log data here: <a href="#INPUT_CSV_HYPERLINK">Test Logs CSV</a></p>
        </body>
        </html>"""

        email_sent = send_email(subject, body_text, body_html, image_path=frame_path)
        if email_sent:
            return {"status": "error", "message": "An error occurred. Notification email sent."}
        else:
            return {"status": "error", "message": "An error occurred. Failed to send notification email."}

    if ocr_no_data:
        subject = f"Test Log: No Tests Recorded - {date_yesterday}"
        body_text = (
            f"No tests were recorded on {date_yesterday}.\n\n"
            f"View the test log data here: #INPUT_CSV_HYPERLINK"
        )
        body_html = f"""<html>
        <head></head>
        <body>
            <p>No tests were recorded on {date_yesterday}.</p>
            <p>View the test log data here: <a href="#INPUT_CSV_HYPERLINK">Test Logs CSV</a></p>
            <p>Captured Image:</p>
            <img src="cid:frame_image" alt="Captured Frame" />
        </body>
        </html>"""

        email_sent = send_email(subject, body_text, body_html, image_path=frame_path)
        if email_sent:
            return {"status": "success", "message": "No tests recorded today. Notification email sent."}
        else:
            return {"status": "error", "message": "No tests recorded today. Failed to send notification email."}

    subject = f"Test Log: Daily Report - {date_yesterday}"
    body_text = f"The following tests were performed on {date_yesterday}:\n"
    for test in rows:
        body_text += f"- {test['test']} at {test['time']} for {test['duration (s)']} seconds\n"
    body_text += (
        "\nView the test log here: #INPUT_CSV_HYPERLINK. "
        "If you see any errors with this data compared to the image below, please contact #INPUT_CONTACT_INFO."
    )

    body_html = f"""<html>
    <head></head>
    <body>
        <p>The following tests were performed on {date_yesterday}:</p>
        <table border="1" cellpadding="5" cellspacing="0">
            <tr>
                <th>Date</th>
                <th>Test</th>
                <th>Time</th>
                <th>Duration (s)</th>
            </tr>
    """
    for test in rows:
        body_html += f"""<tr>
            <td>{test['date']}</td>
            <td>{test['test']}</td>
            <td>{test['time']}</td>
            <td>{test['duration (s)']}</td>
        </tr>"""
    body_html += "</table>"
    body_html += f"""
        <p>View the test log here: <a href="#INPUT_CSV_HYPERLINK">Test Logs CSV</a>. 
        If you see any errors with this data compared to the image below, please contact 
        <a href="mailto:#INPUT_CONTACT_EMAIL">#INPUT_CONTACT_EMAIL</a>.</p>
        <p>Captured Image:</p>
        <img src="cid:frame_image" alt="Captured Frame" />
    </body>
    </html>
    """

    email_sent = send_email(subject, body_text, body_html, image_path=frame_path)
    if email_sent:
        return {"status": "success", "message": "Data updated, pushed to GitHub, and email sent"}
    else:
        return {"status": "error", "message": "Data updated & pushed to GitHub, but failed to send email"}

def main():
    """
    Local testing function.
    """
    event = {}
    context = None
    result = lambda_handler(event, context)
    print(result)

if __name__ == "__main__":
    main()
