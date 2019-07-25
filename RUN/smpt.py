import smtplib
from email.message import EmailMessage
import imghdr
import config
import os

# Email Object
msg = EmailMessage()
msg['Subject'] = config.SUBJECT
msg['From'] = config.EMAIL_SENDER
msg['To'] = config.EMAIL_RECIPIENT
msg.set_content(config.CONTENT)

# Email Attachment
with open('images.jpg', 'rb') as f:
    file_data = f.read()
    file_type = imghdr.what(f.name)
    file_name = f.name

# Add attachement - jpg
msg.add_attachment(file_data, maintype='image', subtype=file_type, filename=file_name)

# Run SMTP protocol
with smtplib.SMTP('smtp.gmail.com:587') as server:
    server.ehlo()
    server.starttls()
    server.login(config.EMAIL_SENDER, config.PASSWORD)
    server.send_message(msg)
    server.quit()
