
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr


sender = '241811605@qq.com'

password = 'navofjaxyfowbieg'

receiver = '838009959@qq.com'


message = MIMEText('Your task on AutoDL has completed', 'plain', 'utf-8')

message['From'] = formataddr(('Autodl', sender))
message['To'] = formataddr(('snoz', receiver))

message['Subject'] = 'AutoDL_server task'

try:

    server = smtplib.SMTP_SSL('smtp.qq.com', 465)

    server.login(sender, password)

    server.sendmail(sender, [receiver], message.as_string())

    server.quit()
    print("success!")
except Exception as e:
    print("fail", e)

