import csv
import jsonlines

with open("../dataset/sms.csv", mode="w") as sms:
  csv_writer = csv.writer(sms, delimiter="|")
  csv_writer.writerow(["sender", "type", "received", "message"])

  with jsonlines.open("../dataset/sms.jl") as reader:
    for obj in reader:

      if obj["type"] == "Penipuan":
        sms_type = "SCAM"
      else:
        sms_type = "SPAM"

      cleaned_msg = obj["message"].strip().replace("\r", "").replace("\n", " ")
      csv_writer.writerow([obj["sender"], sms_type, obj["received"], cleaned_msg])