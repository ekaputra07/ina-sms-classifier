import csv
import jsonlines

with open("../dataset/sms.csv", mode="w") as sms:
  csv_writer = csv.writer(sms, delimiter="|")
  csv_writer.writerow(["message", "label"])

  with jsonlines.open("../dataset/sms.jl") as reader:
    for obj in reader:

      # fraud sms mostly come from regular number
      if obj["type"] == "Penipuan":
        label = 1
      else:
        label = 0

      cleaned_msg = obj["message"].strip().replace("\r", " ").replace("\n", " ")
      csv_writer.writerow([cleaned_msg, label])