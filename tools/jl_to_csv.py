import csv
import jsonlines
import hashlib

checksums = []
total = 0
total_scam = 0
total_spam = 0
total_other = 0

with open("../dataset/sms.csv", mode="w") as sms:
  csv_writer = csv.writer(sms, delimiter="|")
  csv_writer.writerow(["sender", "type", "received", "message"])

  with jsonlines.open("../dataset/sms.jl") as reader:
    for obj in reader:

      # fraud sms mostly come from regular number
      if obj["type"] == "Penipuan" and len(obj["sender"]) > 10:
        sms_type = "SCAM"
      else:
        # sms from short number usually come from operator, OTP
        if len(obj["sender"]) < 10:
          sms_type = "OTHER"
        else:
          sms_type = "SPAM"

      cleaned_msg = obj["message"].strip().replace("\r", " ").replace("\n", " ")
      checksum = hashlib.md5()
      checksum.update(cleaned_msg)
      hexdigest = checksum.hexdigest()

      # avoid duplicate message
      if hexdigest in checksums:
        print "Duplicate message with checksum: %s" % hexdigest
      else:
        total += 1

        if sms_type == "SCAM": total_scam +=1
        if sms_type == "SPAM": total_spam +=1
        if sms_type == "OTHER": total_other +=1

        checksums.append(hexdigest)
        csv_writer.writerow([obj["sender"], sms_type, obj["received"], cleaned_msg])

print "total=%s, scam=%s, spam=%s, other=%s" % (total, total_scam, total_spam, total_other)
print "recommended training dataset number: %s" % (total * 0.7) # 70% of total messages.