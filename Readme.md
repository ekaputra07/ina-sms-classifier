# ina-sms-classifier

A project to create Machine Learning model to classify Indonesian text/sms messages using [Tensorflow](https://www.tensorflow.org) via its [Keras](https://keras.io) api.

The main puspose is **to be able to detect scam/fraud SMS that often received by mobile phone users in Indonesia from unknown person and many have been reported to be victims of this kind of fraud activity**.

_Future plan_: the model can be transformed into [Tensorflow Lite](https://www.tensorflow.org/lite) and can be deployed as a mobile app that classify text message in real-time as it received by users. No need to send the message to model serving server to avoid privacy issue.

For now, it will classify messages into 4 classes:

- Scam (0)
- Online gambling website promotion (1)
- Online loans website promotion (2)
- Others (3)

Thanks to [laporsms.com](https://laporsms.com) for their effort collecting all the data that I've been using in this project.


```
Copyright (C) 2020  Eka Putra

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
```