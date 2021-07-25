# SROIE 2019: Scanned Receipts OCR and Information Extraction

## Introduction
This [challange](https://rrc.cvc.uab.es/?ch=13&com=introduction) consists 3 parts:
- Task 1 - Scanned Receipt Text Localisation: the aim of this task is to accurately localize texts with 4 vertices. In the contest, I used object detection model like
 SSD, EAST.
- Task 2 - Scanned Receipt OCR: the aim of this task is to accurately recognize the text in a receipt image. I implemented CRNN - the combination of CNN and RNN for this recognition.
- Task 3 - Key Information Extraction from Scanned Receipts: the aim of this task is to extract texts of a number of key fields from given receipts and BERT's family is a promise choice for this task.

## Demo
You just change your image path in file ```demo.py``` and run the below script
```bash
cd demo
python3 demo.py

```
