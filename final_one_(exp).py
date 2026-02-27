
from pdf2image import convert_from_path
import time
import requests
from datetime import datetime
from dateutil import parser
from requests.auth import HTTPBasicAuth
import os
from io import BytesIO
from datetime import datetime,timedelta
from PIL import Image
import base64
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import json
import re
import logging
import shutil
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import base64
import uuid
import traceback
from itertools import cycle

load_dotenv()

app = Flask(__name__)


UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


GOOGLE_API_KEYS = os.getenv("GOOGLE_API_KEYS", "").split(",")
sap_username =  os.getenv('SAP_USERNAME')
sap_password =  os.getenv('SAP_PASSWORD')

if not GOOGLE_API_KEYS or GOOGLE_API_KEYS == [""]:
    raise ValueError("No Google API keys found in env")

google_key_cycle = cycle(GOOGLE_API_KEYS)

def get_gemini_model(api_key):

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=0,
        google_api_key=api_key
    )



LOG_FILE =f"{datetime.now().strftime('%d_%m_%Y')}.log"
logs_path = os.path.join(os.getcwd(),"logs")
os.makedirs(logs_path,exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path,LOG_FILE)

yesterday = (datetime.now() - timedelta(days=2)).strftime("%d_%m_%Y") + ".log"
yesterday_log_path = os.path.join(logs_path, yesterday)

if os.path.isfile(yesterday_log_path):
    os.remove(yesterday_log_path)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format= "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    filemode= 'a',
    level = logging.INFO
)


@app.route('/', methods=['GET'])
def index():
    return render_template('New_index.html')

prompt_template = """You are a deterministic information extraction engine.

  Your task is to extract structured data ONLY from the provided document.
  You MUST follow all rules strictly and in the given order.
  Do not provide hallucinated or wrong answers. Extract text exactly as it appears.

  ====================
  HANDLING INSTRUCTIONS FOR UNCLEAR IMAGES
  ====================
  - If the image is tilted or skewed, mentally correct the orientation and read accordingly.
  - If any text is blurry or partially visible, use context clues from surrounding text
    to make your best inference.
  - If a field is completely unreadable, mark it as [UNREADABLE].
  - If a field is not present in the invoice, mark it as [NOT FOUND].
  - Do NOT skip rows from the material table — extract every line item visible.
  - If numbers are ambiguous, prefer the interpretation that makes mathematical sense.
  - Verify your extracted line item totals match the invoice subtotal/total where possible.

  ====================
  DOCUMENT TYPE RULE
  ====================
  If the document text contains any of the following (case-insensitive):
  "TAX INVOICE", "Tax Invoice", "TEX INVOICE"
  Then include: "Tax Invoice": Yes, Otherwise: "Tax Invoice": No

  ====================
  PAN AND GSTIN — CORE UNDERSTANDING (READ THIS FIRST BEFORE ANY EXTRACTION)
  ====================

  PAN:
  - Always exactly 10 characters
  - Format: 5 LETTERS + 4 DIGITS + 1 LETTER
  - Example: AABFG8026H
  - Labels: "PAN", "PAN No", "PAN Number", "Company's PAN", "PAN/IT No"

  GSTIN:
  - Always exactly 15 characters
  - Format: 2 digit state code + 10 char PAN + 1 digit + 1 alphanumeric + 1 check char
  - Example: 24AABFG8026H1ZN
  - Labels: "GSTIN", "GSTIN/UIN", "GST No", "GSTIN No", "GST Registration No"

  RELATIONSHIP BETWEEN PAN AND GSTIN:
  - GSTIN characters at position [2:12] will always equal the PAN of that same entity
  - Example: GSTIN = 24AABFG8026H1ZN → PAN = AABFG8026H
  - Use this ONLY as a validation cross-check, not as a replacement for reading

  DERIVING PAN FROM GSTIN (when PAN not explicitly printed):
  - If PAN label is not found anywhere in the document for vendor or buyer →
    derive PAN by extracting characters at position [2:12] from their GSTIN
  - Example: vendor GSTIN = 24AABFG7831N1Z7 → vendors pan no = AABFG7831N

  STRICT SEPARATION RULE:
  - NEVER assign a 15-character GSTIN value into a PAN field
  - NEVER assign a 10-character PAN value into a GSTIN field
  - If PAN field accidentally contains 15 characters → extract only chars [2:12] as PAN

  GSTIN CAREFUL READING RULE:
  - Read each GSTIN character by character from its exact location in the document
  - Do not rely on memory or assumption — go back to source text each time
  - After extracting any GSTIN re-read it one more time independently before finalizing
  - Pay extra attention to the last 4 characters as they are most prone to misread
    due to image skew, blur, or stamp overlap

  CROSS-VALIDATION RULE (use as correction trigger, not rejection):
  - After extracting both GSTIN and PAN for vendor → check if GSTIN[2:12] == PAN
  - After extracting both GSTIN and PAN for buyer → check if GSTIN[2:12] == PAN
  - If they do NOT match → do NOT reject or guess
    Go back and re-read GSTIN carefully from its source location
    Also re-read PAN carefully from all pages (PAN may be on a different page)
    Accept the reading that makes both values consistent

  ====================
  GSTIN CLASSIFICATION — VENDOR vs BUYER (MOST CRITICAL STEP)
  ====================

  Follow this exact 3-step process before assigning any GSTIN:

  STEP 1 — COLLECT ALL GSTINs FROM ENTIRE DOCUMENT:
  Scan the complete document and list every 15-character alphanumeric string
  that starts with 2 digits. These are all GSTINs present in the document.

  STEP 2 — IDENTIFY BUYER GSTIN FIRST (Buyer details are always in a fixed area):
  Buyer information is always printed in a dedicated fixed block on the invoice.
  A GSTIN belongs to BUYER if it appears inside this buyer block near ANY of:
    "Bill To", "Buyer", "M/s" (in buyer address section),
    "Place of Supply", "Consignee", "Ship To",
    "GSTIN No." label appearing beside buyer name or address
  → Assign this as "buyers gstin no"
  → This is the ONLY location to look for buyer GSTIN
  → Do NOT pick buyer GSTIN from any other location

  STEP 3 — IDENTIFY VENDOR GSTIN (Vendor GSTIN location varies across invoices):
  Any GSTIN that was NOT assigned to buyer in Step 2 belongs to the VENDOR.
  Vendor GSTIN can appear in ANY of these locations — scan ALL of them:

    LOCATION 1 — Header / Top of Invoice:
      Near vendor company name, letterhead, "Bill From", "Supplier", "M/s"
      This is the most common location.

    LOCATION 2 — Below Material Table:
      After the last item row, near bank details, subtotal, or grand total.
      A 15-character alphanumeric string starting with 2 digits in this zone
      is the vendor GSTIN.
      Example: "GSTIN No.: 24AABFG7831N1Z7" near bank details = vendor GSTIN

    LOCATION 3 — Inside Material Table:
      Sometimes printed as a standalone row or cell within the table itself.
      If a cell contains a 15-character alphanumeric string starting with
      2 digits → it is a GSTIN not a product row.
      Extract and classify as vendor GSTIN.

    LOCATION 4 — Footer:
      At the very bottom of the page in smaller or lighter font.

  SCAN PRIORITY: Header → Below Table → Inside Table → Footer

  CRITICAL RULES:
  - NEVER assign a GSTIN from below the table or footer to buyer
  - NEVER assign the buyer block GSTIN to vendor
  - If only ONE GSTIN found in entire document → it belongs to vendor
    Return NA for buyers gstin no
  - If TWO GSTINs found → classify using Step 2 and Step 3 above
  - NEVER assign same GSTIN to both vendor and buyer
  - NEVER leave vendor gstin no as NA if any unassigned 15-char GSTIN
    exists anywhere in the document

  ====================
  HEADER FIELDS
  ====================

  In output use EXACT key names below:

  - "buyer" (this field is must)
    → Company name from the fixed buyer block
    → Labels: "Bill To", "Buyer", "M/s" in buyer section
    → If label missing, use clearly identified buyer company name from buyer area

  - "buyers pan no" (this field is must)
    → Extract ONLY the 10-character PAN value
    → Labels: "PAN", "PAN No", "PAN Number", "PAN/IT No", "Buyer PAN"
    → Must be exactly 10 characters: 5 letters + 4 digits + 1 letter
    → Look inside the buyer block first
    → PAN may also appear on a different page — scan ALL pages
    → If PAN label is not found anywhere → derive from buyers gstin no[2:12]
    → If buyers gstin no is also NA → return NA

  - "buyers gstin no" (this field is must)
    → Extract ONLY from the fixed buyer block area (see GSTIN CLASSIFICATION Step 2)
    → Must be exactly 15 characters
    → Re-read last 4 characters one more time before finalizing
    → Validate: GSTIN[2:12] should match buyers pan no
      If mismatch → re-read from buyer block carefully, also re-check PAN from all pages

  - "buyers order number" (this field is must)
    → May appear anywhere in the document
    → Labels (case-insensitive):
      Buyers Order No, Buyer Order Number, Customer Order No,
      Customer Order Number, Order No, Order Number, Order No.,
      Purchase Order, Purchase Order No, PO, P.O., PO No,
      P.Order, Account PO, Customer Reference No

  - "vendor" (this field is must)
    → Company name of the seller/supplier
    → Look in: document header, letterhead, "Bill From", "Supplier",
      "For [Company Name]" signature block at bottom of invoice
    → If "Bill From" label is missing, use detected seller company name

  - "vendors pan no" (this field is must)
    → Extract ONLY the 10-character PAN value
    → Labels: "PAN", "PAN No", "PAN Number", "Company's PAN"
    → Must be exactly 10 characters: 5 letters + 4 digits + 1 letter
    → Vendor PAN location also varies — scan ALL locations and ALL pages
    → Common locations: header, footer, last page, near bank details
    → If PAN label is not found anywhere → derive from vendor gstin no[2:12]

  - "vendor gstin no" (this field is must)
    → Follow GSTIN CLASSIFICATION Step 3 above
    → Scan ALL locations: header, below table, inside table, footer
    → Must be exactly 15 characters
    → Re-read last 4 characters one more time before finalizing
    → NEVER leave as NA if any unassigned 15-char GSTIN exists in document
    → Validate: GSTIN[2:12] must match vendors pan no
      If mismatch → re-read both from source carefully

  - "invoice number" (this field is must)
    → Labels (case-insensitive):
      Invoice Number, Invoice No, Invoice No., Inv. No

  - "invoice date" (this field is must)
    → Labels (case-insensitive):
      Invoice Date, Inv. Date, Dated, Date

  ====================
  MULTI-PAGE INVOICE RULE (STRICT)
  ====================

  Some invoices span multiple pages. Each page repeats the header
  (vendor, buyer, GSTIN, invoice number) but the material table
  continues from where the previous page ended.

  RULES:
  1. Treat the ENTIRE invoice as ONE single document, not separate pages.
  2. Extract material rows ONLY ONCE — do not repeat rows already extracted
     from a previous page.
  3. If a later page header repeats the same sr no 1, 2, 3... already seen
     on page 1 → these are repeated header rows, SKIP them entirely.
  4. A row is DUPLICATE if it has same sr no + hsn + qty + amount
     as a previously extracted row → discard it, keep only first occurrence.
  5. Rows on later pages with sr no CONTINUING from previous page
     (e.g., page 1 ends at sr 5, page 2 starts at sr 6) →
     these are NEW rows, include them.
  6. Special adjustment rows like "R. OFF", "Round Off", "Less: R. OFF"
     with no sr no → extract ONCE only, do not repeat across pages.

  DEDUPLICATION STEP (mandatory before returning output):
    After collecting all rows across all pages:
    - Remove any row where (sr no + hsn + unit price) combination
      already exists in the list — keep only first occurrence.
    - For adjustment/round-off rows → keep only one instance total.

  ====================
  MATERIAL TABLE RULE
  ====================

  Use ONLY ONE key for materials:
  "material": [ list of objects ]

  Each material item may contain ONLY the following keys
  (use EXACT names, no variations):

  - "sr no" (this field is must)
    Labels: Sr No, SrNo, SI No., Item, Sr, S No.

  - "hsn" (this field is must)
    Labels: HSN code, HSN, HSN/SAC, SAC Code

  - "qty" (this field is must)
    Labels: Qty, Quantity
    Extract ONLY the numeric value — do NOT include unit text.
    Example: "2 NOS." → qty = "2"
    Example: "5 KGS" → qty = "5"
    Example: "10" → qty = "10"
    Note: if unit text appears inside qty cell → extract unit text
          separately into the "unit" field, not here.

  - "unit" (this field is must)
    Labels: Unit, per, UoM
    Note: if column not available → look for unit text inside qty cell
    If still not found → set as None

  - "unit price" (this field is must)
    Labels: Rate, Unit Price, Price

  - "discount" (this field is must)
    Labels: Discount, Dis, Disc, Disc.%
    Scan the row AND summary section. Return numeric value only.
    If value is 0.00 → return "0.00", do NOT return NA
    If column not present and no discount found anywhere → return NA

  - "taxable amt" (this field is must)
    Labels: Taxable Amt, Taxable Amount, Taxable Value, Taxable Value Currency INR
    Note: if no taxable amt column exists in the row →
    derive it as: qty × unit price − discount amount

  - "cgst %" (this field is must)
    Labels: CGST, CGST rate, CGST %, OUTPUT CGST @, Central Tax,
            GST % (when split equally between CGST and SGST)
    Extract numeric value only — do NOT include "%" symbol

  - "cgst amount" (this field is must)

  - "sgst %" (this field is must)
    Labels: SGST, SGST rate, SGST %, OUTPUT SGST @, State/UT Tax,
            GST % (when split equally between CGST and SGST)
    Extract numeric value only — do NOT include "%" symbol

  - "sgst amount" (this field is must)
    Labels: SGST Amount, SGST, State/UT Tax amount

  - "amount" (this field is must)
    Labels: Amount, Line Total, Net Amount

  ====================
  CGST / SGST EXTRACTION RULE (STRICT)
  ====================

  CGST and SGST values appear in TWO possible places — check BOTH:

  CASE A — Inline in Material Table Row:
  → Each row has its own CGST %, CGST Amount, SGST %, SGST Amount columns
  → Extract directly from each row

  CASE B — Summary Box at Bottom of Invoice:
  → A single combined CGST and SGST amount shown for all items together
  → Labels: "OUTPUT CGST", "OUTPUT SGST", "Central Tax", "State/UT Tax"
  → PROPORTIONAL DISTRIBUTION (mandatory — do NOT split equally):
      For each item row:
      item_cgst_amount = ROUND((item_taxable_amt / total_taxable_amt) × total_cgst_amount, 2)
      item_sgst_amount = ROUND((item_taxable_amt / total_taxable_amt) × total_sgst_amount, 2)
  → For CGST % and SGST % → extract percentage from summary label
      e.g. "Central Tax 9.00%" → cgst % = "9.00"
      e.g. "OUTPUT CGST @ 18%" → cgst % = "18"

  SINGLE GST % COLUMN CASE:
  → Some invoices have ONE "GST %" column instead of separate CGST and SGST
  → cgst % = sgst % = GST% value / 2
  → Example: GST % = 18 → cgst % = "9", sgst % = "9"
  → Calculate cgst amount and sgst amount proportionally per item

  COMBINED CASE (both inline and summary exist):
  → Prefer inline row values if clearly readable
  → Use summary box values only as fallback or validation

  NEVER return "NA" for cgst amount / sgst amount if ANY tax value
  exists anywhere in the document — even in footer or summary box.

  VALIDATION:
  cgst amount ≈ (taxable amt × cgst %) / 100
  sgst amount ≈ (taxable amt × sgst %) / 100

  Note: Sometimes CGST % + amount and SGST % + amount appear in same cell.
  Split them correctly and map to respective keys.

  ====================
  ROUND OFF / ADJUSTMENT ROW RULE
  ====================

  Rows labeled "R. OFF", "Round Off", "Less: R. OFF", "Less:", "Adjustment"
  that have no sr no and only an amount value:
  - Extract as ONE row only with all other fields as 'NA'
  - Do NOT repeat this row across multiple pages
  - Extract amount value as-is including negative sign
    Example: "(-)0.39" → amount = "-0.39"
    Example: "-0.40" → amount = "-0.40"

  ====================
  MATERIAL TABLE RULES (KEY POINTS — MUST FOLLOW)
  ====================
  - SR NO MUST BE A NUMBER:
    Sr no is always a small sequential integer (1, 2, 3...).
    If the value in the sr no column is a word or label such as:
    "Less:", "Less", "R. OFF", "Round Off", "Total", "Sub Total",
    "Note", "Adjustment" or any non-numeric text →
    that row is NOT a valid material row — SKIP it entirely.
    Do NOT extract it even if it has an amount value.

  - Do NOT extract item descriptions.
  - If "unit" is missing:
      - First try extracting unit from qty text.
      - If still unavailable → set "unit" to None
  - Sometimes all data is scattered — extract from any location accordingly.
  - If a key-value is not found in expected section →
    scan ENTIRE document including footer, margin, and summary boxes.
    Do NOT return "NA" if information exists anywhere on the page.
  - Treat all text inside () brackets as non-existent.
    Extract ONLY from text outside brackets.
  - Do NOT extract "%" symbol from any numeric value.
  - All extracted values must be of string data type.
  - If value for a key is not available → keep as 'NA'
  - If value is unreadable due to stamp or obstruction →
    keep as 'NA' and continue extracting other columns.
    Same rule applies for Discount column.
  -  A line without a sr no that appears directly below an item row
    is a continuation of that item's description — SKIP it entirely

  ====================
  HEADER-LESS TABLE HANDLING
  ====================

  If the material table has NO headers:

  - If vendor == "Mogli Labs (India) Pvt Ltd":
    Assume column order:
    sr no, hsn, qty, unit, unit price, taxable amt,
    cgst %, cgst amount, sgst %, sgst amount, amount

  - If vendor == "GHCL LTD":
    Assume column order:
    sr no, hsn, qty, unit, unit price, discount, taxable amt,
    cgst %, cgst amount, sgst %, sgst amount, amount

  ====================
  FINAL VALIDATION CHECKLIST (Run before returning output — MANDATORY)
  ====================

  1. PAN VALIDATION:
     - Is vendors pan no exactly 10 characters? Format: 5 letters + 4 digits + 1 letter?
     - Is buyers pan no exactly 10 characters? Format: 5 letters + 4 digits + 1 letter?
     - If either PAN field contains 15 characters → wrong, re-extract chars [2:12] only
     - If PAN not found in document → derive from respective GSTIN[2:12]
     - PAN may be on a different page — confirm you scanned all pages

  2. GSTIN VALIDATION:
     - Did you collect ALL GSTINs from entire document first (Step 1)?
     - Did you identify buyer GSTIN ONLY from the fixed buyer block (Step 2)?
     - Did you assign remaining unassigned GSTIN to vendor (Step 3)?
     - Is vendor gstin no exactly 15 characters?
     - Is buyers gstin no exactly 15 characters?
     - Does vendor gstin no[2:12] == vendors pan no?
       If not → re-read both from source carefully
     - Does buyers gstin no[2:12] == buyers pan no?
       If not → re-read both from source carefully
     - Re-read last 4 characters of each GSTIN one more time independently

  3. GSTIN CONFLICT CHECK:
     - Is vendor gstin no ≠ buyers gstin no?
     - If same value assigned to both → re-scan and correct immediately

  4. UNASSIGNED GSTIN CHECK:
     - Is there any 15-character alphanumeric string in the document
       NOT yet assigned to vendor or buyer?
     - If yes → determine ownership by location and assign accordingly

  5. CGST / SGST CHECK:
     - Did you check both inline rows AND summary box for each item?
     - For summary box → did you distribute PROPORTIONALLY per item
       based on each item's taxable amount — NOT equally split?
     - Is any cgst amount or sgst amount still "NA"?
       If yes → re-scan summary box and distribute proportionally to rows
     - Validate: cgst amount ≈ (taxable amt × cgst %) / 100

  6. DUPLICATE ROW CHECK:
     - Are there any duplicate material rows (same sr no + hsn + unit price)?
     - If yes → remove duplicates, keep only first occurrence
     - Is the adjustment/round-off row appearing more than once?
       If yes → keep only one instance

  7. DISCOUNT CHECK:
     - If invoice has a Disc.% or Discount column → ensure values are extracted
     - A value of 0.00 is valid — do not replace with NA

Final Result:
  
  Return result in the following exact JSON structure. 
  in result output: not need to extract given keys:"total taxable amount","total cgst amount","total sgst amount","total_igst_amount"

  RULES:
  - Every material row must contain ALL keys listed above.
  - Header fields (TaxInvoice, Buyer, BuyersPanNo, BuyersGstinNo,
    BuyersOrderNumber, Vendor, VendorPanNo, VendorGstinNo,
    InvoiceNo, InvoiceDate) must be repeated in every row.
  - Do NOT add any extra keys beyond what is listed above.
  - Do NOT rename any key.
  - Return ONLY the JSON only JSON not list — no extra text, explanation, or markdown.
"""


def image_to_text(image_path, prompt_template, output_dir=None):
    try:
        logging.info('Enter into image_to_text function.')
        # Apply output_dir FIRST
        if output_dir is not None:
            image_path = f'{output_dir}/{image_path}'

        # Extract extension AFTER final path
        ext = image_path.split('.')[-1].lower()
        logging.info(f'ext: {ext}')

        with open(image_path, "rb") as img_file:
            img_bytes = img_file.read()

        # Convert PNG → JPEG in memory
        if ext == "png":
            logging.info('converting png to jpeg')
            img = Image.open(BytesIO(img_bytes)).convert("RGB")

            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=95)
            buffer.seek(0)

            img_bytes = buffer.getvalue()
            ext = "jpeg"   # IMPORTANT: update ext

        # Base64 encode
        b64_string = base64.b64encode(img_bytes).decode('utf-8')

        messages = [
            (
                "user",
                [
                    {"type": "text", "text": prompt_template},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{ext};base64,{b64_string}"
                        },
                    },
                ],
            )
        ]

        last_error = None
        for _ in range(len(GOOGLE_API_KEYS)):
            logging.info(f'in loop:{_}')
            api_key = next(google_key_cycle)
            gemini_model = get_gemini_model(api_key)

            try:
                prompt = ChatPromptTemplate.from_messages(messages)
                chain = prompt | gemini_model | JsonOutputParser()
                result = chain.invoke({"b64_img": b64_string, "ext": ext})
                time.sleep(10)
                return result
            except Exception as e:
                error_msg = str(e).lower()
                last_error = e

                # only switch key if quota/rate-limit
                if "quota" in error_msg or "429" in error_msg or "rate limit" in error_msg:
                    logging.error(f"API key exhausted → switching key")
                    continue

                else:
                    raise e

        raise RuntimeError("All Google API keys exhausted") from last_error

    except (RuntimeError,Exception) as e:
        logging.error(str(e))
        return str(e)

def extract_page_no(filename):
    match = re.search(r'page_(\d+)_', filename, re.IGNORECASE)
    return int(match.group(1)) if match else 0

def normalize_date(date_str):
    try:
        if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            return date_str
        # Otherwise parse and convert
        date_obj = parser.parse(date_str, dayfirst=True)
        return date_obj.strftime("%Y-%m-%d")
    except Exception:
        return None

def pdf_to_text(pdf_path,file_name):
    upload_path = os.path.join('uploads',file_name)
    if pdf_path.endswith('pdf'):
        # Output directory for images
        os.path.basename(pdf_path)
        output_dir = os.path.basename(pdf_path).split('.')[0]
        os.makedirs(output_dir, exist_ok=True)

        # Convert PDF to list of images (one per page)
        images = convert_from_path(pdf_path, dpi=300,poppler_path=r'C:\Program Files\poppler-24.08.0\Library\bin')

        for i, image in enumerate(images):
            image_path = os.path.join(output_dir, f"page_{i}_{uuid.uuid4().hex}.jpeg")
            #image_path = os.path.join(output_dir, f"{output_dir}_{i + 1}.jpeg")
            image.save(image_path, 'JPEG')

        img_file_lst = sorted(os.listdir(output_dir), key=extract_page_no)
        if len(img_file_lst)!=0:
            page_wise_res_final = []
            for image_file in img_file_lst:
                logging.info(f'in image:{image_file}')
                result = image_to_text(image_file,prompt_template,output_dir)
                if isinstance(result,str):
                    shutil.rmtree(output_dir)
                    return {"success": False, "error": str(result)}
                page_wise_res_final.append(result)

            logging.info(f'Combine Image result in List:{page_wise_res_final}')
            print(':::::::::',page_wise_res_final)
            final_dict = page_wise_res_final[0]
            for i in range(1,len(page_wise_res_final)):
                if "material" in page_wise_res_final[i]:
                    final_dict['material'].extend(page_wise_res_final[i]['material'])
            
            logging.info(f"Final List of result: {final_dict}")
            print('final_dict:',final_dict)
            if not isinstance(final_dict,str):
                if final_dict['Tax Invoice'] == 'No':
                    # os.remove(upload_path)
                    # shutil.rmtree(output_dir)
                    email_sent = send_mail('Your Invoice is not submitted Please Connect with Accounts Team', upload_path)
                    return {"success": False, "error": "TAX INVOICE Not Found."}
                    # final_dict['Tax Invoice'] = "TAX INVOICE Not Found."
                
                elif final_dict["buyers gstin no"] != "24AAACG5609C1Z5":
                    print(f'gstin: {final_dict["buyers gstin no"]}')
                    # os.remove(upload_path)
                    # shutil.rmtree(output_dir)
                    email_sent = send_mail('Your Invoice is not submitted Please Connect with Accounts Team', upload_path)
                    return {"success": False, "error": f"Buyers GST number is not 24AAACG5609C1Z5"}
                    # final_dict["buyers gstin no"] = "Buyers GST number is not 24AAACG5609C1Z5"

                elif final_dict["invoice date"]:
                    date_obj = normalize_date(final_dict["invoice date"])

                    date = pd.to_datetime(date_obj).date()

                    final_dict["invoice date"] = str(date)

                    # date = pd.to_datetime(final_dict["invoice date"]).date()
                    print(date,datetime.now().date())
                    if date > datetime.now().date():
                        # os.remove(upload_path)
                        # shutil.rmtree(output_dir)
                        email_sent = send_mail('Your Invoice is not submitted Please Connect with Accounts Team', upload_path)
                        return {"success": False, "error": "Invoice date should not be in future."}
                    else:
                        if not final_dict['buyers order number'].startswith('45'):
                            final_dict['Status'] = 'HE'
                            final_dict['ErrorMessage'] = "Po number not readable"
                            print('HE',"Po number not readable")
                            print('::::::::::::::::::::::::::::::::::::::::::',final_dict['buyers order number'])
                        else:
                            if  len(final_dict['buyers order number']) != 10:
                                print('!=10')
                                # os.remove(upload_path)
                                # shutil.rmtree(output_dir)
                                # return {"success": False, "error": "Buyers number starts with 45 but length is not equal to 10."}
                                # final_dict['buyers order number'] = "Buyers number starts with 45 but length is not equal to 10."
                                final_dict['Status'] = 'HE'
                                final_dict['ErrorMessage'] = "Po number not readable"
                            else:
                                final_dict['Status'] = 'HS'
                                final_dict['ErrorMessage'] = "Invoice validation Successful."
                    
                        final_dict['vendor'] = final_dict['vendor'][:40]
                        # material related Process
                        for dct in final_dict['material']:
                            for key, val in dct.items():
                                if isinstance(val, str):
                                    dct[key] = val.replace(' ', '')

                        mat_df = pd.DataFrame(final_dict['material'])
                    
                        if 'sr no' in final_dict:
                            mat_df = mat_df[~mat_df['sr no'].isna()]
                            mat_df = mat_df[~mat_df.duplicated(subset=['sr no'])]

                        mat_df = mat_df.fillna("NA")
                        mat_df.replace('nan','Na',inplace=True)
                        mat_df.drop_duplicates(inplace=True)
                        
                        invoice_status = final_dict['Status']
                        print('invoice_status:',invoice_status)
                        final_dict['material'] = mat_df.to_dict(orient='records')
                        final_dict,invoice_no = json_format_conversion(final_dict)

                        final_df_1 = pd.DataFrame(final_dict['Material'])
                        print('final_df:',final_df_1)

                        if invoice_status == 'HE':
                            final_df_1['Status'] = 'IE'
                            final_df_1['ErrorMessage'] = "Po number not readable"
                        else:
                            final_df_1['Status'] = 'IS'
                            final_df_1['ErrorMessage'] = "Invoice validation Successful."


                        final_df_1['DataInputDate'] = str(datetime.now().date())
                        final_df_1['DataInputTime'] = str(datetime.now().time())


                        final_df_1 = final_df_1[final_df_1['Srno']!='NA']
                        print('final_df1:',final_df_1)

                        final_res_df = final_df_1.to_dict(orient='records')
                        
                        final_res_dict = {}
                        final_res_dict['Material'] = final_res_df

                        api_response_code,api_response_text = post_data_to_sap(sap_username,sap_password,final_res_dict)
                        if api_response_code == 201:
                            logging.info("Successfully send the data to SAP.")
                        else:

                            logging.error(f"Status Code: {api_response_code} and error is : {api_response_text}")

                        # send pdf binary format to api
                        pdf_base64_json = {}

                        file_name = os.path.split(pdf_path)[1]
                        pdf_base64_json['Filename'] = file_name
                        pdf_base64_json["Mimetype"] = "PDF"
                        pdf_base64_json['InvoiceNo'] = invoice_no
                        # Encode binary to Base64 string
                        # attach file with json 
                        with open(pdf_path, "rb") as f:
                            pdf_binary = f.read()
                            
                        pdf_base64_decode = base64.b64encode(pdf_binary).decode("utf-8")
                        pdf_base64_json['Content'] = pdf_base64_decode

                        response_code,response_text = post_pdf_data_to_sap(sap_username,sap_password,pdf_base64_json)
                        logging.info("Successfully send the data to SAP.")
                        if response_code == 201:
                            logging.info("Successfully send the data to SAP.")
                        else:
                            logging.error(f"Status Code: {response_code} and error is : {response_text}")

                        email_sent = send_mail(final_res_dict, upload_path)
                        shutil.rmtree(output_dir)
                        return final_res_dict
                else:
                    final_dict = 'Not able to capture data from image'
                    shutil.rmtree(output_dir)
                    return final_dict
                
            else:
                final_dict
        else:
            final_dict = 'data not found in pdf'
            shutil.rmtree(output_dir)
            return final_dict
        
    elif pdf_path.lower().endswith(('jpeg','png','jpg')):
        logging.info(f'Image formate so enter in elif: {pdf_path}')
        result = image_to_text(pdf_path,prompt_template)

        if isinstance(result,str):
            return {"success": False, "error": str(result)}
        
        if final_dict['Tax Invoice'] == 'No':
            # os.remove(upload_path)
            # shutil.rmtree(output_dir)
            email_sent = send_mail('Your Invoice is not submitted Please Connect with Accounts Team', upload_path)
            return {"success": False, "error": "TAX INVOICE Not Found."}
            # final_dict['Tax Invoice'] = "TAX INVOICE Not Found."
                
        elif final_dict["buyers gstin no"] != "24AAACG5609C1Z5":
            print(f'gstin: {final_dict["buyers gstin no"]}')
            # os.remove(upload_path)
            # shutil.rmtree(output_dir)
            email_sent = send_mail('Your Invoice is not submitted Please Connect with Accounts Team', upload_path)
            return {"success": False, "error": f"Buyers GST number is not 24AAACG5609C1Z5"}
            # final_dict["buyers gstin no"] = "Buyers GST number is not 24AAACG5609C1Z5"

        elif final_dict["invoice date"]:
            date_obj = normalize_date(final_dict["invoice date"])

            date = pd.to_datetime(date_obj).date()

            final_dict["invoice date"] = str(date)

            # date = pd.to_datetime(final_dict["invoice date"]).date()
            print(date,datetime.now().date())
            if date > datetime.now().date():
                # os.remove(upload_path)
                # shutil.rmtree(output_dir)
                email_sent = send_mail('Your Invoice is not submitted Please Connect with Accounts Team', upload_path)
                return {"success": False, "error": "Invoice date should not be in future."}
                # final_dict["invoice date"] = "Invoice date should not be in future."
            else:
                if not result['buyers order number'].startswith('45'):
                    # os.remove(upload_path)
                    # return 'Buyers number is not start with 45'
                    # return {"success": False, "error": "Buyers number is not start with 45."}
                    # result['buyers order number'] = "Buyers number is not start with 45."
                    result['Status'] = 'HE'
                    result['ErrorMessage'] = "Po number not readable"
                else:
                    if  len(result['buyers order number']) != 10:
                    # os.remove(upload_path)
                    # return 'Buyers number starts with 45 but length is not equal to 10'
                    # return {"success": False, "error": "Buyers number starts with 45 but length is not equal to 10."}
                    # result['buyers order number']= "Buyers number starts with 45 but length is not equal to 10."
                        result['Status'] = 'HE'
                        result['ErrorMessage'] = "Po number not readable"
                    else:
                        result['Status'] = 'HS'
                        result['ErrorMessage'] = "Invoice Validation Successfull."
                result['vendor'] = result['vendor'][:40]
                invoice_status = result['Status']
                # else:
                result,invoice_no = json_format_conversion(result)

                final_df = pd.DataFrame(result['Material'])
                if invoice_status == 'HE':
                    final_df['Status'] = 'IE'
                    final_df['Message'] = "Po number not readable"

                elif invoice_status == 'HS':
                    final_df['Status'] = 'IS'
                    final_df['Error Message'] = "Invoice validation Successful."

                final_df['DataInputDate'] = str(datetime.now().date())
                final_df['DataInputTime'] = str(datetime.now().time())
                
                final_result = final_df.to_dict(orient='records')

                final_res_dict = {}
                final_res_dict['Material'] = final_result

                api_response_code,api_response_text = post_data_to_sap(sap_username,sap_password,final_res_dict)
                if api_response_code == 201:
                    logging.info("Successfully send the data to SAP.")
                else:
                    logging.error(f"Status Code: {api_response_code} and error is : {api_response_text}")

                # send image binary format to api
                img_base64_json = {}

                file_name = os.path.split(pdf_path)[1]
                img_base64_json['Filename'] = file_name
                img_base64_json["Mimetype"] = "PDF"
                img_base64_json['InvoiceNo'] = invoice_no

                    # attach file with json 
                with open(pdf_path, "rb") as f:
                    image_binary = f.read()

                # Encode binary to Base64 string
                img_base64_decode = base64.b64encode(image_binary).decode("utf-8")
                img_base64_json['pdf_base64'] = img_base64_decode
                
                response_code,response_text = post_pdf_data_to_sap(sap_username,sap_password,img_base64_json)
                if response_code == 201:

                    logging.info("Successfully send the data to SAP.")
                else:
                    logging.error(f"Status Code: {response_code} and error is : {response_text}")
                send_mail(final_res_dict,upload_path)
                return final_res_dict
        
        else:
            return "Please provide valid format: pdf, png and jpg"
            # return {"success": False, "error": "Please provide valid format: pdf, png and jpg."}


def rename_keys(data, mapping):
    return {mapping.get(k, k): v for k, v in data.items()}


def post_data_to_sap(username,password,json_data):
    json_data = json.dumps(json_data,indent=4,ensure_ascii=False)

    print('json:',json_data)

    header= {
        "Content-Type": "application/json",
        "X-Requested-With":'X',
        "Accept": "application/json"
        }

    req_response = requests.post(url="https://S4HANASADEV.ghclindia.net:44300/sap/opu/odata/sap/YOCR_INV_SRV/OCRHEADSet",
                            headers=header,
                            data=json_data,
                            auth=HTTPBasicAuth(username, password))

    if req_response.status_code == 201:
        return req_response.status_code,req_response.text
    else:
        return req_response.status_code,req_response.text
    

def post_pdf_data_to_sap(username,password,json_data):
    json_data = json.dumps(json_data,indent=4,ensure_ascii=False)


    header= {
        "Content-Type": "application/json",
        "X-Requested-With":'X',
        "Accept": "application/json"
        }

    req_response = requests.post(url="https://S4HANASADEV.ghclindia.net:44300/sap/opu/odata/SAP/YOCR_IMG_SRV/YINVMEDSet",
                            headers=header,
                            data=json_data,
                            auth=HTTPBasicAuth(username, password))

    if req_response.status_code == 201:
        return req_response.status_code,req_response.text
    else:
        return req_response.status_code,req_response.text

def json_format_conversion(json_data):
    InvoiceNo = json_data['invoice number']
    
    invo_dict = {}

    key_need_to_replace = {"invoice number":"InvoiceNo",
                        "Tax Invoice": "TaxInvoice",
                        "buyer": "Buyer",
                        "buyers pan no":"BuyersPanNo",
                        "buyers gstin no":"BuyersGstinNo",
                        "buyers order number":"BuyersOrderNumber",
                        "vendor": "Vendor",
                        "vendors pan no": "VendorPanNo",
                        "vendor gstin no": "VendorGstinNo",
                        "invoice date":"InvoiceDate",
                        "material":"Material"
                        }

    material_key_to_replace = {"sr no":"Srno","hsn":"Hsn","qty":"Qty","unit":"Unit",
                           "unit price":"UnitPrice","discount":"Discount",
                           "taxable amt":"TaxableAmt","cgst %":"Cgst",
                           "cgst amount":"CgstAmount","sgst %":"Sgst",
                           "sgst amount":"SgstAmount","amount":"Amount"}
    
    #['InvoiceNo', 'Hsn', 'Qty', 'Unit', 'UnitPrice', 'Discount', 'TaxableAmt', 'Cgst', 'CgstAmount', 'Sgst', 'SgstAmount', 'Amount', 'TaxInvoice', 'Buyer', 'BuyersPanNo', 'BuyersGstinNo', 'BuyersOrderNumber', 'Vendor', 'VendorPanNo', 'VendorGstinNo', 'InvoiceDate']

    new_json = rename_keys(json_data, key_need_to_replace)
    material = []
    if 'Material' in new_json:
        for i,mat_json in enumerate(new_json['Material']):
            mat_j = rename_keys(mat_json, material_key_to_replace)
            material.append(mat_j)


    new_json['Material'] = material

    data = json.dumps(new_json,indent=4)

    data  = json.loads(data)

    mat_dict = data["Material"]

    del data['Material']

    material_lst = []
    for dict in mat_dict:
        final_dict = dict | data
        material_lst.append(final_dict)


    invo_dict['Material'] = material_lst
    final_json_data = json.dumps(invo_dict,indent=4)
    final_json_data = json.loads(final_json_data)
    logging.info(f'final_json_data:{final_json_data}')
    return final_json_data, InvoiceNo

@app.route('/upload', methods=['POST'])
def upload_file():

    if 'pdfFile' not in request.files:
        return jsonify({"success": False, "error": "No file part in request."})

    file = request.files['pdfFile']
    if file.filename == '':
        return jsonify({"success": False, "error": "No file selected."})

    file_path = os.path.join('uploads',file.filename)
    filename = secure_filename(file.filename)
    file.save(file_path)

    # Encode PDF to base64 for preview
    with open(file_path, "rb") as f:
        encoded_pdf = base64.b64encode(f.read()).decode('utf-8')

    return jsonify({
        "success": True,
        "message": "File uploaded successfully.",
        "filename": filename,  # Needed later in /extract
        "original_filename": file.filename,
        "pdf_data": encoded_pdf  # For PDF viewer
    })

@app.route('/clear-uploads', methods=['POST'])
def clear_uploads():
    try:
        uploads_folder = 'uploads'
        if os.path.exists(uploads_folder):
            print(f"Uploads folder exists: {os.path.exists(uploads_folder)}")
            for filename in os.listdir(uploads_folder):
                file_path = os.path.join(uploads_folder, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print("All files in 'uploads' folder have been deleted.")
        else:
            print("Uploads folder does not exist.")
        return jsonify({'success': True, 'message': 'Uploads folder cleared successfully'})
    
    except Exception as e:
        # Print detailed error and traceback for debugging
        print("Error occurred while clearing uploads folder:")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Uploads folder nottttt cleared successfully',
            'traceback': traceback.format_exc()
        })
    
@app.route('/extract', methods=['POST'])
def extract_data():
    try:
        data = request.get_json()
        filename = data.get('filename')
        original_filename = data.get('original_filename')

        if not filename:
            return jsonify({"success": False, "error": "Filename is missing."}), 400

        upload_path = os.path.join(UPLOAD_FOLDER, original_filename)
        if not os.path.exists(upload_path):
            return jsonify({"success": False, "error": "Uploaded file not found."}), 400


        # Extract text from PDF
        result = pdf_to_text(upload_path, original_filename)
        print('res::',result)
        
        result = str(result)

        return jsonify({
                "success": True,
                "message": "Text extracted successfully.",
                "data": result
            })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": "Internal server error: " + str(e)}), 500


def send_mail(data, file_path):
    try:
        message = MIMEMultipart()
        sender_email = 'mayurnandanwar@ghcl.co.in'
        message["From"] = sender_email
        receiver_email = 'mayurnandanwar@ghcl.co.in'
        message["To"] = receiver_email
        message["Subject"] = "Information From Invoice"

        json_string = json.dumps(data, indent=4)
        body = "Extracted_Information: \n"+str(json_string)
        message.attach(MIMEText(body, "plain"))

        with open(file_path, "rb") as file:
            attachment = MIMEApplication(file.read(), Name=os.path.split(file_path)[1])
            message.attach(attachment)

        password = os.getenv('email_password')
        s = smtplib.SMTP('smtp.gmail.com', 587)
        s.starttls()
        s.login(sender_email, password)
        s.sendmail(sender_email, receiver_email, message.as_string())
        s.quit()
        return True
    except Exception as e:
        return False
    

@app.route('/download/<filename>')
def download_file(filename):
    try:
        return send_file(os.path.join('templates', filename), as_attachment=True)
    except Exception as e:
        return jsonify({"error": "File not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)






