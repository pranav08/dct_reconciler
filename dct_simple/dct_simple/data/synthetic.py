"""
data/synthetic.py — 30 records across 3 messy formats.

Designed to exercise every concept:
  - Vendor JSON: PULSE alias, mixed units, 1 unit conversion needed (Fahrenheit)
  - Site CRF CSV: human-entered casual phrasing, 1 out-of-range
  - HL7: clean LOINC codes, 1 duplicate

Two subjects: SUBJ-001 (normal-ish), SUBJ-002 (one HR spike at 285)
"""

VENDOR_JSON = """
{
  "batch_id": "wear-2025-01-08-A",
  "records": [
    {"subj":"SUBJ-001","metric":"PULSE","val":72,"u":"bpm","ts":"2025-01-08T08:00:00"},
    {"subj":"SUBJ-001","metric":"PULSE","val":75,"u":"bpm","ts":"2025-01-08T09:00:00"},
    {"subj":"SUBJ-001","metric":"BodyTemp","val":98.4,"u":"deg F","ts":"2025-01-08T08:00:00"},
    {"subj":"SUBJ-001","metric":"SpO2","val":97,"u":"%","ts":"2025-01-08T08:00:00"},
    {"subj":"SUBJ-002","metric":"HR","val":68,"u":"/min","ts":"2025-01-08T08:00:00"},
    {"subj":"SUBJ-002","metric":"HR","val":285,"u":"/min","ts":"2025-01-08T08:30:00"},
    {"subj":"SUBJ-002","metric":"HR","val":74,"u":"/min","ts":"2025-01-08T09:00:00"},
    {"subj":"SUBJ-002","metric":"BodyTemp","val":36.9,"u":"Cel","ts":"2025-01-08T08:00:00"},
    {"subj":"SUBJ-002","metric":"SpO2","val":98,"u":"%","ts":"2025-01-08T08:00:00"},
    {"subj":"SUBJ-001","metric":"WeirdCode","val":42,"u":"x","ts":"2025-01-08T10:00:00"}
  ]
}
""".strip()


SITE_CRF_CSV = """subject_id,visit,parameter,value,unit,datetime
SUBJ-001,V1,Systolic BP,120,mmHg,2025-01-08 08:00:00
SUBJ-001,V1,Diastolic BP,78,mmHg,2025-01-08 08:00:00
SUBJ-001,V1,Heart Rate,72,bpm,2025-01-08 08:00:00
SUBJ-001,V1,Body Temp,36.7,Cel,2025-01-08 08:00:00
SUBJ-002,V1,Systolic BP,210,mmHg,2025-01-08 08:00:00
SUBJ-002,V1,Diastolic BP,95,mmHg,2025-01-08 08:00:00
SUBJ-002,V1,Heart Rate,68,bpm,2025-01-08 08:00:00
SUBJ-002,V1,Body Temp,37.2,Cel,2025-01-08 08:00:00
SUBJ-001,V2,Systolic BP,118,mmHg,2025-01-15 08:00:00
SUBJ-001,V2,Heart Rate,70,bpm,2025-01-15 08:00:00
"""


HL7_FRAGMENT = """MSH|^~\\&|DEVICE-001|SITE-001|PIPELINE|TRIAL|20250108080000||ORU
PID|||SUBJ-001
OBX|1|NM|8867-4^Heart Rate||71|/min|||F|||20250108080000
OBX|2|NM|8480-6^Systolic BP||119|mm[Hg]|||F|||20250108080000
OBX|3|NM|8462-4^Diastolic BP||77|mm[Hg]|||F|||20250108080000
OBX|4|NM|8310-5^Body Temperature||36.8|Cel|||F|||20250108080000
OBX|5|NM|9279-1^Respiratory Rate||14|/min|||F|||20250108080000
PID|||SUBJ-002
OBX|6|NM|8867-4^Heart Rate||69|/min|||F|||20250108080000
OBX|7|NM|8480-6^Systolic BP||211|mm[Hg]|||F|||20250108080000
OBX|8|NM|8462-4^Diastolic BP||96|mm[Hg]|||F|||20250108080000
OBX|9|NM|8867-4^Heart Rate||69|/min|||F|||20250108080000
OBX|10|NM|9279-1^Respiratory Rate||16|/min|||F|||20250108080000
"""


__all__ = ["VENDOR_JSON", "SITE_CRF_CSV", "HL7_FRAGMENT"]
