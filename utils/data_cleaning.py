import re

import pandas as pd


DATE_PARSE_MODES = {
    "auto",
    "dayfirst",
    "monthfirst",
    "ymd",
    "dmy_dot",
    "dmy_slash",
    "mdy_slash",
    "yyyymmdd",
    "excel_serial",
}

DATE_GRAINS = {"day", "month", "year"}
BAD_VALUE_POLICIES = {"zero", "drop", "error"}
BAD_DATE_POLICIES = {"drop", "error", "keep"}


_THAI_MONTH_NAMES = [
    ("ม.ค.", "มกราคม"), ("ก.พ.", "กุมภาพันธ์"), ("มี.ค.", "มีนาคม"), ("เม.ย.", "เมษายน"),
    ("พ.ค.", "พฤษภาคม"), ("มิ.ย.", "มิถุนายน"), ("ก.ค.", "กรกฎาคม"), ("ส.ค.", "สิงหาคม"),
    ("ก.ย.", "กันยายน"), ("ต.ค.", "ตุลาคม"), ("พ.ย.", "พฤศจิกายน"), ("ธ.ค.", "ธันวาคม"),
]
# key ตัดจุดออก จึงรับได้ทั้ง "ม.ค." "ม.ค" "มค" และชื่อเต็ม
THAI_MONTHS = {
    name.replace(".", ""): number
    for number, names in enumerate(_THAI_MONTH_NAMES, start=1)
    for name in names
}
# เดือนไทย + ปี 2 หรือ 4 หลัก เช่น "ม.ค.69", "มี.ค. 2569", "มกราคม-2569"
THAI_MONTH_YEAR_PATTERN = r"([ก-๎.]+?)\s*[-/]?\s*(\d{4}|\d{2})"


def blank_mask(series):
    text = series.astype("string").str.strip()
    return (
        series.isna()
        | text.isna()
        | text.eq("")
        | text.str.lower().isin(["nan", "none", "null", "na", "n/a", "-"])
    )


def clean_numeric_series(series):
    """Convert accounting-style values to numbers while preserving invalid cells as NaN."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    text = series.astype("string").str.strip()
    text = text.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    text = text.str.replace(r"[,\s$฿%]", "", regex=True)
    text = text.str.replace("\u00a0", "", regex=False)
    # string dtype -> to_numeric ได้ Float64 (nullable) ซึ่ง np.select ใน engine รับ mask ที่มี pd.NA ไม่ได้
    return pd.to_numeric(text, errors="coerce").astype("float64")


def numeric_bad_mask(source, numeric):
    return (~blank_mask(source)) & numeric.isna()


def apply_value_transform(series, multiplier=1.0, add=0.0):
    return (series * float(multiplier)) + float(add)


def parse_date_series(series, mode="auto"):
    """Parse dates with Excel/Thai finance friendly defaults."""
    mode = mode if mode in DATE_PARSE_MODES else "auto"

    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")

    if mode == "excel_serial" or (mode == "auto" and _looks_like_excel_serial(series)):
        numeric = pd.to_numeric(series, errors="coerce")
        return pd.to_datetime(numeric, unit="D", origin="1899-12-30", errors="coerce")

    text = series.astype("string").str.strip()

    # เดือนไทยไม่มี mode ไหน parse ได้ จึงเช็คก่อนเสมอไม่ว่าเลือก mode อะไร
    if _pattern_ratio(text, THAI_MONTH_YEAR_PATTERN) >= 0.6:
        return _parse_thai_month_year(text)

    format_by_mode = {
        "ymd": "%Y-%m-%d",
        "dmy_dot": "%d.%m.%Y",
        "dmy_slash": "%d/%m/%Y",
        "mdy_slash": "%m/%d/%Y",
        "yyyymmdd": "%Y%m%d",
    }
    if mode in format_by_mode:
        return pd.to_datetime(text, format=format_by_mode[mode], errors="coerce")

    if mode == "dayfirst":
        return pd.to_datetime(text, errors="coerce", dayfirst=True)
    if mode == "monthfirst":
        return pd.to_datetime(text, errors="coerce", dayfirst=False)

    for pattern, fmt in [
        (r"\d{1,2}\.\d{1,2}\.\d{4}", "%d.%m.%Y"),
        (r"\d{4}-\d{1,2}-\d{1,2}", "%Y-%m-%d"),
        (r"\d{4}/\d{1,2}/\d{1,2}", "%Y/%m/%d"),
        (r"\d{4}-\d{1,2}", "%Y-%m"),
        (r"\d{4}/\d{1,2}", "%Y/%m"),
        (r"\d{8}", "%Y%m%d"),
        (r"\d{1,2}[A-Za-z]{3}\d{4}", "%d%b%Y"),
        (r"\d{1,2}/\d{1,2}/\d{4}", "%d/%m/%Y"),
    ]:
        if _pattern_ratio(text, pattern) >= 0.6:
            return pd.to_datetime(text, format=fmt, errors="coerce")

    default = pd.to_datetime(text, errors="coerce", dayfirst=False)
    dayfirst = pd.to_datetime(text, errors="coerce", dayfirst=True)
    return dayfirst if dayfirst.notna().sum() > default.notna().sum() else default


def _parse_thai_month_year(text):
    """'ม.ค.69' / 'มี.ค. 2569' / 'มกราคม 2569' -> วันที่ 1 ของเดือน (ค.ศ.), ที่ไม่เข้ารูปแบบเป็น NaT"""
    parts = text.str.extract("^" + THAI_MONTH_YEAR_PATTERN + "$")
    # float64 ธรรมดา (ไม่ใช่ nullable) เพื่อให้แถวที่ไม่เข้ารูปแบบกลายเป็น NaT แทนที่จะ throw
    month = pd.to_numeric(parts[0].str.replace(".", "", regex=False).map(THAI_MONTHS), errors="coerce").astype("float64")
    year = pd.to_numeric(parts[1], errors="coerce").astype("float64")
    # ponytail: ปี 2 หลักที่มากับเดือนไทยถือเป็น พ.ศ. 25xx (69 -> 2569 -> 2026)
    # ถ้าเจอไฟล์ที่ใช้ ค.ศ. 2 หลักกับเดือนไทย ค่อยเพิ่ม date_parse_mode แยก
    year = year.where(year >= 100, year + 2500)
    year = year.where(year < 2400, year - 543)
    return pd.to_datetime(pd.DataFrame({"year": year, "month": month, "day": 1}), errors="coerce")


def apply_date_grain(series, grain="month"):
    grain = grain if grain in DATE_GRAINS else "month"
    dates = pd.to_datetime(series, errors="coerce")
    if grain == "year":
        return dates.dt.to_period("Y").dt.to_timestamp()
    if grain == "day":
        return dates.dt.floor("D")
    return dates.dt.to_period("M").dt.to_timestamp()


def format_date_label(value, grain="month"):
    grain = grain if grain in DATE_GRAINS else "month"
    try:
        timestamp = pd.Timestamp(value)
    except Exception:
        return str(value)
    if pd.isna(timestamp):
        return ""
    if grain == "year":
        return timestamp.strftime("%Y")
    if grain == "day":
        return timestamp.strftime("%Y-%m-%d")
    return timestamp.strftime("%Y-%m")


def _pattern_ratio(text, pattern):
    sample = text.dropna().head(100)
    if sample.empty:
        return 0
    return sample.str.fullmatch(pattern, flags=re.IGNORECASE).sum() / len(sample)


def _looks_like_excel_serial(series):
    numeric = pd.to_numeric(series, errors="coerce").dropna().head(100)
    if numeric.empty:
        return False
    return numeric.between(20000, 80000).mean() >= 0.8
