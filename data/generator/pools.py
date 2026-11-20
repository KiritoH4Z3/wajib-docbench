"""Fixed vocabularies the generator draws from, English and Arabic side by side.

Labels follow the research notes on UAE licences, FTA tax invoices, FTA registration
certificates and quotations (unconfirmed against official specimens, see README). Names of
companies and people are invented. Arabic strings carry no diacritics: the PDF text layer
check rebuilds Arabic from glyph positions, and combining marks would make that ambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass

Pair = tuple[str, str]  # (English, Arabic)

LABELS: dict[str, Pair] = {
    # Trade licence
    "licence_title": ("Trade Licence", "رخصة تجارية"),
    "licence_no": ("Licence No.", "رقم الرخصة"),
    "trade_name": ("Trade Name", "الاسم التجاري"),
    "trade_name_en": ("Trade Name in English", "الاسم التجاري بالإنجليزية"),
    "trade_name_ar": ("Trade Name in Arabic", "الاسم التجاري بالعربية"),
    "legal_form": ("Legal Form", "الشكل القانوني"),
    "licence_type": ("Licence Type", "نوع الرخصة"),
    "issue_date": ("Issue Date", "تاريخ الإصدار"),
    "expiry_date": ("Expiry Date", "تاريخ الانتهاء"),
    "cr_no": ("Commercial Register No.", "رقم السجل التجاري"),
    "registration_no": ("Registration No.", "رقم التسجيل"),
    "address": ("Address", "العنوان"),
    "manager": ("Manager", "المدير"),
    "owners": ("Owners and Partners", "الملاك والشركاء"),
    "owner_name": ("Name", "الاسم"),
    "nationality": ("Nationality", "الجنسية"),
    "share": ("Share", "الحصة"),
    "activities": ("Licensed Activities", "الأنشطة المرخصة"),
    "activity_code": ("Code", "الرمز"),
    "activity": ("Activity", "النشاط"),
    "activity_ar": ("Activity in Arabic", "النشاط بالعربية"),
    # Tax invoice and quotation
    "invoice_title": ("Tax Invoice", "فاتورة ضريبية"),
    "invoice_no": ("Invoice No.", "رقم الفاتورة"),
    "invoice_date": ("Invoice Date", "تاريخ الفاتورة"),
    "supply_date": ("Date of Supply", "تاريخ التوريد"),
    "due_date": ("Due Date", "تاريخ الاستحقاق"),
    "supplier_trn": ("Supplier TRN", "الرقم الضريبي للمورد"),
    "bill_to": ("Bill To", "فاتورة إلى"),
    "customer_name": ("Customer Name", "اسم العميل"),
    "customer_name_en": ("Customer Name in English", "اسم العميل بالإنجليزية"),
    "customer_name_ar": ("Customer Name in Arabic", "اسم العميل بالعربية"),
    "customer_trn": ("Customer TRN", "الرقم الضريبي للعميل"),
    "customer_address": ("Customer Address", "عنوان العميل"),
    "line_no": ("No.", "م"),
    "description": ("Description", "الوصف"),
    "qty": ("Qty", "الكمية"),
    "unit_price": ("Unit Price", "سعر الوحدة"),
    "vat_rate": ("VAT", "الضريبة"),
    "line_amount": ("Amount", "المبلغ"),
    "subtotal": ("Subtotal excl. VAT", "الإجمالي قبل الضريبة"),
    "vat_amount": ("VAT", "ضريبة القيمة المضافة"),
    "total": ("Total incl. VAT", "الإجمالي شامل الضريبة"),
    "amount_words": ("Amount in words", "المبلغ بالحروف"),
    "payment_terms": ("Payment Terms", "شروط الدفع"),
    "iban": ("IBAN", "رقم الحساب المصرفي الدولي"),
    "tel": ("Tel", "هاتف"),
    "po_box": ("P.O. Box", "ص.ب"),
    "quote_title": ("Quotation", "عرض سعر"),
    "quote_no": ("Quotation No.", "رقم عرض السعر"),
    "quote_date": ("Date", "التاريخ"),
    "valid_until": ("Valid Until", "صالح حتى"),
    "client": ("Client", "العميل"),
    "client_en": ("Client in English", "العميل بالإنجليزية"),
    "client_ar": ("Client in Arabic", "العميل بالعربية"),
    "attention": ("Attention", "عناية"),
    "grand_total": ("Grand Total", "الإجمالي"),
    "delivery": ("Delivery", "التسليم"),
    "prepared_by": ("Prepared by", "أعده"),
    # Tax registration certificate
    "fta": ("Federal Tax Authority", "الهيئة الاتحادية للضرائب"),
    "vat_cert_title": (
        "Value Added Tax Registration Certificate",
        "شهادة التسجيل في ضريبة القيمة المضافة",
    ),
    "ct_cert_title": ("Corporate Tax Registration Certificate", "شهادة التسجيل في ضريبة الشركات"),
    "cert_trn": ("Tax Registration Number", "رقم التسجيل الضريبي"),
    "legal_name": ("Legal Name", "الاسم القانوني"),
    "legal_name_en": ("Legal Name in English", "الاسم القانوني بالإنجليزية"),
    "legal_name_ar": ("Legal Name in Arabic", "الاسم القانوني بالعربية"),
    "registered_address": ("Registered Address", "العنوان المسجل"),
    "effective_date": ("Effective Registration Date", "تاريخ سريان التسجيل"),
    "cert_issue_date": ("Date of Issue", "تاريخ إصدار الشهادة"),
    "tax_period": ("Tax Period", "الفترة الضريبية"),
    "financial_year": ("Financial Year", "السنة المالية"),
    # Shared
    "notes": ("Notes", "ملاحظات"),
}

FOOTER = "Synthetic sample for benchmarking / نموذج اصطناعي للاختبار"
VERIFY_DOMAIN = "verify.docbench.example"

EMIRATES: dict[str, Pair] = {
    "dxb": ("Dubai", "دبي"),
    "auh": ("Abu Dhabi", "أبوظبي"),
    "shj": ("Sharjah", "الشارقة"),
    "ajm": ("Ajman", "عجمان"),
    "rak": ("Ras Al Khaimah", "رأس الخيمة"),
}

AREAS: dict[str, list[Pair]] = {
    "dxb": [
        ("Al Qusais Industrial Area 2", "منطقة القصيص الصناعية 2"),
        ("Business Bay", "الخليج التجاري"),
        ("Al Barsha 1", "البرشاء 1"),
        ("Deira, Al Rigga", "ديرة، الرقة"),
    ],
    "auh": [
        ("Mussafah Industrial Area", "منطقة مصفح الصناعية"),
        ("Al Khalidiyah", "الخالدية"),
        ("Al Nahyan", "آل نهيان"),
    ],
    "shj": [
        ("Industrial Area 6", "المنطقة الصناعية 6"),
        ("Al Majaz 2", "المجاز 2"),
        ("Al Nahda", "النهدة"),
    ],
    "ajm": [("Al Jurf Industrial 1", "الجرف الصناعية 1"), ("Al Nuaimiya", "النعيمية")],
    "rak": [("Al Nakheel", "النخيل"), ("Al Hamra", "الحمراء")],
}


@dataclass(frozen=True)
class Authority:
    en: str
    ar: str
    emirate: str
    free_zone: bool
    number_format: str  # "digits6", "cn7", "dmcc6", "digits5", "digits7"
    # Free-zone areas override the emirate's areas for addresses.
    zone_area: Pair | None = None


AUTHORITIES: list[Authority] = [
    Authority(
        "Dubai Department of Economy and Tourism",
        "دائرة الاقتصاد والسياحة في دبي",
        "dxb",
        False,
        "digits6",
    ),
    Authority(
        "Abu Dhabi Department of Economic Development",
        "دائرة التنمية الاقتصادية في أبوظبي",
        "auh",
        False,
        "cn7",
    ),
    Authority(
        "Sharjah Economic Development Department",
        "دائرة التنمية الاقتصادية بالشارقة",
        "shj",
        False,
        "digits6",
    ),
    Authority(
        "Ajman Department of Economic Development",
        "دائرة التنمية الاقتصادية بعجمان",
        "ajm",
        False,
        "digits5",
    ),
    Authority(
        "Ras Al Khaimah Department of Economic Development",
        "دائرة التنمية الاقتصادية برأس الخيمة",
        "rak",
        False,
        "digits5",
    ),
    Authority(
        "Dubai Multi Commodities Centre",
        "مركز دبي للسلع المتعددة",
        "dxb",
        True,
        "dmcc6",
        ("Jumeirah Lakes Towers", "أبراج بحيرات جميرا"),
    ),
    Authority(
        "International Free Zone Authority",
        "سلطة المنطقة الحرة الدولية",
        "dxb",
        True,
        "digits5",
        ("Dubai Silicon Oasis", "واحة دبي للسيليكون"),
    ),
    Authority(
        "Ras Al Khaimah Economic Zone",
        "منطقة رأس الخيمة الاقتصادية",
        "rak",
        True,
        "digits7",
        ("Al Hamra Industrial Zone", "منطقة الحمراء الصناعية"),
    ),
    Authority(
        "Sharjah Media City",
        "مدينة الشارقة للإعلام",
        "shj",
        True,
        "digits7",
        ("Sharjah Media City", "مدينة الشارقة للإعلام"),
    ),
]

MAINLAND_FORMS = ("LLC", "SOLE_ESTABLISHMENT", "CIVIL_COMPANY", "BRANCH")
FREE_ZONE_FORMS = ("FZE", "FZ_LLC", "FZCO")

# Printed form of each legal form, chosen from extract.schemas.ENUM_ALIASES so the
# grounding check can find it. The Arabic LLC form adds the word "شركة" in front of an alias.
LEGAL_FORM_PRINTED: dict[str, Pair] = {
    "LLC": ("Limited Liability Company", "شركة ذات مسؤولية محدودة"),
    "SOLE_ESTABLISHMENT": ("Sole Establishment", "مؤسسة فردية"),
    "CIVIL_COMPANY": ("Civil Company", "شركة مدنية"),
    "BRANCH": ("Branch of a Foreign Company", "فرع شركة أجنبية"),
    "FZE": ("Free Zone Establishment", "مؤسسة منطقة حرة"),
    "FZ_LLC": ("Free Zone Limited Liability Company", "شركة منطقة حرة ذات مسؤولية محدودة"),
    "FZCO": ("Free Zone Company", "شركة منطقة حرة"),
}

LICENCE_TYPES: list[Pair] = [
    ("Commercial", "تجارية"),
    ("Professional", "مهنية"),
    ("Industrial", "صناعية"),
]

COMPANY_STEMS: list[Pair] = [
    ("Al Noor", "النور"),
    ("Al Waha", "الواحة"),
    ("Bait Al Khaleej", "بيت الخليج"),
    ("Al Reem", "الريم"),
    ("Najm Al Sahra", "نجم الصحراء"),
    ("Al Fajr", "الفجر"),
    ("Al Manara", "المنارة"),
    ("Dar Al Yasmin", "دار الياسمين"),
    ("Sahil Al Dhahab", "ساحل الذهب"),
    ("Al Marjan", "المرجان"),
    ("Al Thuraya", "الثريا"),
    ("Al Safwa", "الصفوة"),
    ("Al Dana", "الدانة"),
    ("Barq Al Shamal", "برق الشمال"),
    ("Wadi Al Sidr", "وادي السدر"),
    ("Sama Al Khaleej", "سما الخليج"),
    ("Al Mizan", "الميزان"),
    ("Durrat Al Shati", "درة الشاطئ"),
    ("Al Nakhla", "النخلة"),
    ("Al Hadaf", "الهدف"),
]


@dataclass(frozen=True)
class Activity:
    code: str
    en: str
    ar: str


ACTIVITIES: dict[str, Activity] = {
    a.code: a
    for a in [
        Activity("4690.01", "General Trading", "التجارة العامة"),
        Activity("4663.01", "Building Materials Trading", "تجارة مواد البناء"),
        Activity("6202.01", "IT Consultancy", "استشارات تقنية المعلومات"),
        Activity("8121.01", "Building Cleaning Services", "خدمات تنظيف المباني"),
        Activity("5610.01", "Restaurant", "مطعم"),
        Activity("4651.01", "Computer Software Trading", "تجارة برمجيات الحاسوب"),
        Activity("4321.01", "Electrical Fittings Contracting", "مقاولات التركيبات الكهربائية"),
        Activity("4630.01", "Foodstuff Trading", "تجارة المواد الغذائية"),
        Activity("4530.01", "Auto Spare Parts Trading", "تجارة قطع غيار السيارات"),
        Activity("8230.01", "Event Management", "تنظيم الفعاليات"),
        Activity("7410.01", "Interior Design", "التصميم الداخلي"),
        Activity("5229.01", "Cargo Services", "خدمات الشحن"),
        Activity("7020.01", "Management Consultancy", "الاستشارات الإدارية"),
        Activity("1811.01", "Commercial Printing", "الطباعة التجارية"),
        Activity("4649.01", "Household Appliances Trading", "تجارة الأجهزة المنزلية"),
        Activity("4772.01", "Perfumes and Cosmetics Trading", "تجارة العطور ومستحضرات التجميل"),
        Activity("4759.01", "Furniture Trading", "تجارة الأثاث"),
    ]
}


@dataclass(frozen=True)
class BusinessLine:
    en: str
    ar: str
    activity_codes: tuple[str, ...]
    item_group: str


BUSINESS_LINES: list[BusinessLine] = [
    BusinessLine("General Trading", "للتجارة العامة", ("4690.01", "4649.01", "4759.01"), "office"),
    BusinessLine("Building Materials", "لمواد البناء", ("4663.01", "4690.01"), "build"),
    BusinessLine("Technical Services", "للخدمات الفنية", ("4321.01", "8121.01"), "build"),
    BusinessLine("IT Solutions", "لحلول تقنية المعلومات", ("6202.01", "4651.01"), "it"),
    BusinessLine("Cleaning Services", "لخدمات التنظيف", ("8121.01",), "clean"),
    BusinessLine("Foodstuff Trading", "لتجارة المواد الغذائية", ("4630.01", "4690.01"), "food"),
    BusinessLine("Auto Parts", "لقطع غيار السيارات", ("4530.01",), "auto"),
    BusinessLine("Events", "لتنظيم الفعاليات", ("8230.01", "1811.01"), "events"),
    BusinessLine("Interior Design", "للتصميم الداخلي", ("7410.01", "4759.01"), "office"),
    BusinessLine("Logistics", "للخدمات اللوجستية", ("5229.01",), "logistics"),
    BusinessLine("Consultancy", "للاستشارات", ("7020.01", "6202.01"), "consult"),
    BusinessLine("Perfumes", "للعطور", ("4772.01",), "perfume"),
]


@dataclass(frozen=True)
class Item:
    en: str
    ar: str
    low: int  # unit price range in AED
    high: int
    max_qty: int = 40  # one delivery charge, a few consultancy days, many bags of cement


ITEMS: dict[str, list[Item]] = {
    "office": [
        Item("A4 copy paper, carton of 5 reams", "ورق تصوير A4، كرتونة 5 رزم", 95, 160),
        Item("Ergonomic office chair", "كرسي مكتب مريح", 350, 1200),
        Item("Office desk, 160 cm", "مكتب عمل 160 سم", 650, 1900),
        Item("Filing cabinet, 4 drawers", "خزانة ملفات 4 أدراج", 420, 980),
        Item("Delivery charges", "رسوم التوصيل", 50, 350, 1),
    ],
    "it": [
        Item("Laptop, 14 inch", "حاسوب محمول 14 بوصة", 2800, 5600),
        Item("Network switch, 24 port", "محول شبكة 24 منفذ", 900, 2600),
        Item("Annual software licence", "ترخيص برنامج سنوي", 1200, 9000, 10),
        Item("Website maintenance, monthly", "صيانة شهرية للموقع الإلكتروني", 600, 2500, 12),
        Item("Installation labour", "أجور التركيب", 200, 1500, 3),
    ],
    "build": [
        Item("Cement, 50 kg bag", "إسمنت، كيس 50 كجم", 14, 22),
        Item("Steel rebar, per tonne", "حديد تسليح للطن", 2400, 3200),
        Item("Ceramic floor tiles, per sqm", "بلاط سيراميك للمتر المربع", 28, 95),
        Item("LED panel light 60x60", "إضاءة لوحية 60x60", 65, 180),
        Item("Air conditioner servicing", "صيانة مكيفات الهواء", 150, 400, 12),
    ],
    "clean": [
        Item("Deep cleaning, per visit", "تنظيف عميق للزيارة الواحدة", 450, 1800, 6),
        Item("Office cleaning, monthly contract", "عقد تنظيف مكاتب شهري", 1500, 6000, 12),
        Item("Window cleaning, per floor", "تنظيف النوافذ للطابق", 300, 900, 20),
        Item("Cleaning supplies kit", "مجموعة مواد التنظيف", 80, 240),
    ],
    "food": [
        Item("Basmati rice, 20 kg bag", "أرز بسمتي، كيس 20 كجم", 70, 140),
        Item("Sunflower oil, 1.8 litres", "زيت دوار الشمس 1.8 لتر", 14, 28),
        Item("Dates, premium, 5 kg box", "تمور فاخرة، صندوق 5 كجم", 90, 260),
        Item("Mineral water, carton of 24", "مياه معدنية، كرتونة 24", 18, 32),
    ],
    "auto": [
        Item("Brake pads, front set", "فحمات فرامل أمامية", 120, 480),
        Item("Engine oil, 4 litres", "زيت محرك 4 لتر", 85, 190),
        Item("Car battery, 70 Ah", "بطارية سيارة 70 أمبير", 320, 690),
        Item("Tyre 265/65 R17", "إطار 265/65 R17", 380, 820),
    ],
    "events": [
        Item("Event stage setup", "تجهيز منصة الفعالية", 3500, 12000, 1),
        Item("Printed banners, per piece", "لافتات مطبوعة للقطعة", 60, 250),
        Item("Sound system rental, per day", "تأجير نظام صوت لليوم", 900, 3200, 5),
        Item("Event staff, per person per day", "طاقم الفعالية للفرد لليوم", 250, 600, 30),
    ],
    "logistics": [
        Item("Container freight, local", "شحن حاوية محلي", 900, 2400, 6),
        Item("Pallet storage, per month", "تخزين منصات شهري", 40, 120),
        Item("Customs clearance service", "خدمة التخليص الجمركي", 350, 1200, 3),
        Item("Last mile delivery, per drop", "توصيل للعميل النهائي للشحنة", 25, 60),
    ],
    "consult": [
        Item("Consultancy, per day", "استشارات لليوم الواحد", 1800, 4500, 10),
        Item("Process review workshop", "ورشة مراجعة الإجراءات", 3500, 9000, 2),
        Item("Report preparation", "إعداد التقرير", 1200, 5000, 1),
    ],
    "perfume": [
        Item("Perfume gift set", "طقم عطور للهدايا", 180, 950),
        Item("Oud oil, 12 ml", "دهن عود 12 مل", 250, 1400),
        Item("Bakhoor, 500 g", "بخور 500 جرام", 120, 450),
    ],
}

PEOPLE: list[tuple[str, str, Pair]] = [
    ("Ahmed Saeed Al Ketbi", "أحمد سعيد الكتبي", ("United Arab Emirates", "الإمارات")),
    ("Fatima Ali Al Mansoori", "فاطمة علي المنصوري", ("United Arab Emirates", "الإمارات")),
    ("Khalid Rashid Al Nuaimi", "خالد راشد النعيمي", ("United Arab Emirates", "الإمارات")),
    ("Mariam Yousef Al Hammadi", "مريم يوسف الحمادي", ("United Arab Emirates", "الإمارات")),
    ("Omar Hassan Al Shamsi", "عمر حسن الشامسي", ("United Arab Emirates", "الإمارات")),
    ("Rajesh Kumar Nair", "راجيش كومار ناير", ("India", "الهند")),
    ("Sara Ahmed Al Zaabi", "سارة أحمد الزعابي", ("United Arab Emirates", "الإمارات")),
    ("Yousef Ibrahim Haddad", "يوسف إبراهيم حداد", ("Jordan", "الأردن")),
    ("Layla Mahmoud Saleh", "ليلى محمود صالح", ("Egypt", "مصر")),
    ("Imran Tariq Qureshi", "عمران طارق قريشي", ("Pakistan", "باكستان")),
    ("Hamdan Obaid Al Suwaidi", "حمدان عبيد السويدي", ("United Arab Emirates", "الإمارات")),
    ("Noura Salem Al Dhaheri", "نورة سالم الظاهري", ("United Arab Emirates", "الإمارات")),
    ("Maria Santos Cruz", "ماريا سانتوس كروز", ("Philippines", "الفلبين")),
    ("David John Wilson", "ديفيد جون ويلسون", ("United Kingdom", "المملكة المتحدة")),
    ("Hessa Khalfan Al Muhairi", "حصة خلفان المهيري", ("United Arab Emirates", "الإمارات")),
    ("Tariq Nabil Khoury", "طارق نبيل خوري", ("Lebanon", "لبنان")),
]

PAYMENT_TERMS: list[Pair] = [
    ("Payment within 30 days by bank transfer", "الدفع خلال 30 يوما بالتحويل البنكي"),
    ("Payment within 15 days of invoice date", "الدفع خلال 15 يوما من تاريخ الفاتورة"),
    ("Cash or card on delivery", "الدفع نقدا أو بالبطاقة عند التسليم"),
]
QUOTE_PAYMENT_TERMS: list[Pair] = [
    ("50% advance, balance on delivery", "50% دفعة مقدمة والباقي عند التسليم"),
    ("100% on delivery", "الدفع كاملا عند التسليم"),
    ("30 days from invoice date", "30 يوما من تاريخ الفاتورة"),
]
DELIVERY_TERMS: list[Pair] = [
    ("Within 14 days of purchase order", "خلال 14 يوما من أمر الشراء"),
    ("Within 7 working days", "خلال 7 أيام عمل"),
    ("Ex stock, subject to prior sale", "من المخزون حسب التوفر"),
]
VAT_PERIODS: list[Pair] = [("Quarterly", "ربع سنوية"), ("Monthly", "شهرية")]
CT_YEAR: Pair = ("1 January to 31 December", "من 1 يناير إلى 31 ديسمبر")
CERT_STATEMENT: Pair = (
    "This certificate confirms registration with the Federal Tax Authority.",
    "تؤكد هذه الشهادة التسجيل لدى الهيئة الاتحادية للضرائب.",
)
