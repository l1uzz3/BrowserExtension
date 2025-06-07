# Inside libs/FeaturesExtract.py
import os, pickle as pk, pandas as pd, httpx, whois
from urllib.parse import urlparse
from backend.libs import ExtractFunc as ef

def featureExtraction(url):
    # 1) Address-bar features (6)
    url_length   = ef.getLength(url)
    url_depth    = ef.getDepth(url)
    tiny_url     = ef.tinyURL(url)
    prefix_suf   = ef.prefixSuffix(url)
    n_dots       = ef.no_of_dots(url)
    sensitive    = ef.sensitive_word(url)

    # 2) Domain-based features (2)
    domain_name = ''
    try:
        domain_name = whois.whois(urlparse(url).netloc)
        domain_age = ef.domainAge(domain_name)
        domain_end = ef.domainEnd(domain_name)
    except Exception:
        # fallback
        domain_age = 1
        domain_end = 1

    # 3) Unicode / “@” / IP features (3 separate columns)
    has_unicode = ef.has_unicode(url)
    has_at      = ef.haveAtSign(url)
    has_ip      = ef.havingIP(url)

    # 4) Raw DOM flags (3 columns)
    try:
        response = httpx.get(url, timeout=5.0)
        iframe_flag     = ef.iframe(response)
        web_forwards    = ef.forwarding(response)
        mouse_over_flag = ef.mouseOver(response)
    except Exception:
        iframe_flag     = 1
        web_forwards    = 1
        mouse_over_flag = 1

    # 5) Build a DataFrame of exactly these columns (same names as at training time)
    feature_dict = {
        'URL_Length':    url_length,
        'URL_Depth':     url_depth,
        'TinyURL':       tiny_url,
        'Prefix/Suffix': prefix_suf,
        'No_Of_Dots':    n_dots,
        'Sensitive_Words': sensitive,
        'Domain_Age':    domain_age,
        'Domain_End':    domain_end,
        'Has_Unicode':   has_unicode,
        'Have_AtSign':   has_at,
        'Having_IP':     has_ip,
        'iFrame':        iframe_flag,
        'Web_Forwards':  web_forwards,
        'Mouse_Over':    mouse_over_flag
    }

    # 6) Return a 1×13 DataFrame (or however many columns PyCaret originally saw)
    row = pd.DataFrame([feature_dict])
    return row
