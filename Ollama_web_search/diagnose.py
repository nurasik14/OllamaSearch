import requests

import truststore
truststore.inject_into_ssl()

url = "https://adilet.zan.kz/kaz/docs/K1400000226"

try:
    r = requests.get(url, timeout=20)
    print("default verify OK", r.status_code)
except Exception as e:
    print("default verify failed:", repr(e))
