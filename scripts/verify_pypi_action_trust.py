"""Compare an old public wheel's CA data with the action's installed certifi."""

import hashlib
import io
import json
import ssl
import urllib.request
import zipfile
from importlib.metadata import version

import certifi

with urllib.request.urlopen("https://pypi.org/pypi/certifi/2026.7.22/json") as response:
    metadata = json.load(response)
wheel = next(item for item in metadata["urls"] if item["packagetype"] == "bdist_wheel")
with urllib.request.urlopen(wheel["url"]) as response:
    data = response.read()
assert hashlib.sha256(data).hexdigest() == wheel["digests"]["sha256"]
archive = zipfile.ZipFile(io.BytesIO(data))
old_pem = archive.read("certifi/cacert.pem").decode()
old = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
old.load_verify_locations(cadata=old_pem)
new = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
new.load_verify_locations(cafile=certifi.where())


def roots(context):
    return {(cert["issuer"], cert["serialNumber"]): cert for cert in context.get_ca_certs()}


before, after = roots(old), roots(new)
removed = [before[key] for key in before.keys() - after.keys()]
print("installed certifi=" + version("certifi"))
print("roots: old=" + str(len(before)) + ", new=" + str(len(after)))
for cert in removed:
    print("removed:", cert["subject"], "serial:", cert["serialNumber"])
assert removed, "No roots removed from the old trust store"
assert all("emsign" in str(cert["subject"]).lower() for cert in removed), removed
assert any(cert["serialNumber"] == "3CF607A968700EDA8B84" for cert in removed), removed
print("PASS: removed emSign root certificates are absent from the action trust store")
