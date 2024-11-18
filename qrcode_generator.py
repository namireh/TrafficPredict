import qrcode

# URL you want to encode
url = "https://traffic-volume-predictor.streamlit.app/"

# Create QR code object
qr = qrcode.QRCode(
    version=1,  # controls the size of the QR Code
    error_correction=qrcode.constants.ERROR_CORRECT_L,  # error correction level
    box_size=10,  # size of each box in the QR code
    border=4,  # border width in boxes
)

# Add the URL data to the QR code
qr.add_data(url)
qr.make(fit=True)

# Generate the QR code image
img = qr.make_image(fill="black", back_color="white")
img.save("qrcode.png")
