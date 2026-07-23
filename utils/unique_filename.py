import datetime
import random
import string

def generate_unique_filename():
    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix_length = 6
    suffix = "".join(
        random.choices(string.ascii_letters + string.digits, k=suffix_length)
    )
    filename = f"{date_str}_{suffix}"
    return filename
