import random
from datetime import datetime, timedelta

def generate_date_dataset(num_samples, start_date="1900-01-01", end_date="2099-12-31", seed=None):
    if seed is not None:
        random.seed(seed)
    
    input_formats = [
        "%Y-%m-%d",   # e.g., 2023-07-26
        "%d/%m/%Y",   # e.g., 26/07/2023
        "%m/%d/%Y",   # e.g., 07/26/2023
        "%B %d, %Y",  # e.g., July 26, 2023
        "%d %b %Y",   # e.g., 26 Jul 2023
        "%Y/%m/%d",   # e.g., 2023/07/26
        "%d-%m-%Y",   # e.g., 26-07-2023
        "%b %d, %Y",  # e.g., Jul 26, 2023
    ]

    output_formats = [
        "%B %d, %Y",  # e.g., July 26, 2023
        "%d %b %Y",   # e.g., 26 Jul 2023
        "%Y-%m-%d",   # e.g., 2023-07-26
        "%d/%m/%Y",   # e.g., 26/07/2023
        "%m/%d/%Y",   # e.g., 07/26/2023
        "%d-%m-%Y",   # e.g., 26-07-2023
        "%b %d, %Y",  # e.g., Jul 26, 2023
    ]

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    date_range = (end_date - start_date).days

    input_dates = []
    output_dates = []

    for _ in range(num_samples):
        random_days = random.randint(0, date_range)
        date = start_date + timedelta(days=random_days)
        
        input_format = random.choice(input_formats)
        output_format = random.choice(output_formats)
        
        # Ensure input and output formats are not the same
        while input_format == output_format:
            output_format = random.choice(output_formats)
        
        input_date = date.strftime(input_format)
        output_date = date.strftime(output_format)
        
        input_dates.append(input_date)
        output_dates.append(output_date)

    return input_dates, output_dates

# Generate a dataset of 10,000 samples
input_dates, output_dates = generate_date_dataset(10000)

# Print a few examples
for i in range(5):
    print(f"Input: {input_dates[i]} -> Output: {output_dates[i]}")