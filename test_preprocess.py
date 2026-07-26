from src.preprocess import clean_resume

sample = """
John Doe
Email: john@gmail.com
Phone: 9876543210

Experienced Python Developer with 5 years of experience.
Skilled in Machine Learning, SQL, and Data Science.

https://linkedin.com/in/johndoe
"""

print(clean_resume(sample))