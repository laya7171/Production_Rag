import logfire
from unstructured.partition.auto import partition


def parse_office(file_path: str):
    """Parse office files and return a list of elements"""


    with logfire.span("parse_office", filename = file_path):
        try:
            elements = partition(file_path)
            full_text = "\n".join([element.text for element in elements])

            if not full_text.strip():
                logfire.warning("No text found in the document.")
            else:
                logfire.info(f"Parsed {len(elements)} elements from the document.")


            return full_text

        except Exception as e:
            logfire.error(f"Error parsing office file: {e}")
            return ""