import logfire

def parse_text(file_path: str):
    """Parse text file and return the text content."""

    with logfire.span("Text Parser", filename=file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                return content

        except Exception as e:
            logfire.error(f"Error parsing text file: {e}")
            raise e