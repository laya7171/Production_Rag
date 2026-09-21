from bs4 import BeautifulSoup
import logfire

def parser_html(file_path: str):
    """Parse HTML file and return the text content."""

    with logfire.span("HTML Parser", filename = file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

                soup = BeautifulSoup(content, "html.parser")

                for script in soup(["script", "style","meta","noscript"]):
                    script.decompose()
                
                text = soup.get_text(separator="\n")

                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text_clean = '\n'.join(chunk for chunk in chunks if chunk)

                return text_clean

        except Exception as e:
            logfire.error(f"Error parsing HTML file: {e}")
            return None