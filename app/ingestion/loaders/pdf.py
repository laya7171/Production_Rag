import logfire
from pypdf import PdfReader


def parse_pdf(file_path: str):
    """Extract text from a PDF file and return the text content."""

    with logfire.span("PDF Parser", filename=file_path):
        try:
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            logfire.info(f"Total pages in PDF: {total_pages}")

            text_parts: list[str] = []
            blank_pages: list[int] = []

            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""

                if text.strip():
                    text_parts.append(text)
                else:
                    blank_pages.append(i + 1)

            if blank_pages:
                logfire.info(
                    f"pypdf returned blank text for pages: {blank_pages}"
                )

                try:
                    import pdfplumber

                    with pdfplumber.open(file_path) as pdf:
                        for page_num in blank_pages:
                            page = pdf.pages[page_num - 1]
                            fallback_text = page.extract_text() or ""

                            if fallback_text.strip():
                                text_parts.append(fallback_text)

                except Exception as e:
                    logfire.error(
                        f"Error using pdfplumber for fallback: {e}"
                    )

            if not text_parts:
                logfire.error("Failed to extract text from the PDF.")

            return "\n".join(text_parts)

        except Exception as e:
            logfire.error(f"Error parsing PDF: {e}")
            raise