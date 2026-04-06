import argparse
from lib.llm import LLM
from mimetypes import guess_type

def main():
    parser = argparse.ArgumentParser(description="Describe text in text format")
    parser.add_argument("--image", type=str, help="Source path of the image file")
    parser.add_argument("--query", type=str, help="Text query to be rewritten based on supplied image")

    args = parser.parse_args()
    image = args.image
    query = args.query

    mime, _ = guess_type(args.image)
    mime = mime or "image/jpeg"

    with open(image, "rb") as f:
        llm = LLM()
        llm.describe_image(f.read(),mime, query)


if __name__ == "__main__":
    main()