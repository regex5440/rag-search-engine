import argparse
from lib.multimodal_search import verify_image_embedding, image_search_command

def main():
    parser = argparse.ArgumentParser(description="Create & Verify embeddings for a image")
    sp = parser.add_subparsers(dest="command")
    sp.add_parser("verify_image_embedding", help="Verify an image embeddings").add_argument("path", type=str, help="Path to an image file")

    sp.add_parser("image_search", help="Search movie with an image").add_argument("path", type=str, help="Path to an image file")

    args = parser.parse_args()
    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.path)
        case "image_search":
            results = image_search_command(args.path)
            for i, r in enumerate(results):
                print(f'{i+1}. {r["title"]} (similarity: {r["score"]:.3f})')
                print(f'   {r["description"][:100]}\n')


if __name__ == "__main__":
    main()