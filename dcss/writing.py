from pathlib import Path

def split_markdown_by_chapters(input_file):
    # Read the markdown file's content
    input_path = Path(input_file)
    content = input_path.read_text(encoding='utf-8').splitlines()

    # Create output directory for chapters if it doesn't exist
    output_dir = Path("chapters")
    output_dir.mkdir(exist_ok=True)

    # Initialize variables for processing
    chapter_content = []
    chapter_title = None
    output_files = []

    # Iterate over each line in the markdown file
    for line in content:
        # Check if the line is a chapter title
        if line.startswith('# ') and '.CT-Chapter-Title' in line:
            # If we're currently building a chapter, write it to a file
            if chapter_title and chapter_content:
                # Generate filename from the previous chapter title
                chapter_filename = chapter_title.lower().replace(' -- ', '-').replace(' ', '-').replace('.', '').replace(',', '').replace(':', '').replace('?', '').replace('/', '') + '.md'
                chapter_path = output_dir / chapter_filename
                # Write the current chapter to a file
                chapter_path.write_text('\n'.join(chapter_content), encoding='utf-8')
                output_files.append(chapter_filename)

            # Reset chapter content and set new chapter title
            chapter_content = [line]
            chapter_title = line.split(' ', 1)[1].split(' {', 1)[0].strip()  # Extract the chapter title text only

        else:
            # Add line to the current chapter content
            chapter_content.append(line)

    # Write the last chapter to a file if exists
    if chapter_title and chapter_content:
        chapter_filename = chapter_title.lower().replace(' ', '-').replace('.', '').replace(',', '') + '.md'
        chapter_path = output_dir / chapter_filename
        chapter_path.write_text('\n'.join(chapter_content), encoding='utf-8')
        output_files.append(chapter_filename)

    return output_files