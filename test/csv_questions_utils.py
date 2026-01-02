import csv
from collections import defaultdict
from pathlib import Path


def load_questions(csv_path="../questions.csv"):
    """
    Load questions from CSV file and group by category.

    Args:
        csv_path: Path to the questions CSV file

    Returns:
        dict: Dictionary mapping categories to lists of questions
    """
    script_dir = Path(__file__).parent
    csv_file = script_dir / csv_path

    questions_by_category = defaultdict(list)

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = row["Category"]
            question = row["Question"]
            questions_by_category[category].append(question)

    return questions_by_category


def display_category_menu(categories):
    """
    Display a numbered menu of categories.

    Args:
        categories: List of category names

    Returns:
        None
    """
    print("\n" + "=" * 80)
    print("SELECT CATEGORIES")
    print("=" * 80)
    print("\nAvailable Categories:\n")

    for idx, category in enumerate(categories, 1):
        print(f"  {idx}. {category}")

    print(f"\n  {len(categories) + 1}. All Categories")
    print("\n" + "─" * 80)


def get_user_selection(categories):
    """
    Get category selection from user.

    Args:
        categories: List of category names

    Returns:
        list: Selected category names
    """
    while True:
        try:
            choice = (
                input(
                    "\nEnter your choice(s) separated by commas (e.g., 1,3,5) or 'all': "
                )
                .strip()
                .lower()
            )

            if choice == "all":
                return categories

            # Parse comma-separated numbers
            selected_indices = [int(x.strip()) for x in choice.split(",")]

            # Validate indices
            valid_categories = []
            for idx in selected_indices:
                if 1 <= idx <= len(categories):
                    valid_categories.append(categories[idx - 1])
                else:
                    print(f"⚠️  Invalid choice: {idx}. Skipping...")

            if valid_categories:
                return valid_categories
            else:
                print("❌ No valid categories selected. Please try again.")

        except ValueError:
            print(
                "❌ Invalid input. Please enter numbers separated by commas or 'all'."
            )
        except KeyboardInterrupt:
            print("\n\n👋 Exiting...")
            exit(0)


def print_selected_questions(selected_categories, questions_by_category):
    """
    Print questions for selected categories in a pretty format.

    Args:
        selected_categories: List of selected category names
        questions_by_category: Dictionary mapping categories to questions

    Returns:
        list: List of all selected questions as strings
    """
    all_questions = []

    print("\n" + "=" * 80)
    print("SELECTED QUESTIONS")
    print("=" * 80)

    for category in selected_categories:
        if category not in questions_by_category:
            print(f"\n⚠️  Category '{category}' not found. Skipping...")
            continue

        questions = questions_by_category[category]
        all_questions.extend(questions)

        print(f"\n{'─' * 80}")
        print(f"📁 {category}")
        print(f"{'─' * 80}")
        print(f"Total Questions: {len(questions)}\n")

        for idx, question in enumerate(questions, 1):
            print(f"  {idx:2d}. {question}")

    print("\n" + "=" * 80)
    print(f"Total Questions Selected: {len(all_questions)}")
    print("=" * 80)

    return all_questions


def format_questions_as_python_string(questions):
    """
    Format questions as a Python list of strings.

    Args:
        questions: List of question strings

    Returns:
        str: Python code string representing the list
    """
    python_code = "questions = [\n"
    for question in questions:
        # Use repr() to properly escape all special characters
        python_code += f"    {repr(question)},\n"
    python_code += "]"
    return python_code


def interactive_menu():
    """
    Main interactive entry point for selecting and displaying questions.
    """
    print("\n" + "=" * 80)
    print("QUESTION SELECTOR")
    print("=" * 80)

    # Load questions
    questions_by_category = load_questions()
    sorted_categories = sorted(questions_by_category.keys())

    # Display menu
    display_category_menu(sorted_categories)

    # Get user selection
    selected_categories = get_user_selection(sorted_categories)

    # Print selected questions
    selected_questions = print_selected_questions(
        selected_categories, questions_by_category
    )

    # Format as Python string
    print("\n" + "=" * 80)
    print("PYTHON STRING FORMAT")
    print("=" * 80)
    print("\nCopy the following code for future use:\n")
    print(format_questions_as_python_string(selected_questions))
    print("\n" + "=" * 80)


def print_questions_pretty(csv_path="../questions.csv"):
    """
    Read and print questions from CSV file in a pretty formatted way.

    Args:
        csv_path: Path to the questions CSV file
    """
    questions_by_category = load_questions(csv_path)

    # Print in a pretty format
    print("=" * 80)
    print("QUESTIONS BY CATEGORY")
    print("=" * 80)
    print()

    # Sort categories for consistent output
    sorted_categories = sorted(questions_by_category.keys())

    for category in sorted_categories:
        questions = questions_by_category[category]
        print(f"\n{'─' * 80}")
        print(f"📁 {category}")
        print(f"{'─' * 80}")
        print(f"Total Questions: {len(questions)}\n")

        for idx, question in enumerate(questions, 1):
            print(f"  {idx:2d}. {question}")

        print()

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total_questions = sum(
        len(questions) for questions in questions_by_category.values()
    )
    print(f"Total Categories: {len(questions_by_category)}")
    print(f"Total Questions: {total_questions}")
    print("\nQuestions per Category:")
    for category in sorted_categories:
        count = len(questions_by_category[category])
        print(f"  • {category}: {count}")


if __name__ == "__main__":
    interactive_menu()
