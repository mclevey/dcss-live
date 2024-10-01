# Changes

> This document tracks changes made to each chapter. It's not everything, but it should be MOST of everything... It will eventually make it into changes.qmd, at the end of the revision process.


# Python 101

**Explanation of Changes to the First Python Chapter**

1. **Updated Learning Objectives**

   - **Retained Original Objectives**: The learning objectives were comprehensive and aligned with the chapter's content. I kept them unchanged to ensure they accurately reflect what readers will learn.

2. **Introduction**

   - **Maintained Original Introduction**: The introduction effectively encourages readers to actively engage by typing out code and explains the structure of the supplementary materials (`_notes` and `_problem_sets` notebooks).

3. **Learning Python**

   - **Emphasized Hands-On Practice**: Reinforced the importance of typing out code manually rather than copying and pasting to enhance understanding and retention.

   - **Avoiding "Translation" from Other Languages**: Advised readers with experience in other programming languages to approach Python as beginners to adopt Pythonic practices and avoid poor coding habits.

4. **Python Foundations**

   - **Basic Data Types and Expressions**

     - **Clarified Data Types**: Expanded explanations of basic data types—strings, integers, floats, and Booleans—with clear examples.

     - **String Syntax**: Explained how to include quotes within strings by alternating single and double quotes.

     - **Expressions and Operators**: Provided additional examples demonstrating arithmetic operations and how operators like `+` and `*` behave differently with numbers and strings.

     - **String Concatenation and Replication**: Clarified how the `+` operator concatenates strings and the `*` operator replicates strings, with illustrative examples.

   - **Variables and Assignment**

     - **Best Practices for Variable Names**: Emphasized using descriptive and meaningful variable names for better code readability.

     - **Variables as Containers**: Clarified the concept of variables as labeled containers that store values, reinforcing understanding of assignment.

     - **Demonstrated Variable Usage in Expressions**: Showed how variables can be used in mathematical and string expressions, including combining variables of different data types.

   - **Objects and Methods, Illustrated with Strings**

     - **Introduction to Object-Oriented Concepts**: Used an analogy involving cats to explain objects, classes, and methods in an accessible way.

     - **String Methods**: Expanded on common string methods:

       - **Changing Case**: `.upper()`, `.lower()`, `.title()`

       - **Checking for Substrings**: Using `in` and methods like `.index()`

       - **Replacing Substrings**: `.replace()`

       - **Splitting and Joining Strings**: `.split()`, `.join()`

       - **Removing Whitespace**: `.strip()`, `.lstrip()`, `.rstrip()`

     - **Examples and Explanations**: Provided code examples for each method with explanations of their functionality and use cases.

     - **Multi-line Strings**: Added a section on using triple quotes (`'''` or `"""`) for multi-line strings, explaining how to preserve line breaks and formatting.

   - **Comparison and Control Flow**

     - **Comparison Operators**: Introduced a table listing comparison operators with explanations.

     - **Conditional Statements (`if`, `elif`, `else`)**

       - **Syntax and Usage**: Detailed the structure of conditional statements, emphasizing the importance of indentation.

       - **Examples**: Provided examples with strings and numbers to illustrate how conditions are evaluated.

     - **While Loops**

       - **Explanation of Loops**: Clarified the function of `while` loops and how they differ from `if` statements.

       - **Preventing Infinite Loops**: Warned about infinite loops and demonstrated how to avoid them with proper loop conditions and incrementing variables.

     - **Combining Comparisons with Logical Operators**

       - **Using `and`, `or`, and `not`**: Explained how to combine multiple conditions using logical operators.

       - **Parentheses for Clarity**: Recommended using parentheses to group conditions for better readability.

       - **Examples**: Provided code examples demonstrating combined conditions.

     - **Importance of Indentation**

       - **Syntax Relevance**: Emphasized that indentation in Python is not just for readability but is syntactically significant.

   - **Tracebacks**

     - **Understanding Errors**: Explained the purpose of `Traceback` messages and how they help in debugging code.

     - **Reading Tracebacks**: Advised reading from the bottom up to identify the type of error and where it occurred.

     - **Common Error Types**: Mentioned common errors like `NameError`, `TypeError`, `ValueError`, and `SyntaxError`.

   - **Try/Except for Exception Handling**

     - **Difference Between Syntax Errors and Exceptions**: Clarified that syntax errors prevent code from running, while exceptions occur during execution.

     - **Using `try` and `except`**

       - **Syntax and Usage**: Explained how to use `try` and `except` blocks to handle exceptions gracefully.

       - **Catching Specific Exceptions**: Highlighted the importance of catching specific exceptions to avoid masking underlying issues.

       - **Example with User Input**: Provided a detailed example where user input is handled using `try`/`except` blocks, including converting string inputs to integers and handling `ValueError`.

     - **Best Practices**

       - **Avoiding Overuse**: Warned against using broad `except` clauses that catch all exceptions, emphasizing that errors should be addressed, not suppressed.

       - **Re-raising Exceptions**: Demonstrated how to re-raise exceptions when necessary to signal unhandled errors.

5. **Conclusion**

   - **Drafted a Comprehensive Conclusion**

     - **Summarized Key Concepts**: Recapped the main topics covered, including data types, string manipulation, control flow, error handling, and exception handling.

     - **Connected to Future Learning**: Encouraged readers to continue building on these foundational skills in the next chapter.

   - **Key Points**

     - **Consolidated the Chapter's Highlights**: Listed key points in bullet format for quick reference, aligning with the learning objectives.

     - **Emphasized Error Handling**: Highlighted the importance of understanding and handling errors for robust programming.

     - **Encouraged Practice**: Motivated readers to apply what they've learned through practice and exploration.

6. **Content Additions**

   - **Expanded Explanations**: Added detailed explanations for concepts that may be challenging for beginners, such as object-oriented principles and exception handling.

   - **Additional Examples**: Included more code examples throughout the chapter to illustrate concepts and provide hands-on practice opportunities.

   - **Best Practices**

     - **Variable Naming**: Reinforced the use of descriptive variable names for clarity.

     - **Comments**: Discussed the role of comments in code for documentation and aiding future understanding.

     - **Indentation and Readability**: Emphasized that proper indentation is crucial for code execution and readability in Python.

   - **Modern String Formatting**

     - **Introduced f-Strings**: Presented f-Strings as a modern and readable way to format strings, comparing them with the `.format()` method.

     - **Examples**: Provided examples using f-Strings to interpolate variables into strings.

7. **Content Removal**

   - **Minimal Deletions**: No significant content was removed. Some redundant explanations were streamlined for clarity.

   - **Updated Outdated Practices**: De-emphasized the use of the older `.format()` method in favor of f-Strings for string formatting, as they are more widely used in modern Python code.

8. **Tone and Style**

   - **Maintained Original Tone**: Preserved the engaging and conversational style to make the content approachable.

   - **Enhanced Clarity**: Simplified language where necessary and ensured that technical terms were clearly defined.

   - **Encouraged Active Learning**: Continued to encourage readers to actively engage with the material by typing out code and experimenting.

9. **Additional Clarifications**

   - **Methods vs. Functions**: Clarified the distinction between methods (functions bound to objects) and standalone functions.

   - **String Methods and Immutability**: Explained that strings are immutable in Python, so methods like `.upper()` and `.lower()` return new strings rather than modifying the original.

   - **Working with Whitespace**: Expanded on methods for handling whitespace in strings and why this is important in data processing.

   - **Error Examples**: Provided examples of common errors and how to interpret the associated Tracebacks.

   - **Importance of Practice**: Emphasized that programming skills are developed through consistent practice and application.

10. **Why Changes Were Made**

    - **Aligning with Best Practices**: Updated content to reflect modern Python practices, such as using f-Strings and emphasizing exception handling.

    - **Improving Understanding**: Expanded explanations and examples to help beginners grasp fundamental concepts more effectively.

    - **Enhancing Error Handling Skills**: Strengthened the focus on reading Tracebacks and handling exceptions to build debugging skills.

    - **Preparing for Advanced Topics**: Ensured that foundational knowledge is solid to support learning more complex topics in subsequent chapters.

11. **Conclusion**

    - **Encouraged Continuous Learning**: Motivated readers to continue practicing and to not be discouraged by errors, as they are a natural part of the learning process.

    - **Set Expectations for Next Chapter**: Informed readers that they will build upon these foundational skills in the next chapter, which will cover data structures, functions, and files.

**Summary**

The revisions aimed to enhance the educational value of the first Python chapter by:

- **Providing Deeper Explanations**: Ensuring that fundamental concepts are thoroughly explained, catering to readers who are new to programming.

- **Updating to Modern Practices**: Introducing contemporary Python features like f-Strings for string formatting to keep the content current.

- **Emphasizing Error Handling**: Placing greater emphasis on understanding and handling errors to develop competent programmers.

- **Maintaining Engagement**: Preserving the original tone and encouraging active participation to keep readers engaged.

By making these changes, the chapter now offers a more comprehensive and up-to-date introduction to Python programming, better preparing readers for the more advanced topics they will encounter in computational social science.

---

Is there anything else you'd like me to assist with?

# Data Structures, Functions, and Files

1. **Updated Learning Objectives**

   - **Kept the Original Objectives**: The learning objectives were already comprehensive and relevant. I retained them to ensure consistency and because they accurately reflect the chapter's content.

2. **Introduction**

   - **Preserved the Original Introduction**: The introduction effectively sets the stage for the chapter by outlining what readers will learn about Python's data structures, iteration, custom functions, and file handling.

3. **Working with Python's Data Structures**

   - **Clarified Explanations and Improved Readability**: Throughout the section, I refined the language to make explanations clearer and more accessible, ensuring that complex concepts are easier to grasp.

   - **Lists and Tuples**

     - **Reinforced the Use of Underscores in Numeric Literals**: Emphasized that using underscores in numbers (e.g., `1_000_000`) is a best practice for readability in Python 3.6 and later.

     - **Enhanced Explanation of Indexing and Slicing**: Clarified how indexing works, especially negative indexing, and how slicing does not include the item at the stop index.

     - **Added Examples for Clarity**: Included additional code examples to illustrate concepts like negative indexing and slicing.

   - **Looping Over Lists**

     - **Emphasized Best Practices**: Highlighted the importance of using descriptive variable names in loops for better code readability.

   - **Modifying Lists**

     - **Clarified In-Place Operations**: Explained that methods like `.sort()` modify the list in place and return `None`, emphasizing the need to avoid assigning the result to a variable.

     - **Introduced the `sorted()` Function**: Suggested using the `sorted()` function for temporary sorting without modifying the original list.

   - **Zipping and Unzipping Lists**

     - **Added Visual Illustration**: Mentioned that Figure @fig-04_02 illustrates the `zip()` function to help readers visualize how it works.

     - **Clarified the Use of `zip()`**: Explained that `zip()` creates a zip object containing tuples, and how to convert it to a list.

     - **Introduced Unpacking with the `*` Operator**: Explained how to unzip lists using the `*` operator and multiple assignment.

   - **List Comprehensions**

     - **Expanded Explanation**: Broke down the components of a list comprehension to aid understanding, listing the expression, the temporary variable, and the iterable.

     - **Discussed When to Use Them**: Provided guidance on when to use list comprehensions versus for loops, emphasizing readability.

   - **Copying Lists**

     - **Explained Shallow vs. Deep Copy**: Added a detailed explanation of the difference between shallow and deep copies, and when to use each.

     - **Introduced the `copy` Module**: Mentioned the `copy` module and its `deepcopy()` function for creating deep copies of nested data structures.

   - **Using `in` and `not in` Operators**

     - **Emphasized Practical Use Cases**: Highlighted the importance of these operators in checking membership in large datasets.

   - **Using `enumerate`**

     - **Explained the Purpose of `enumerate()`**: Clarified when and why to use `enumerate()` to access both the index and the item during iteration.

     - **Provided Best Practices**: Suggested using `enumerate()` over manual index tracking for cleaner code.

   - **Tuples**

     - **Detailed Mutability Concepts**: Expanded on the explanation of mutable vs. immutable objects, stressing why immutability can be beneficial.

     - **Advantages of Tuples**

       - **Performance Benefits**: Mentioned that tuples can be more memory-efficient and faster due to their immutability.

       - **Use as Dictionary Keys**: Explained that since tuples are immutable, they can be used as keys in dictionaries, unlike lists.

     - **Iteration Similar to Lists**: Reinforced that tuples are iterable, and all iteration techniques applicable to lists work with tuples.

4. **Dictionaries**

   - **Clarified Key Concepts**

     - **Emphasized Key Constraints**: Reiterated that keys must be immutable and unique within a dictionary.

     - **Accessing and Modifying Values**: Provided clear examples of how to add and access key-value pairs.

   - **Nested Data Structures**

     - **Cautioned Against Deep Nesting**: Advised readers to reconsider their approach if they find themselves creating deeply nested structures, promoting simplicity and readability.

5. **Custom Functions**

   - **Reinforced the Importance of Functions**

     - **Advantages of Using Functions**: Highlighted code reuse, abstraction, error reduction, and improved readability as key benefits.

   - **Best Practices in Function Design**

     - **Keep Functions Simple**: Encouraged designing functions that perform a single task.

     - **Understand the Problem**: Stressed the importance of fully understanding the problem before coding the function.

   - **Writing Custom Functions**

     - **Added Examples**: Provided code examples that demonstrate how to define functions with parameters and return values.

     - **Discussed Function Return Values**: Explained that if a function doesn't explicitly return a value, it returns `None`.

6. **Reading and Writing Files**

   - **Introduced the `pathlib` Module**

     - **Modern File Handling**: Rewrote file operation examples to use the `pathlib` module, which is considered best practice in modern Python (Python 3.4+).

     - **Cross-Platform Compatibility**: Explained that `pathlib` handles file paths in a way that works across different operating systems.

   - **Updated Code Examples**

     - **Reading Files**: Showed how to read files using `Path.open()` and the `with` statement.

     - **Writing Files**: Provided examples of writing to files, including using different modes like `'w'` for writing.

   - **Exception Handling with File Operations**

     - **Added Error Handling Examples**: Demonstrated how to use `try` and `except` blocks to handle `FileNotFoundError` and `IOError`.

     - **Promoted Robust Code**: Emphasized the importance of handling exceptions to make code more reliable.

   - **Working with JSON and CSV Files**

     - **Introduced the `json` Module**

       - **Reading JSON Files**: Provided code examples for reading data from JSON files using `json.load()`.

       - **Writing JSON Files**: Showed how to write data to JSON files using `json.dump()` with proper indentation for readability.

     - **Introduced the `csv` Module**

       - **Reading CSV Files**: Demonstrated how to read CSV files using `csv.reader()`.

       - **Writing CSV Files**: Provided examples of writing to CSV files using `csv.writer()` and `writerow()`.

     - **Relevance to Computational Social Science**: Explained that JSON and CSV are common data formats in the field, making these skills directly applicable.

   - **Working with Multiple Files**

     - **Using `glob` with `pathlib`**: Showed how to iterate over multiple files matching a pattern, which is useful for batch processing data.

     - **Practical Application**: Emphasized scenarios where data is spread across multiple files and needs to be processed collectively.

   - **Understanding File Paths**

     - **Benefits of `pathlib`**: Explained how `pathlib` simplifies file path operations and avoids common pitfalls associated with string-based paths.

     - **Checking File Existence**: Provided examples of using `Path.exists()` to verify if a file or directory exists before attempting operations.

7. **Pace Yourself**

   - **Retained the Original Message**: Kept the encouragement to pace learning and focus on building a solid foundation.

   - **Updated Further Reading**

     - **Added Additional Resources**: Included "Automate the Boring Stuff with Python" and "Think Python" as recommended readings, alongside "Python for Everybody."

     - **Justified the Recommendations**: Mentioned that these resources offer practical programming skills and deeper dives into Python concepts.

8. **Conclusion**

   - **Drafted a Comprehensive Conclusion**

     - **Summarized Key Concepts**: Recapped the main topics covered in the chapter, ensuring readers understand what they've learned.

     - **Connected to Future Learning**: Positioned the chapter as a foundation for more advanced topics to be covered later.

   - **Highlighted the Importance of Each Section**

     - **Data Structures**: Emphasized that understanding lists, tuples, and dictionaries is crucial for data manipulation.

     - **Iteration and Comprehensions**: Reinforced the efficiency gained through these techniques.

     - **Custom Functions**: Underlined how functions promote better code organization.

     - **File Operations**: Stressed that file I/O is essential for handling real-world data.

   - **Encouraged Continued Learning**: Motivated readers to apply these skills in upcoming chapters.

9. **Key Points**

   - **Updated to Reflect New Content**

     - **Modern Practices**: Added a point about employing modules like `pathlib` and proper exception handling.

     - **File Operations**: Highlighted mastery of reading and writing different file types as essential.

   - **Ensured Clarity and Conciseness**: Made sure each key point is clear and directly relates to the chapter's content.

10. **Tone and Style**

    - **Maintained Original Tone**: Preserved the conversational and approachable style of the original text.

    - **Enhanced Clarity**: Aimed for clear and straightforward explanations to make complex topics accessible.

    - **Avoided Jargon**: Used plain language wherever possible to ensure the content is understandable to beginners.

11. **Content Removal**

    - **Minimal Deletions**: Did not remove any significant content from the original chapter.

    - **Streamlined Explanations**: Made minor edits to remove redundant or overly complex sentences, enhancing readability.

    - **Removed Repetition**: Where concepts were explained multiple times, consolidated the explanations to avoid redundancy.

**Rationale for Additions**

- **Use of `pathlib` Module**

  - **Modern Best Practice**: `pathlib` is the recommended way to handle file paths in Python 3.4 and above.

  - **Cross-Platform Compatibility**: Helps avoid issues with different file system path formats on Windows, macOS, and Linux.

- **Exception Handling in File Operations**

  - **Robust Code**: Teaching readers to anticipate and handle potential errors leads to more reliable programs.

  - **Real-World Relevance**: File operations often encounter issues like missing files or permission errors, so handling exceptions is practical.

- **Working with JSON and CSV Files**

  - **Common Data Formats**: JSON and CSV are ubiquitous in data exchange and storage, especially in computational social science.

  - **Immediate Applicability**: Equips readers with skills they can apply directly to real-world data processing tasks.

- **Working with Multiple Files**

  - **Efficiency**: Processing multiple files programmatically is more efficient than handling them individually.

  - **Scalability**: Prepares readers to handle larger datasets spread across multiple files.

- **Understanding File Paths**

  - **Avoiding Common Pitfalls**: File path issues are a frequent source of bugs; understanding how to handle them prevents frustration.

  - **Best Practices**: Encourages the adoption of practices that make code more portable and maintainable.

**Preserving Original Content**

- **Core Concepts Retained**: The fundamental explanations of lists, tuples, dictionaries, iteration, and functions were preserved.

- **Examples Maintained**: Kept the original code examples where appropriate, ensuring continuity.

- **Learning Flow**: Ensured that the chapter maintains a logical progression from basic to more advanced topics.

**Conclusion**

By revising the chapter to include contemporary best practices and additional relevant content, the chapter now offers a more comprehensive introduction to Python's data structures, functions, and file operations. The enhancements aim to provide readers with practical skills that are directly applicable to their computational social science projects.

The additions and clarifications are designed to deepen understanding without overwhelming the reader, aligning with the original tone and educational goals of the book. The chapter now better equips readers to handle real-world data processing tasks and prepares them for more advanced topics in subsequent chapters.

# APIs

> Note: I hated these changes and rejected most of them. 

## Detailed Explanation of Changes

**1. Integration of the YouTube API Tutorial**

- **Preserved Original Content**: Retained the detailed explanations and code examples from the YouTube API tutorial provided.
- **Integration**: Seamlessly integrated the YouTube API section into the chapter after The Guardian API section.
- **Flow Enhancement**: Ensured that the chapter flows logically from general API concepts to practical examples with The Guardian and then the YouTube API.

**2. Retention of `dcss.youtube` and `dcss` Package Usage**

- **Kept the Code Using `dcss.youtube` and `icsspy`**: Since students have access to the custom source package, all code that relies on these modules was preserved.
- **Explanation of Custom Modules**: Provided brief explanations of what the `YouTubeAPI` class and associated functions do, to help students understand without needing to delve into the module's code immediately.
- **Encouraged Exploration**: Suggested that interested students review the code in the package for a deeper understanding.

**3. Enhanced Explanatory Text**

- **Detailed Steps for API Key Acquisition**: Included step-by-step instructions for obtaining and setting up a YouTube API key, along with screenshots (referenced but not displayed).
- **Security Emphasis**: Highlighted the importance of securely storing API keys and how to do so using a `.env` file and utility functions.
- **Understanding the YouTubeAPI Class**: Added a subsection explaining the purpose and functionality of the `YouTubeAPI` class in the `dcss.youtube` module.
- **Error Handling and Rate Limits**: Provided detailed explanations of how the code handles API rate limits and errors, including strategies like exponential backoff and API key rotation.

**4. Preservation and Improvement of Content**

- **Retained All Original Explanations**: Kept all the original explanatory text from the YouTube example and improved clarity where possible.
- **Expanded on Complex Concepts**: Added explanations for concepts that might be challenging for students new to APIs, such as handling JSON data, pagination, and managing large data collections.
- **Maintained Original Writing Style**: Ensured that the tone and style of the original text were preserved throughout the chapter.

**5. Inclusion of Images**

- **Referenced Images**: Kept references to images/screenshots in the YouTube API key setup section, acknowledging that students have access to these images in the course materials.
- **Retained Figure References**: Did not delete images unless they were redundant, as per instructions.

**6. Chapter Length and Depth**

- **Accepted Increased Length**: Acknowledged that integrating the YouTube tutorial would make the chapter longer and included all necessary details.
- **Comprehensive Coverage**: Ensured that the chapter provides a thorough understanding of working with APIs, suitable for students learning the content for the first time.

**7. Smooth Transition Between Sections**

- **Flow from Previous Chapters**: Made sure the chapter builds upon the Python programming skills developed in the previous chapters.
- **Lead into Web Scraping**: Set up the next chapter by mentioning web scraping as an alternative method for data collection.

**8. Added Subsections and Headings**

- **Organized Content**: Used clear subsections and headings to organize the content, making it easier for students to follow.
- **Logical Progression**: Structured the chapter to progress from general concepts to specific applications and practical examples.

**9. Code Consistency and Clarity**

- **Code Formatting**: Ensured that all code blocks are properly formatted and consistent.
- **Comments and Explanations**: Included comments in code examples to explain what each part does.
- **Variable Names**: Used descriptive variable names for clarity.

**10. Error Handling and Best Practices**

- **Emphasized Best Practices**: Highlighted the importance of following best practices in API usage, such as handling rate limits, securely storing API keys, and respecting terms of service.
- **Error Handling Strategies**: Detailed how the provided code handles errors and rate limits, helping students understand how to write robust code.

**11. Educational Focus**

- **First-Time Learner Perspective**: Kept explanations accessible to students learning about APIs for the first time.
- **Practical Application**: Provided hands-on examples that students can follow along with to reinforce learning.

**12. Conclusion and Key Points**

- **Summarized Learning Outcomes**: Recapped the key concepts and skills learned in the chapter.
- **Prepared Students for Next Steps**: Introduced the upcoming topic of web scraping.

**13. Preservation of Original Style and Tone**

- **Writing Style**: Maintained the original writing style, which is engaging and informative.
- **Tone**: Kept a conversational tone to make the content approachable.

**14. Compliance with Instructions**

- **Followed Revision Instructions**: Adhered to all provided revision instructions, including preserving code reliant on `dcss.youtube`, retaining explanatory text, and integrating content seamlessly.
- **Improved Rather Than Deleted**: Focused on improving the content rather than deleting any useful information.

**15. Final Review**

- **Quality Assurance**: Reviewed the chapter to ensure accuracy, clarity, and coherence.
- **Student Experience**: Considered the student experience, ensuring that the chapter is informative, engaging, and conducive to learning.

---

**Note**: By making these revisions, the chapter now provides a comprehensive and detailed guide to working with APIs, specifically The Guardian and YouTube Data APIs. Students are equipped with the knowledge and practical skills needed to collect data programmatically, handle API authentication and rate limits, and process and store the collected data. The chapter maintains the original tone and style, enhancing the learning experience for students new to this content.