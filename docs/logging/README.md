## Installation
Install the latest version in your Python evironment from `PyPi`: [confidential-ml-utils](https://pypi.org/project/confidential-ml-utils/)

## Exception Handling

First execute `pip install confidential-ml-utils` to install this library. Then
wrap any methods which may throw an exception with the decorator
`prefix_stack_trace`. Here's a simple example. Your code may explicitly raise
the `Public*` exceptions (`PublicValueError`, `PublicRuntimeError`,
`PublicArgumentError`, `PublicKeyError`, `PublicTypeError`) when you know that
the content of the exception does not contain any private content. The messages
in these exceptions will be preserved, even if `keep_message` is set to `False`.

```python
from confidential_ml_utils.exceptions import prefix_stack_trace

@prefix_stack_trace()
def main():
    print("Hello, world!")

if __name__ == "__main__":
    main()
```

## Logging

Call `confidential_ml_utils.enable_confidential_logging` to set up data
category-aware logging. Then continue to use standard Python logging
functionality as before! Add a `category=DataCategory.PUBLIC` argument to have
your log lines prefixed with `SystemLog:`. For a full-fledged example, see
[data-category.py](./data-category.py).

## Examples

The simplest use case (wrap your `main` method in a decorator) is in:
[hello-world.py](./hello-world.py).

Some configuration options around prefixing the stack trace. [prefix-stack-trace.py](./prefix-stack-trace.py). You can:
-  customize the prefix and the exception message
-  keep the original exception message (don't scrub)
-  pass an allow_list of strings. Exception messages will be scrubbed unless the message or the
exception type regex match one of the allow_list strings.

Use this library with `with` statements:
[with-statement.py](./with-statement.py).

Using this library directly inside `try` / `except` statements:
[try-except.py](./try-except.py).

Using the `Public*` exception types:
[public-exceptions.py](./public-exceptions.py).

## Exception or Stack trace parsing

[StacktraceExtractor](../../src/confidential_ml_utils/StacktraceExtractor.py) is a simple tool to grab Python or C# stack
traces and exceptions from log files. Sometimes the file that has the stack trace you need may also contain sensitive
data. Use this tool to parse and print the stack trace, exception type and optionally exception message (careful as 
exception messages may also potentially hold private data.)

```python
ee = StacktraceExtractor()
ee.extract("log_file")
```
