# Confidential ML Utilities

[![python](https://github.com/Azure/confidential-ml-utils/workflows/python/badge.svg)](https://github.com/Azure/confidential-ml-utils/actions?query=workflow%3Apython)
[![codecov](https://codecov.io/gh/Azure/confidential-ml-utils/branch/main/graph/badge.svg?token=TEWT51C5FK)](https://codecov.io/gh/Azure/confidential-ml-utils)
[![CodeQL](https://github.com/Azure/confidential-ml-utils/workflows/CodeQL/badge.svg)](https://github.com/Azure/confidential-ml-utils/actions?query=workflow%3ACodeQL)
[![Component Governance](https://dev.azure.com/msdata/Vienna/_apis/build/status/aml-ds/Azure.confidential-ml-utils%20Component%20Governance?branchName=master)](https://dev.azure.com/msdata/Vienna/_build/latest?definitionId=13909&branchName=master)
[![PyPI version](https://badge.fury.io/py/confidential-ml-utils.svg)](https://badge.fury.io/py/confidential-ml-utils)
[![Python versions](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/confidential-ml-utils)](https://pypi.org/project/confidential-ml-utils/)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![license: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

Confidential ML is the practice of training machine learning models without
seeing the training data. It is needed in many enterprises to satisfy the
strict compliance and privacy guarantees they provide to their customers. This
repository contains a set of utilities for confidential ML, with a special
emphasis on using PyTorch in
[Azure Machine Learning pipelines](https://aka.ms/pl-concept).
 
## Using

Compliant logging library see:
[docs/logging](https://github.com/Azure/confidential-ml-utils/blob/master/docs/logging/README.md)

Minimal use case:

```python
from confidential_ml_utils import DataCategory, enable_confidential_logging, prefix_stack_trace
import logging


@prefix_stack_trace(allow_list=["FileNotFoundError", "TypeError"])
def main():
    enable_confidential_logging()

    log = logging.getLogger(__name__)
    log.info("Hi there", category=DataCategory.PUBLIC)

if __name__ == "__main__":
    main()
```

VS Code Snippets see:
[docs/snippets](https://github.com/Azure/confidential-ml-utils/blob/master/docs/snippets/README.md)

## Contributing

This project welcomes contributions and suggestions. Most contributions require
you to agree to a Contributor License Agreement (CLA) declaring that you have
the right to, and actually do, grant us the rights to use your contribution.
For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether
you need to provide a CLA and decorate the PR appropriately (e.g., status check,
comment). Simply follow the instructions provided by the bot. You will only need
to do this once across all repos using our CLA.

This project has adopted the
[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the
[Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any
additional questions or comments.
