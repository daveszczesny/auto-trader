# Drep
Drep (data grabber) is a program that gathers data, and formulates it


# Pre requisites
Run
```bash
make setup-venv
```

# Use
Run below command for example of usage
```bash
make help
```

## Help
Drep is designed to be edited on a per requirement basis. Adding or removing columns, or indicators or extra data can easily be done via the existing scripts.
<br/>Example
```python

import pandas as pd

df = pd.read_csv('path/to/csv/file')

df = df.dropna()
# etcetera

```
